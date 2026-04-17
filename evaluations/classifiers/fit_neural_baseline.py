#!/usr/bin/env python
"""
Train a neural classifier (DistilBERT fine-tune) as a baseline for Table 6.

Used to compare cross-entropy loss of factorised (Lasso) vs neural classifiers.

Usage::

    python -m evaluations.classifiers.fit_neural_baseline --attribute toxicity
    python -m evaluations.classifiers.fit_neural_baseline --attribute politics
"""
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TextScoreDataset(Dataset):
    """Dataset of (text, score) pairs for regression."""

    def __init__(self, texts, scores, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.scores[idx]
        return item


class DistilBERTRegressor(nn.Module):
    """DistilBERT with a single regression head."""

    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        from transformers import DistilBertModel

        self.bert = DistilBertModel.from_pretrained(model_name)
        self.head = nn.Linear(self.bert.config.dim, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        logit = self.head(cls_output).squeeze(-1)
        return torch.sigmoid(logit)


def log_mse_loss(preds, targets, eps=1e-7):
    """Log-MSE loss matching the factorised classifier's objective (Section 4.3)."""
    preds = torch.clamp(preds, eps, 1 - eps)
    targets = torch.clamp(targets, eps, 1 - eps)
    return ((torch.log(preds) - torch.log(targets)) ** 2).mean()


def main():
    parser = argparse.ArgumentParser(
        description="Train neural baseline classifier for Table 6."
    )
    parser.add_argument(
        "--data_path", type=str, default="data/RTP_train.jsonl",
        help="Training data JSONL",
    )
    parser.add_argument(
        "--attribute", type=str, default="toxicity",
        choices=["toxicity", "politics"],
        help="Attribute to predict",
    )
    parser.add_argument(
        "--output_dir", type=str, default="classifiers/",
        help="Directory for model checkpoint and metrics JSON",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--politics_data_path", type=str, default=None,
        help="Misra news JSONL (defaults to data/misra_news.json)",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.data_path):
        args.data_path = str(PROJECT_ROOT / args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Neural Baseline: {args.attribute}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.attribute == "toxicity":
        texts, scores = [], []
        with open(args.data_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record["continuation"]["text"]
                score = float(record["continuation"]["toxicity"])
                texts.append(text)
                scores.append(score)
    elif args.attribute == "politics":
        # Politics labels come from Misra news scored by zero-shot DeBERTa.
        # fit_nonpoliticalness.py writes the cache as a side effect.
        politics_scores_cache = str(PROJECT_ROOT / "classifiers/misra_news_political_scores.json")
        if os.path.exists(politics_scores_cache):
            with open(politics_scores_cache) as f:
                cached = json.load(f)
            texts, scores = cached["texts"], cached["scores"]
        else:
            raise FileNotFoundError(
                f"Politics scores cache not found at {politics_scores_cache}. "
                f"Run fit_nonpoliticalness.py first — it writes this cache "
                f"as a side effect."
            )
    else:
        raise ValueError(f"Unknown attribute: {args.attribute}")
    print(f"Loaded {len(texts)} samples")

    # ------------------------------------------------------------------
    # 2. Tokenise and split
    # ------------------------------------------------------------------
    from transformers import DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = TextScoreDataset(texts, scores, tokenizer, max_length=args.max_length)

    val_size = int(len(dataset) * args.val_frac)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    model = DistilBERTRegressor().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = log_mse_loss

    best_val_loss = float("inf")
    metrics_log = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(preds, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(labels)

        train_loss /= train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                val_loss += loss_fn(preds, labels).item() * len(labels)
        val_loss /= val_size

        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(
                args.output_dir, f"neural_classifier_{args.attribute}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)

    # Reload best checkpoint so the in-memory model is the best one
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # ------------------------------------------------------------------
    # 4. Save metrics
    # ------------------------------------------------------------------
    metrics_path = os.path.join(
        args.output_dir, f"neural_classifier_{args.attribute}_metrics.json"
    )
    with open(metrics_path, "w") as f:
        json.dump({
            "attribute": args.attribute,
            "epochs": args.epochs,
            "best_val_loss": best_val_loss,
            "history": metrics_log,
        }, f, indent=2)

    print(f"\n✓ Model saved to {ckpt_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
