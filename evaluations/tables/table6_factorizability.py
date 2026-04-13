#!/usr/bin/env python
"""
Table 6 — Factorizability.

Computes cross-entropy loss of factorised (Lasso) vs neural (DistilBERT)
classifier against oracle Detoxify scores, for toxicity and politics.

Pure number-crunching — no generation needed.

Usage::

    python -m evaluations.tables.table6_factorizability
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.metrics import cross_entropy_loss


def _load_factorised_predictions(
    data_path: str,
    coefficients_path: str,
    attribute: str,
) -> tuple:
    """Compute factorised classifier predictions on test data.

    Returns (oracle_scores, factorised_scores) as numpy arrays.
    """
    from transformers import GPT2Tokenizer
    from scipy.sparse import lil_matrix

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

    # Load coefficients
    coeff_df = pd.read_csv(coefficients_path)
    coefficients = coeff_df["Coefficient"].values.astype(float)

    # Load data
    texts, oracle_scores = [], []
    with open(data_path, "r") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["continuation"]["text"])
            oracle_scores.append(float(record["continuation"][attribute]))

    oracle_scores = np.array(oracle_scores)

    # Compute factorised predictions: exp(X @ coefficients)
    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    X = lil_matrix((len(tokenized), len(coefficients)), dtype=float)
    for i, toks in enumerate(tokenized):
        for tid in toks:
            if tid < len(coefficients):
                X[i, tid] += 1
    X = X.tocsr()

    log_preds = X.dot(coefficients)
    factorised_scores = np.exp(log_preds)
    factorised_scores = np.clip(factorised_scores, 1e-15, 1 - 1e-15)

    return oracle_scores, factorised_scores


def _load_neural_predictions(
    data_path: str,
    model_path: str,
    attribute: str,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """Compute neural baseline predictions on test data.

    Returns (oracle_scores, neural_scores) as numpy arrays.
    """
    from transformers import DistilBertTokenizerFast
    from evaluations.classifiers.fit_neural_baseline import DistilBERTRegressor

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    model = DistilBERTRegressor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    texts, oracle_scores = [], []
    with open(data_path, "r") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["continuation"]["text"])
            oracle_scores.append(float(record["continuation"][attribute]))

    oracle_scores = np.array(oracle_scores)

    # Batch predict
    neural_scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Neural predictions"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt",
        )
        with torch.no_grad():
            preds = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        neural_scores.extend(preds.cpu().numpy().tolist())

    neural_scores = np.array(neural_scores)
    neural_scores = np.clip(neural_scores, 1e-15, 1 - 1e-15)

    return oracle_scores, neural_scores


def main():
    parser = argparse.ArgumentParser(description="Table 6: factorizability.")
    parser.add_argument("--data_path", type=str, default="data/RTP_train.jsonl")
    parser.add_argument(
        "--factorised_toxicity", type=str,
        default="data/coefficients_nontoxicity.csv",
    )
    parser.add_argument(
        "--neural_toxicity", type=str,
        default="data/neural_classifier_toxicity.pt",
    )
    parser.add_argument(
        "--factorised_politics", type=str,
        default="data/coefficients_nonpoliticalness.csv",
    )
    parser.add_argument(
        "--neural_politics", type=str,
        default="data/neural_classifier_politics.pt",
    )
    parser.add_argument("--output_dir", type=str, default="evaluations/results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    results = {}

    for attribute, fact_path, neural_path in [
        ("toxicity", args.factorised_toxicity, args.neural_toxicity),
        ("politics", args.factorised_politics, args.neural_politics),
    ]:
        fact_path_abs = fact_path if os.path.isabs(fact_path) else str(PROJECT_ROOT / fact_path)
        neural_path_abs = neural_path if os.path.isabs(neural_path) else str(PROJECT_ROOT / neural_path)
        data_path_abs = args.data_path if os.path.isabs(args.data_path) else str(PROJECT_ROOT / args.data_path)

        if not os.path.exists(fact_path_abs):
            print(f"Skipping {attribute}: factorised model not found at {fact_path_abs}")
            continue
        if not os.path.exists(neural_path_abs):
            print(f"Skipping {attribute}: neural model not found at {neural_path_abs}")
            continue

        print(f"\n--- {attribute} ---")
        oracle_f, factorised = _load_factorised_predictions(
            data_path_abs, fact_path_abs, attribute,
        )
        oracle_n, neural = _load_neural_predictions(
            data_path_abs, neural_path_abs, attribute, device,
        )

        # Convert to log-probs (binary: score vs 1-score)
        log_p_fact = torch.tensor(np.stack([
            np.log(factorised), np.log(1 - factorised),
        ], axis=-1))
        log_p_neural = torch.tensor(np.stack([
            np.log(neural), np.log(1 - neural),
        ], axis=-1))
        log_p_oracle = torch.tensor(np.stack([
            np.log(np.clip(oracle_f, 1e-15, 1 - 1e-15)),
            np.log(np.clip(1 - oracle_f, 1e-15, 1 - 1e-15)),
        ], axis=-1))

        ce_fact_vs_oracle = cross_entropy_loss(log_p_oracle, log_p_fact)
        ce_neural_vs_oracle = cross_entropy_loss(log_p_oracle, log_p_neural)

        results[attribute] = {
            "ce_factorised_vs_oracle": ce_fact_vs_oracle,
            "ce_neural_vs_oracle": ce_neural_vs_oracle,
        }
        print(f"  CE(oracle || factorised) = {ce_fact_vs_oracle:.4f}")
        print(f"  CE(oracle || neural)     = {ce_neural_vs_oracle:.4f}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    output = {
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    json_path = os.path.join(args.output_dir, "table6_factorizability.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    rows = []
    for attr, vals in results.items():
        rows.append({"attribute": attr, **vals})
    csv_path = os.path.join(args.output_dir, "table6_factorizability.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n✓ Table 6 → {json_path}")
    print(f"           → {csv_path}")


if __name__ == "__main__":
    main()
