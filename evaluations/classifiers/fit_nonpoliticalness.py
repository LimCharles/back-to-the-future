#!/usr/bin/env python
"""
Fit a nonpoliticalness classifier.

Source data: News Category Dataset (``Misra/News-Category-Dataset``).
Labels with ``MoritzLaurer/deberta-v3-base-zeroshot-v2.0`` zero-shot
for "politics", models ``1 - p(politics)``, transforms with
``b=1, c=-10``.

Usage::

    python -m evaluations.classifiers.fit_nonpoliticalness
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.special import logit, expit
from scipy.sparse import lil_matrix
from sklearn.linear_model import Lasso
from transformers import GPT2Tokenizer, pipeline
from tqdm import tqdm

from src.fit import fit_lasso_model, save_coefficients


def main():
    parser = argparse.ArgumentParser(
        description="Fit nonpoliticalness classifier."
    )
    parser.add_argument(
        "--output_path", type=str,
        default="data/coefficients_nonpoliticalness.csv",
        help="Output path for coefficients",
    )
    parser.add_argument("--b", type=float, default=1.0, help="Logit transform scaling")
    parser.add_argument("--c", type=float, default=-10.0, help="Logit transform shift")
    parser.add_argument("--alpha", type=float, default=1e-6, help="Lasso regularization")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument(
        "--zeroshot_model", type=str,
        default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        help="Zero-shot classification model",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Zero-shot batch size")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    args = parser.parse_args()

    if not os.path.isabs(args.output_path):
        args.output_path = str(PROJECT_ROOT / args.output_path)

    print("Nonpoliticalness Classifier Fitting")
    print("=" * 50)
    print(f"Transformation: b={args.b}, c={args.c}")
    print(f"Output: {args.output_path}")
    print()

    # ------------------------------------------------------------------
    # 1. Load News Category Dataset
    # ------------------------------------------------------------------
    print("Loading News Category Dataset…")
    from datasets import load_dataset

    ds = load_dataset("Misra/News-Category-Dataset", split="train")
    texts = [
        f"{row['headline']} {row.get('short_description', '')}".strip()
        for row in ds
    ]
    if args.max_samples:
        texts = texts[: args.max_samples]
    print(f"  {len(texts)} texts loaded")

    # ------------------------------------------------------------------
    # 2. Score with zero-shot classifier
    # ------------------------------------------------------------------
    print(f"Scoring with {args.zeroshot_model}…")
    device_id = 0 if args.device is None else int(args.device.split(":")[-1]) if ":" in (args.device or "") else -1
    classifier = pipeline(
        "zero-shot-classification",
        model=args.zeroshot_model,
        device=device_id if device_id >= 0 else -1,
    )

    candidate_labels = ["politics", "not politics"]
    political_scores = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Zero-shot scoring"):
        batch = texts[i : i + args.batch_size]
        results = classifier(batch, candidate_labels=candidate_labels)
        if not isinstance(results, list):
            results = [results]
        for res in results:
            idx = res["labels"].index("politics")
            political_scores.append(res["scores"][idx])

    political_scores = np.array(political_scores)
    # Nonpoliticalness = 1 - p(politics)
    nonpolitical_scores = 1.0 - political_scores
    print(f"  Mean nonpoliticalness: {nonpolitical_scores.mean():.4f}")

    # ------------------------------------------------------------------
    # 3. Preprocess and fit
    # ------------------------------------------------------------------
    print("Fitting classifier…")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    vocab_size = tokenizer.vocab_size

    # Logit transform
    eps = 1e-15
    clipped = np.clip(nonpolitical_scores, eps, 1 - eps)
    logit_scores = logit(clipped)
    modified = args.b * (logit_scores - args.c)
    transformed = expit(modified)
    clipped_tf = np.clip(transformed, eps, 1 - eps)
    log_targets = np.log(clipped_tf)

    # Token count matrix
    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    unique_ids = sorted(set(tid for toks in tokenized for tid in toks))
    tid_to_idx = {tid: idx for idx, tid in enumerate(unique_ids)}

    X = lil_matrix((len(tokenized), len(unique_ids)), dtype=int)
    for i, toks in enumerate(tokenized):
        for tid in toks:
            if tid in tid_to_idx:
                X[i, tid_to_idx[tid]] += 1
    X = X.tocsr()

    coefficients, fitted_log_scores = fit_lasso_model(X, log_targets, args.alpha)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_coefficients(coefficients, tid_to_idx, vocab_size, args.output_path)

    print(f"\n✓ Nonpoliticalness coefficients saved to {args.output_path}")


if __name__ == "__main__":
    main()
