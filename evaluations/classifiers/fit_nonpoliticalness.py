#!/usr/bin/env python
"""
Fit a nonpoliticalness classifier.

Source data: Misra News Category Dataset (local JSONL at ``data/misra_news.json``).
Labels with ``MoritzLaurer/deberta-v3-base-zeroshot-v2.0`` zero-shot
for "politics", models ``1 - p(politics)``, transforms with
``b=1, c=10`` (the paper specifies ``c=-10`` under a ``b*logit + c``
convention; ``src/fit.py`` uses ``b*(logit - c)``, so the sign flips).

Usage::

    python -m evaluations.classifiers.fit_nonpoliticalness
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
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
        "--data_path", type=str, default="data/misra_news.json",
        help="Local Misra News Category JSONL file (one JSON object per line)",
    )
    parser.add_argument(
        "--output_path", type=str,
        default="classifiers/coefficients_nonpoliticalness.csv",
        help="Output path for coefficients",
    )
    parser.add_argument("--b", type=float, default=1.0, help="Logit transform scaling")
    # Paper specifies c=-10 under b*logit+c convention; src/fit.py uses b*(logit-c), so the sign flips here.
    parser.add_argument("--c", type=float, default=10.0, help="Logit transform shift")
    parser.add_argument("--alpha", type=float, default=1e-6, help="Lasso regularization")
    parser.add_argument("--max_samples", type=int, default=20000,
                        help="Cap total samples (with --balance, ~half political/half not)")
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                        help="Stratified 50/50 sample of POLITICS vs non-POLITICS rows")
    parser.add_argument(
        "--zeroshot_model", type=str,
        default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        help="Zero-shot classification model",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Zero-shot batch size")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    args = parser.parse_args()

    if not os.path.isabs(args.data_path):
        args.data_path = str(PROJECT_ROOT / args.data_path)
    if not os.path.isabs(args.output_path):
        args.output_path = str(PROJECT_ROOT / args.output_path)

    print("Nonpoliticalness Classifier Fitting")
    print("=" * 50)
    print(f"Data: {args.data_path}")
    print(f"Transformation: b={args.b}, c={args.c}")
    print(f"Output: {args.output_path}")
    print()

    # ------------------------------------------------------------------
    # 1. Load Misra News Category Dataset (local JSONL)
    # ------------------------------------------------------------------
    print("Loading Misra News Category Dataset…")
    rows = []
    with open(args.data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  {len(rows)} raw rows")

    # Stratified balanced sampling using gold `category` field
    if args.balance:
        political = [r for r in rows if r.get("category", "").upper() == "POLITICS"]
        nonpolitical = [r for r in rows if r.get("category", "").upper() != "POLITICS"]
        print(f"  POLITICS: {len(political)}  non-POLITICS: {len(nonpolitical)}")

        per_class_cap = (args.max_samples // 2) if args.max_samples else len(political)
        rng = np.random.RandomState(42)

        if len(political) > per_class_cap:
            idx = rng.choice(len(political), per_class_cap, replace=False)
            political = [political[i] for i in idx]
        n_neg = min(len(political), len(nonpolitical), per_class_cap)
        if len(nonpolitical) > n_neg:
            idx = rng.choice(len(nonpolitical), n_neg, replace=False)
            nonpolitical = [nonpolitical[i] for i in idx]
        rows = political + nonpolitical
        rng.shuffle(rows)
        print(f"  Balanced sample: {len(political)} political + {len(nonpolitical)} non-political")
    elif args.max_samples:
        rows = rows[: args.max_samples]

    texts = [
        f"{row['headline']} {row.get('short_description', '')}".strip()
        for row in rows
    ]

    pairs = [(r, t) for r, t in zip(rows, texts) if t]
    rows, texts = [p[0] for p in pairs], [p[1] for p in pairs]
    print(f"  {len(texts)} non-empty texts to score")

    # ------------------------------------------------------------------
    # 2. Score with zero-shot classifier
    # ------------------------------------------------------------------
    print(f"Scoring with {args.zeroshot_model}…")
    if args.device is None:
        device_id = 0 if torch.cuda.is_available() else -1
    elif args.device == "cpu":
        device_id = -1
    else:
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0

    classifier = pipeline(
        "zero-shot-classification",
        model=args.zeroshot_model,
        device=device_id,
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

    # Cache political scores for fit_neural_baseline.py reuse
    cache_path = str(PROJECT_ROOT / "classifiers/misra_news_political_scores.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "texts": texts,
            "scores": political_scores.tolist(),
            "model": args.zeroshot_model,
        }, f)
    print(f"  Cached political scores → {cache_path}")

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

    # Sanity checks: catch sign-of-c collapse bug
    print(f"Transformed targets: min={transformed.min():.6f}, max={transformed.max():.6f}, mean={transformed.mean():.6f}")
    print(f"Log targets: min={log_targets.min():.4f}, max={log_targets.max():.4f}")
    print(f"Log target percentiles (1/25/50/75/99): {np.percentile(log_targets, [1, 25, 50, 75, 99])}")
    if log_targets.max() - log_targets.min() < 1.0:
        raise ValueError(
            "Log targets span less than 1 unit — the transform is collapsing all targets. "
            "Check the sign of --c (should be positive 10 in this codebase's convention)."
        )

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
