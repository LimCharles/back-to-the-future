#!/usr/bin/env python
"""
Transformation distribution plots.

Top panel: three histograms — raw Detoxify scores on RTP train,
logit-transformed targets (b=10, c=3), fitted classifier predictions.
``histplot`` with ``hue`` and ``kde=True``.

Bottom panel: first-step EAP histogram before/after the ``--a``
transform, hued by variant.  Requires ``.npz`` EAP dump files
produced by ``generate.py --dump_eap_path``.

Usage::

    python -m evaluations.plots.plot_transformation_distributions \\
        --train_data data/RTP_train.jsonl \\
        --coefficients data/coefficients_nontoxicity.csv \\
        --eap_dumps results/eap_hmm1.npz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logit, expit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Plot transformation distributions."
    )
    parser.add_argument("--train_data", type=str, default="data/RTP_train.jsonl")
    parser.add_argument("--coefficients", type=str, default="classifiers/coefficients_nontoxicity.csv")
    parser.add_argument(
        "--eap_dumps", type=str, nargs="*", default=[],
        help="One or more .npz EAP dump files (from --dump_eap_path)",
    )
    parser.add_argument(
        "--eap_labels", type=str, nargs="*", default=[],
        help="Labels for each EAP dump (e.g., hmm1 hmm2)",
    )
    parser.add_argument("--b", type=float, default=10.0)
    parser.add_argument("--c", type=float, default=3.0)
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    for attr in ("train_data", "coefficients", "output_dir"):
        p = getattr(args, attr)
        if not os.path.isabs(p):
            setattr(args, attr, str(PROJECT_ROOT / p))
    os.makedirs(args.output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    has_eap = bool(args.eap_dumps)
    nrows = 2 if has_eap else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 5 * nrows))
    if nrows == 1:
        axes = [axes]

    # ------------------------------------------------------------------
    # Top panel: score distributions
    # ------------------------------------------------------------------
    # Load raw scores
    scores = []
    with open(args.train_data, "r") as f:
        for line in f:
            record = json.loads(line)
            scores.append(float(record["continuation"]["toxicity"]))
    raw_scores = np.array(scores)

    # Inverted (nontoxicity)
    inverted = 1 - raw_scores

    # Logit transform
    eps = 1e-15
    clipped = np.clip(inverted, eps, 1 - eps)
    logit_vals = logit(clipped)
    modified = args.b * (logit_vals - args.c)
    transformed = expit(modified)

    # Fitted predictions (from coefficients)
    coeff_df = pd.read_csv(args.coefficients)
    coefficients = coeff_df["Coefficient"].values.astype(float)

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

    fitted_scores = []
    with open(args.train_data, "r") as f:
        for line in f:
            record = json.loads(line)
            text = record["continuation"]["text"]
            toks = tokenizer.encode(text, add_special_tokens=False)
            log_pred = sum(coefficients[t] for t in toks if t < len(coefficients))
            fitted_scores.append(np.exp(log_pred))
    fitted = np.clip(np.array(fitted_scores), 0, 1)

    ax = axes[0]
    hist_df = pd.DataFrame({
        "score": np.concatenate([inverted, transformed, fitted]),
        "Distribution": (
            ["Raw (1−toxicity)"] * len(inverted)
            + [f"Logit-transformed (b={args.b}, c={args.c})"] * len(transformed)
            + ["Fitted predictions"] * len(fitted)
        ),
    })
    sns.histplot(
        data=hist_df, x="score", hue="Distribution",
        kde=True, stat="density", common_norm=False,
        alpha=0.4, ax=ax, bins=100,
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions: Raw → Transformed → Fitted")

    # ------------------------------------------------------------------
    # Bottom panel: EAP distributions (if dumps provided)
    # ------------------------------------------------------------------
    if has_eap:
        ax = axes[1]
        eap_frames = []
        labels = args.eap_labels if args.eap_labels else [
            f"dump_{i}" for i in range(len(args.eap_dumps))
        ]

        for dump_path, label in zip(args.eap_dumps, labels):
            dump_path_abs = dump_path if os.path.isabs(dump_path) else str(PROJECT_ROOT / dump_path)
            data = np.load(dump_path_abs)
            # Take first batch element, flatten
            pre = data["eap_pre_transform"][0]
            post = data["eap_post_transform"][0]

            # Sample top-k tokens for visibility
            top_k = min(5000, len(pre))
            top_idx = np.argsort(-np.abs(pre - 0.5))[:top_k]

            eap_frames.append(pd.DataFrame({
                "EAP": np.concatenate([pre[top_idx], post[top_idx]]),
                "Stage": ["Pre-transform"] * top_k + ["Post-transform"] * top_k,
                "Variant": label,
            }))

        eap_df = pd.concat(eap_frames, ignore_index=True)
        sns.histplot(
            data=eap_df, x="EAP", hue="Stage", col="Variant" if len(labels) > 1 else None,
            kde=True, stat="density", common_norm=False,
            alpha=0.4, ax=ax, bins=100,
        )
        ax.set_xlabel("EAP Value")
        ax.set_ylabel("Density")
        ax.set_title("First-Step EAP: Pre vs Post Transform")

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "transformation_distributions.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Companion JSON
    json_path = os.path.join(
        PROJECT_ROOT, "evaluations", "results",
        "transformation_distributions_data.json",
    )
    with open(json_path, "w") as f:
        json.dump({
            "raw_mean": float(inverted.mean()),
            "transformed_mean": float(transformed.mean()),
            "fitted_mean": float(fitted.mean()),
            "n_samples": len(raw_scores),
        }, f, indent=2)

    print(f"✓ Plot → {out_path}")
    print(f"  Data → {json_path}")


if __name__ == "__main__":
    main()
