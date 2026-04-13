#!/usr/bin/env python
"""
Role quality scatter plot.

For all 76 RoleBench characters: generate with prompting baseline
(instruction-only) and with TRACE+role classifier, score each set
with the character's own classifier (mean probability), scatter
prompting (x) vs TRACE (y), ``hue=variant``, with ``y=x`` reference
line.

Usage::

    python -m evaluations.plots.plot_role_quality_scatter \\
        --role_results evaluations/results/role_eval.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.generation_runner import GenerationRunner
from evaluations.metrics import parse_scored_csv


def _evaluate_role(
    character: str,
    coefficients_path: str,
    scored_csv: str,
) -> float:
    """Score a set of generations with a character's own classifier.

    Returns mean exp(X @ coefficients) across all generations.
    """
    import numpy as np
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    coeff_df = pd.read_csv(coefficients_path)
    coefficients = coeff_df["Coefficient"].values.astype(float)

    df = parse_scored_csv(scored_csv)
    gen_cols = [c for c in df.columns if c.startswith("trace_gen_")]

    scores = []
    for _, row in df.iterrows():
        for col in gen_cols:
            cell = row[col]
            if cell is None:
                continue
            text = cell.get("continuation", "") if isinstance(cell, dict) else ""
            if not text.strip():
                continue
            toks = tokenizer.encode(text, add_special_tokens=False)
            log_pred = sum(coefficients[t] for t in toks if t < len(coefficients))
            scores.append(float(np.exp(log_pred)))

    return float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Plot role quality scatter."
    )
    parser.add_argument(
        "--role_results", type=str, default=None,
        help="Pre-computed role evaluation JSON. If not provided, "
             "generates and evaluates from scratch.",
    )
    parser.add_argument(
        "--coefficients_dir", type=str, default="data/coefficients_role/",
        help="Directory with per-character coefficient CSVs",
    )
    parser.add_argument("--hmm_variant", type=str, default="hmm1")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="evaluations/figures")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.role_results and os.path.exists(args.role_results):
        with open(args.role_results, "r") as f:
            records = json.load(f)
    else:
        # Load manifest
        manifest_path = os.path.join(
            PROJECT_ROOT if os.path.isabs(args.coefficients_dir) else "",
            args.coefficients_dir if os.path.isabs(args.coefficients_dir) else str(PROJECT_ROOT / args.coefficients_dir),
            "manifest.json",
        )
        if not os.path.exists(manifest_path):
            print(
                f"Manifest not found at {manifest_path}. "
                "Run fit_role.py first, or provide --role_results.",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        print(f"Evaluating {len(manifest)} characters…")
        records = []
        # This would require generating for each character with both prompting
        # and TRACE approaches — placeholder for when role eval data exists.
        print(
            "NOTE: Full role evaluation requires generating with each character's "
            "classifier and a prompting baseline. Provide --role_results with "
            "pre-computed results.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.scatterplot(
        data=df, x="prompting_score", y="trace_score",
        hue="variant", s=40, alpha=0.7, ax=ax,
    )

    # y=x reference line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Prompting Baseline Quality")
    ax.set_ylabel("TRACE Quality")
    ax.set_title("Role-Play Quality: Prompting vs TRACE")
    ax.legend(title="Variant")

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "role_quality_scatter.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Companion JSON
    json_path = os.path.join(
        PROJECT_ROOT, "evaluations", "results",
        "role_quality_scatter_data.json",
    )
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"✓ Plot → {out_path}")
    print(f"  Data → {json_path}")


if __name__ == "__main__":
    main()
