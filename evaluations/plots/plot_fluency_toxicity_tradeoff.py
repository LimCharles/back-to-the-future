#!/usr/bin/env python
"""
Fluency–toxicity tradeoff plot (Figure 3 in the paper).

Loads ``results/table1_detoxification.json`` (and a sweep over ``--a``).
Scatter+line: x = avg max toxicity, y = perplexity, hue = variant,
point labels = ``a`` value.

Usage::

    python -m evaluations.plots.plot_fluency_toxicity_tradeoff
    python -m evaluations.plots.plot_fluency_toxicity_tradeoff \\
        --input results/table1_detoxification.json
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot fluency-toxicity tradeoff (Figure 3)."
    )
    parser.add_argument(
        "--input", type=str,
        default="results/tables/table1_detoxification.json",
    )
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    args = parser.parse_args()

    if not os.path.isabs(args.input):
        args.input = str(PROJECT_ROOT / args.input)
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, "r") as f:
        data = json.load(f)

    # Build dataframe
    rows = []
    for entry in data:
        rows.append({
            "variant": entry["variant"],
            "a": entry["a"],
            "avg_max_tox": entry["metrics"]["avg_max_tox"],
            "mean_perplexity": entry["metrics"]["mean_perplexity"],
        })
    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=df, x="avg_max_tox", y="mean_perplexity",
        hue="variant", style="variant", s=80, ax=ax,
    )

    # Connect points per variant with lines (ordered by a)
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant].sort_values("a")
        ax.plot(
            sub["avg_max_tox"], sub["mean_perplexity"],
            alpha=0.5, linewidth=1,
        )

    # Label each point with its a value
    for _, row in df.iterrows():
        ax.annotate(
            f"a={row['a']}",
            (row["avg_max_tox"], row["mean_perplexity"]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, alpha=0.8,
        )

    ax.set_xlabel("Avg Max Toxicity")
    ax.set_ylabel("Mean Perplexity")
    ax.set_title("Fluency–Toxicity Tradeoff")
    ax.legend(title="Variant")

    fig.tight_layout()
    out_path = os.path.join(
        args.output_dir, f"fluency_toxicity_tradeoff.{args.format}"
    )
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Save companion JSON
    json_path = os.path.join(
        PROJECT_ROOT, "evaluations", "results",
        "fluency_toxicity_tradeoff_data.json",
    )
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"✓ Plot → {out_path}")
    print(f"  Data → {json_path}")


if __name__ == "__main__":
    main()
