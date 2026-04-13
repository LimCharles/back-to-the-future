#!/usr/bin/env python
"""
HMM quality vs toxicity plot.

For each checkpoint in ``--checkpoints_dir``: reads ``metadata.json``
for validation log-likelihood (warns and skips LL axis if absent), runs
a small detox eval, records ``(step, val_ll, avg_max_tox)``.

Twin-axis seaborn lineplot per variant + a combined overlay.

Usage::

    python -m evaluations.plots.plot_hmm_quality_vs_toxicity \\
        --checkpoints_dir models/checkpoints/ --hmm_variant hmm1
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.generation_runner import GenerationRunner
from evaluations.metrics import aggregate_detox_run, parse_scored_csv


def _find_checkpoints(checkpoints_dir: str) -> list:
    """Find checkpoint directories sorted by step number."""
    ckpts = []
    for entry in os.listdir(checkpoints_dir):
        full = os.path.join(checkpoints_dir, entry)
        if not os.path.isdir(full):
            continue
        # Try to extract step number from directory name
        meta_path = os.path.join(full, "metadata.json")
        step = None
        val_ll = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            step = meta.get("step", meta.get("global_step"))
            val_ll = meta.get("val_ll", meta.get("val_log_likelihood"))
        if step is None:
            # Try extracting from directory name
            import re
            m = re.search(r"(\d+)", entry)
            if m:
                step = int(m.group(1))
        if step is not None:
            ckpts.append({"path": full, "step": step, "val_ll": val_ll})

    return sorted(ckpts, key=lambda x: x["step"])


def main():
    parser = argparse.ArgumentParser(
        description="Plot HMM quality vs toxicity."
    )
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--hmm_variant", type=str, default="hmm1")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument("--weights_path", type=str, default="data/coefficients.csv")
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = _find_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoints_dir}", file=sys.stderr)
        sys.exit(1)

    has_val_ll = any(c["val_ll"] is not None for c in checkpoints)
    if not has_val_ll:
        print("WARNING: No metadata.json with val_ll found. LL axis will be omitted.",
              file=sys.stderr)

    runner = GenerationRunner(device=args.device)

    records = []
    for ckpt in tqdm(checkpoints, desc="Checkpoint sweep"):
        scored_csv = runner.generate_and_score(
            hmm_variant=args.hmm_variant,
            a=args.a,
            prompts_path=args.prompts_path,
            weights_path=args.weights_path,
            hmm_model_path=ckpt["path"],
            num_generations=args.num_generations,
        )
        df = parse_scored_csv(scored_csv)
        metrics = aggregate_detox_run(df)
        records.append({
            "step": ckpt["step"],
            "val_ll": ckpt["val_ll"],
            "avg_max_tox": metrics["avg_max_tox"],
            "variant": args.hmm_variant,
        })

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Plot: per-variant twin-axis
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_tox = "tab:red"
    color_ll = "tab:blue"

    sns.lineplot(data=df, x="step", y="avg_max_tox", ax=ax1, color=color_tox, marker="o")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Avg Max Toxicity", color=color_tox)
    ax1.tick_params(axis="y", labelcolor=color_tox)

    if has_val_ll:
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x="step", y="val_ll", ax=ax2, color=color_ll, marker="s")
        ax2.set_ylabel("Validation Log-Likelihood", color=color_ll)
        ax2.tick_params(axis="y", labelcolor=color_ll)

    ax1.set_title(f"HMM Quality vs Toxicity — {args.hmm_variant}")
    fig.tight_layout()

    variant_path = os.path.join(
        args.output_dir,
        f"hmm_quality_vs_toxicity_{args.hmm_variant}.png",
    )
    fig.savefig(variant_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Save data JSON
    json_path = os.path.join(
        PROJECT_ROOT, "evaluations", "results",
        f"hmm_quality_vs_toxicity_{args.hmm_variant}.json",
    )
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"✓ Per-variant plot → {variant_path}")
    print(f"  Data → {json_path}")


if __name__ == "__main__":
    main()
