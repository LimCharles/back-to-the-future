#!/usr/bin/env python
"""
Table 8 — LM-as-judge evaluation.

For each prompt, takes the 25 continuations and runs
``LocalJudge.score_set``.  Reports mean nontoxicity / fluency /
diversity per variant.

**This is the slowest script** — it is fully resumable via checkpoint
files.

Usage::

    python -m evaluations.tables.table8_lm_judge \\
        --scored_csv results/evaluation/detox_hmm1_a1.0_scored.csv \\
        --judge_model meta-llama/Llama-3.3-70B-Instruct
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.judge import LocalJudge
from evaluations.metrics import extract_continuations, parse_scored_csv


def main():
    parser = argparse.ArgumentParser(description="Table 8: LM-as-judge evaluation.")
    parser.add_argument("--scored_csv", type=str, required=True,
                        help="Scored CSV file from table1 or similar")
    parser.add_argument("--hmm_variant", type=str, required=True)
    parser.add_argument(
        "--judge_model", type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HuggingFace model ID for the judge",
    )
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/tables")
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isabs(args.scored_csv):
        args.scored_csv = str(PROJECT_ROOT / args.scored_csv)

    if not os.path.exists(args.scored_csv):
        raise FileNotFoundError(
            f"Scored CSV not found: {args.scored_csv}. "
            "Run table1_detoxification.py first."
        )

    # ------------------------------------------------------------------
    # 1. Parse scored CSV
    # ------------------------------------------------------------------
    df = parse_scored_csv(args.scored_csv)
    continuations = extract_continuations(df, mode="trace")
    prompts = df["prefix"].tolist()

    # Validate variant is in filename
    if args.hmm_variant not in os.path.basename(args.scored_csv):
        print(
            f"WARNING: --hmm_variant={args.hmm_variant} not found in "
            f"filename {os.path.basename(args.scored_csv)}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 2. Build judge items
    # ------------------------------------------------------------------
    items = []
    for prompt, conts in zip(prompts, continuations):
        if not conts:
            continue
        items.append({"prompt": prompt, "continuations": conts})

    print(f"Judging {len(items)} prompts with {args.judge_model}…")

    # ------------------------------------------------------------------
    # 3. Score with LocalJudge
    # ------------------------------------------------------------------
    judge = LocalJudge(model_id=args.judge_model)

    checkpoint_path = os.path.join(
        args.output_dir,
        f"table8_{args.hmm_variant}_checkpoint.json",
    )
    results = judge.score_batch(
        items,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )

    # ------------------------------------------------------------------
    # 4. Aggregate
    # ------------------------------------------------------------------
    nontox, fluency, diversity = [], [], []
    for r in results:
        if r is None:
            continue
        if "nontoxicity" in r:
            nontox.append(r["nontoxicity"])
        if "fluency" in r:
            fluency.append(r["fluency"])
        if "diversity" in r:
            diversity.append(r["diversity"])

    aggregated = {
        "mean_nontoxicity": float(np.mean(nontox)) if nontox else None,
        "mean_fluency": float(np.mean(fluency)) if fluency else None,
        "mean_diversity": float(np.mean(diversity)) if diversity else None,
        "n_scored": len([r for r in results if r is not None]),
        "n_failed": len([r for r in results if r is None]),
    }

    print(f"\nResults for {args.hmm_variant}:")
    for k, v in aggregated.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    output = {
        "variant": args.hmm_variant,
        "judge_model": args.judge_model,
        "scored_csv": args.scored_csv,
        "aggregated": aggregated,
        "per_prompt": [
            {"prompt": items[i]["prompt"], "scores": results[i]}
            for i in range(len(items))
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    json_path = os.path.join(args.output_dir, f"table8_lm_judge_{args.hmm_variant}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = os.path.join(args.output_dir, f"table8_lm_judge_{args.hmm_variant}.csv")
    pd.DataFrame([{"variant": args.hmm_variant, **aggregated}]).to_csv(csv_path, index=False)

    print(f"\n✓ Table 8 → {json_path}")
    print(f"           → {csv_path}")


if __name__ == "__main__":
    main()
