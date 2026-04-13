#!/usr/bin/env python
"""
Table 1 — Full RTP detoxification evaluation.

Sweeps ``--hmm_variant`` and ``--a`` values, reports avg max toxicity,
prob>0.5, dist-2, dist-3, perplexity per variant.

Resumable: checks for existing scored CSVs before generating.

Usage::

    python -m evaluations.tables.table1_detoxification \\
        --prompts_path data/rtp_10k.jsonl \\
        --variants hmm1 --a_values 1.0,1.2,1.4,1.6,1.8
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.generation_runner import GenerationRunner
from evaluations.metrics import aggregate_detox_run, parse_scored_csv
from evaluations.csv_schema import validate_scored_csv, SchemaError


def _aggregate_mode(df, mode: str) -> dict:
    """Compute per-mode aggregates directly from parsed gen cells.

    ``src/score.py`` writes a single ``max_toxicity`` column that mixes
    trace and baseline gens. When evaluating a comparison-mode CSV we
    need per-mode numbers, so we recompute from the JSON cells.
    """
    prefix = f"{mode}_gen_"
    cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not cols:
        raise ValueError(f"No '{mode}_gen_*' columns in scored CSV")

    max_tox_per_row, any_tox_per_row, flu_per_row = [], [], []
    d1_per_row, d2_per_row, d3_per_row = [], [], []
    from evaluations.metrics import compute_distinct_n
    for _, row in df.iterrows():
        toxes, flus, conts = [], [], []
        for c in cols:
            cell = row[c]
            if not isinstance(cell, dict):
                continue
            t = cell.get("toxicity")
            f = cell.get("fluency")
            if t is not None:
                toxes.append(float(t))
            if isinstance(f, (int, float)):
                flus.append(float(f))
            conts.append(str(cell.get("continuation", "")))
        if not toxes:
            continue
        max_tox_per_row.append(max(toxes))
        any_tox_per_row.append(1.0 if max(toxes) > 0.5 else 0.0)
        if flus:
            flu_per_row.append(sum(flus) / len(flus))
        d1_per_row.append(compute_distinct_n(conts, 1))
        d2_per_row.append(compute_distinct_n(conts, 2))
        d3_per_row.append(compute_distinct_n(conts, 3))

    import statistics
    return {
        "avg_max_tox": statistics.fmean(max_tox_per_row),
        "prob_tox_gt_0.5": statistics.fmean(any_tox_per_row),
        "mean_perplexity": statistics.fmean(flu_per_row) if flu_per_row else float("nan"),
        "dist1": statistics.fmean(d1_per_row),
        "dist2": statistics.fmean(d2_per_row),
        "dist3": statistics.fmean(d3_per_row),
        "num_prompts": len(max_tox_per_row),
        "num_generations_per_prompt": len(cols),
    }


def main():
    parser = argparse.ArgumentParser(description="Table 1: detoxification evaluation.")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument("--variants", type=str, default="hmm1",
                        help="Comma-separated HMM variants to evaluate")
    parser.add_argument("--a_values", type=str, default="1.0,1.2,1.4,1.6,1.8",
                        help="Comma-separated guidance strengths")
    parser.add_argument("--num_generations", type=int, default=25)
    parser.add_argument("--weights_path", type=str, default="data/coefficients.csv")
    parser.add_argument("--baseline", action="store_true",
                        help="Also generate baseline (no HMM) for comparison")
    parser.add_argument("--output_dir", type=str, default="evaluations/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scored_csv_override", type=str, default=None,
        help="Path to a pre-scored CSV. When set, skips generation/scoring "
             "and aggregates per-mode (trace+baseline) directly from the file. "
             "Intended for evaluating checked-in generations.",
    )
    args = parser.parse_args()

    if args.scored_csv_override:
        csv_path = args.scored_csv_override
        if not os.path.isabs(csv_path):
            csv_path = str(PROJECT_ROOT / csv_path)
        try:
            validate_scored_csv(csv_path)
        except (FileNotFoundError, SchemaError) as exc:
            print(f"Invalid scored CSV: {exc}", file=sys.stderr)
            sys.exit(1)

        variants = [v.strip() for v in args.variants.split(",")]
        a_values = [float(v.strip()) for v in args.a_values.split(",")]
        if len(variants) != 1 or len(a_values) != 1:
            print("--scored_csv_override requires exactly one --variants and one --a_values",
                  file=sys.stderr)
            sys.exit(1)
        variant, a = variants[0], a_values[0]

        if not os.path.isabs(args.output_dir):
            args.output_dir = str(PROJECT_ROOT / args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        df = parse_scored_csv(csv_path)
        all_results = []
        for mode, label_a in [("trace", a), ("baseline", 0.0)]:
            cols = [c for c in df.columns if c.startswith(f"{mode}_gen_")]
            if not cols:
                continue
            metrics = _aggregate_mode(df, mode)
            n_gens = metrics["num_generations_per_prompt"]
            degraded = []
            if n_gens < 5:
                degraded = ["dist1", "dist2", "dist3", "prob_tox_gt_0.5"]
            entry = {
                "variant": variant if mode == "trace" else f"{variant}_baseline",
                "mode": mode,
                "a": label_a,
                "metrics": metrics,
                "num_generations": n_gens,
                "degraded_metrics": degraded,
                "source_csv": csv_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            all_results.append(entry)

        json_path = os.path.join(args.output_dir, "table1_detoxification.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        rows = []
        for entry in all_results:
            row = {"variant": entry["variant"], "mode": entry["mode"], "a": entry["a"]}
            row.update(entry["metrics"])
            rows.append(row)
        csv_out = os.path.join(args.output_dir, "table1_detoxification.csv")
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"\n✓ Table 1 (override) → {json_path}")
        print(f"                      → {csv_out}")
        return

    variants = [v.strip() for v in args.variants.split(",")]
    a_values = [float(v.strip()) for v in args.a_values.split(",")]

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    runner = GenerationRunner(device=args.device)
    all_results = []

    total = len(variants) * len(a_values)
    with tqdm(total=total, desc="Table 1", file=sys.stderr) as pbar:
        for variant in variants:
            for a in a_values:
                pbar.set_postfix(variant=variant, a=a)

                # Check if scored CSV already exists (resumability)
                scored_csv = str(
                    PROJECT_ROOT / f"results/detox_{variant}_a{a}_scored.csv"
                )
                if not os.path.exists(scored_csv):
                    gen_csv = runner.generate(
                        hmm_variant=variant,
                        a=a,
                        prompts_path=args.prompts_path,
                        weights_path=args.weights_path,
                        num_generations=args.num_generations,
                        baseline=args.baseline,
                        seed=args.seed,
                    )
                    scored_csv = runner.score(gen_csv)

                # Aggregate metrics
                df = parse_scored_csv(scored_csv)
                metrics = aggregate_detox_run(df, mode="trace")

                entry = {
                    "variant": variant,
                    "a": a,
                    "metrics": metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "params": {
                        "prompts_path": args.prompts_path,
                        "weights_path": args.weights_path,
                        "num_generations": args.num_generations,
                        "seed": args.seed,
                    },
                }
                all_results.append(entry)
                pbar.update(1)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "table1_detoxification.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Flatten for CSV
    rows = []
    for entry in all_results:
        row = {"variant": entry["variant"], "a": entry["a"]}
        row.update(entry["metrics"])
        rows.append(row)

    csv_path = os.path.join(args.output_dir, "table1_detoxification.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n✓ Table 1 → {json_path}")
    print(f"           → {csv_path}")


if __name__ == "__main__":
    main()
