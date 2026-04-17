#!/usr/bin/env python
"""
Table 2 — Transformation ablation.

Toggles the logit transformation at training (fit.py b/c params) and
decoding (sigmoid-logit reshaping of EAP in logits_processor.py).
All three rows use the full HMM forward-backward.

Rows:
  1. No transformation: classifier fit WITHOUT logit transform + decode WITHOUT
     sigmoid-logit reshaping (``--no_decode_transform``).
  2. Training TF only: classifier fit WITH logit transform (b=10, c=3) + decode
     WITHOUT reshaping.
  3. Training + decoding TF: full TRACE (default).

Usage::

    python -m evaluations.tables.table2_transformation_ablation \\
        --hmm_variant hmm1 --prompts_path data/rtp_10k.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.generation_runner import GenerationRunner
from evaluations.metrics import aggregate_detox_run, parse_scored_csv


def _fit_coefficients(output_path: str, b: float, c: float, data_path: str):
    """Run fit_nontoxicity.py to produce a coefficient file."""
    cmd = [
        sys.executable,
        "-m", "evaluations.classifiers.fit_nontoxicity",
        "--data_path", data_path,
        "--output_path", output_path,
        "--b", str(b),
        "--c", str(c),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Table 2: transformation ablation.")
    parser.add_argument("--hmm_variant", type=str, default="hmm1")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl")
    parser.add_argument("--train_data_path", type=str, default="data/RTP_train.jsonl")
    parser.add_argument("--a", type=float, default=1.0, help="Guidance strength for all rows")
    parser.add_argument("--num_generations", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="results/tables")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = str(PROJECT_ROOT / args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    runner = GenerationRunner(device=args.device)

    # ------------------------------------------------------------------
    # Ensure both coefficient files exist
    # ------------------------------------------------------------------
    coeff_no_tf = str(PROJECT_ROOT / "data" / "coefficients_nontoxicity_notf.csv")
    coeff_with_tf = str(PROJECT_ROOT / "data" / "coefficients_nontoxicity.csv")

    if not os.path.exists(coeff_no_tf):
        print("Fitting classifier WITHOUT logit transform (b=1, c=0)…")
        _fit_coefficients(coeff_no_tf, b=1.0, c=0.0, data_path=args.train_data_path)

    if not os.path.exists(coeff_with_tf):
        print("Fitting classifier WITH logit transform (b=10, c=3)…")
        _fit_coefficients(coeff_with_tf, b=10.0, c=3.0, data_path=args.train_data_path)

    # ------------------------------------------------------------------
    # Three ablation rows
    # ------------------------------------------------------------------
    rows_config = [
        {
            "label": "no_transform",
            "weights_path": coeff_no_tf,
            "no_decode_transform": True,
        },
        {
            "label": "training_tf_only",
            "weights_path": coeff_with_tf,
            "no_decode_transform": True,
        },
        {
            "label": "training_and_decoding_tf",
            "weights_path": coeff_with_tf,
            "no_decode_transform": False,
        },
    ]

    all_results = []

    for cfg in rows_config:
        print(f"\n--- Row: {cfg['label']} ---")
        scored_csv = runner.generate_and_score(
            hmm_variant=args.hmm_variant,
            a=args.a,
            prompts_path=args.prompts_path,
            weights_path=cfg["weights_path"],
            num_generations=args.num_generations,
            no_decode_transform=cfg["no_decode_transform"],
            seed=args.seed,
        )
        df = parse_scored_csv(scored_csv)
        metrics = aggregate_detox_run(df, mode="trace")

        all_results.append({
            "label": cfg["label"],
            "variant": args.hmm_variant,
            "a": args.a,
            "metrics": metrics,
            "weights_path": cfg["weights_path"],
            "no_decode_transform": cfg["no_decode_transform"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "table2_transformation_ablation.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for entry in all_results:
        row = {"label": entry["label"], "variant": entry["variant"], "a": entry["a"]}
        row.update(entry["metrics"])
        rows.append(row)
    csv_path = os.path.join(args.output_dir, "table2_transformation_ablation.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n✓ Table 2 → {json_path}")
    print(f"           → {csv_path}")


if __name__ == "__main__":
    main()
