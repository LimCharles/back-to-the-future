#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --output=/home/c/caplim/back-to-the-future/output/analyze.out
#SBATCH --error=/home/c/caplim/back-to-the-future/output/analyze.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail

nvidia-smi
source /home/c/caplim/back-to-the-future/.venv/bin/activate
cd /home/c/caplim/back-to-the-future

mkdir -p results/tables results/figures

A=1.0

# ------------------------------------------------------------------
# Table 1: aggregate per-variant (trace + baseline) from scored CSVs.
# Merges all three variants into one results/tables/table1_detoxification.json
# ------------------------------------------------------------------
python <<'PY'
import json, os, sys, statistics
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/home/c/caplim/back-to-the-future")
sys.path.insert(0, str(ROOT))

from evaluations.tables.table1_detoxification import _aggregate_mode
from evaluations.metrics import parse_scored_csv

TAGS = [
    ("hmm1",       "hmm1",     ROOT / "results/evaluation/comparison_hmm1_a1.0_scored.csv"),
    ("hmm2_64",    "hmm2",     ROOT / "results/evaluation/comparison_hmm2_64_a1.0_scored.csv"),
    ("hmm2_256",   "hmm2",     ROOT / "results/evaluation/comparison_hmm2_256_a1.0_scored.csv"),
]

results = []
for label, variant, csv_path in TAGS:
    if not csv_path.exists():
        print(f"MISSING scored CSV: {csv_path}", file=sys.stderr)
        sys.exit(2)
    df = parse_scored_csv(str(csv_path))
    for mode, a in [("trace", 1.0), ("baseline", 0.0)]:
        cols = [c for c in df.columns if c.startswith(f"{mode}_gen_")]
        if not cols:
            continue
        metrics = _aggregate_mode(df, mode)
        n_gens = metrics["num_generations_per_prompt"]
        degraded = ["dist1","dist2","dist3","prob_tox_gt_0.5"] if n_gens < 5 else []
        results.append({
            "variant": label if mode == "trace" else f"{label}_baseline",
            "mode": mode,
            "a": a,
            "metrics": metrics,
            "num_generations": n_gens,
            "degraded_metrics": degraded,
            "source_csv": str(csv_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

out_json = ROOT / "results/tables/table1_detoxification.json"
out_csv  = ROOT / "results/tables/table1_detoxification.csv"
out_json.write_text(json.dumps(results, indent=2))

import pandas as pd
rows = []
for e in results:
    row = {"variant": e["variant"], "mode": e["mode"], "a": e["a"]}
    row.update(e["metrics"])
    rows.append(row)
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"Table 1 -> {out_json}")
print(f"         -> {out_csv}")
PY

# ------------------------------------------------------------------
# Table 6: factorizability (toxicity only; politics skipped if no neural ckpt).
# Not variant-dependent.
# ------------------------------------------------------------------
python -m evaluations.tables.table6_factorizability || echo "table6 failed (non-fatal)"

# ------------------------------------------------------------------
# Plot: fluency-toxicity tradeoff (reads table1 JSON)
# ------------------------------------------------------------------
python -m evaluations.plots.plot_fluency_toxicity_tradeoff \
    --input results/tables/table1_detoxification.json

# ------------------------------------------------------------------
# Plot: capacity-vs-toxicity scatter (reads scored CSVs directly)
# ------------------------------------------------------------------
python -m evaluations.plots.plot_hmm_quality_vs_toxicity \
    --models "hmm1:models/hmm_gpt2-large_bttf,hmm2:models/hmm2_gpt2-large_64_bttf,hmm2:models/hmm2_gpt2-large_256_bttf" \
    --a "$A"

# ------------------------------------------------------------------
# Plot: transformation distributions (needs EAP dumps from gen_all.sh)
# ------------------------------------------------------------------
python -m evaluations.plots.plot_transformation_distributions \
    --train_data data/RTP_train.jsonl \
    --coefficients classifiers/coefficients_nontoxicity.csv \
    --eap_dumps \
        results/figures/eap_hmm1.npz \
        results/figures/eap_hmm2_64.npz \
        results/figures/eap_hmm2_256.npz \
    --eap_labels hmm1 hmm2_64 hmm2_256

echo "==== ANALYSIS DONE ===="
ls -la results/tables/ results/figures/
