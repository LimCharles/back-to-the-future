#!/bin/bash
#SBATCH --job-name=score_all
#SBATCH --output=/home/c/caplim/back-to-the-future/output/score_all.out
#SBATCH --error=/home/c/caplim/back-to-the-future/output/score_all.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

nvidia-smi
source /home/c/caplim/back-to-the-future/.venv/bin/activate
cd /home/c/caplim/back-to-the-future

mkdir -p results/evaluation

A=1.0

for TAG in hmm1 hmm2_64 hmm2_256 chmm_uniform6 chmm_uniform8 chmm_quadratic_log; do
    IN=results/generated/comparison_${TAG}_a${A}_generated.csv
    OUT=results/evaluation/comparison_${TAG}_a${A}_scored.csv

    if [ ! -f "$IN" ]; then
        echo "MISSING: $IN" >&2
        exit 2
    fi

    rm -f "$OUT"
    echo "==== SCORING $IN -> $OUT ===="
    python src/score.py --input_csv "$IN" --output_csv "$OUT"
done

echo "==== SCORING DONE ===="
ls -la results/evaluation/comparison_*_a${A}_scored.csv
