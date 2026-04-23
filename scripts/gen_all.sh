#!/bin/bash
#SBATCH --job-name=gen_all
#SBATCH --output=/home/c/caplim/back-to-the-future/output/gen_all.out
#SBATCH --error=/home/c/caplim/back-to-the-future/output/gen_all.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

nvidia-smi
source /home/c/caplim/back-to-the-future/.venv/bin/activate
cd /home/c/caplim/back-to-the-future

mkdir -p results/generated results/figures results/evaluation results/tables

PROMPTS=data/RTP_test.jsonl
A=1.0
NG=25
ML=20
GBS=10  # 10 gens per prompt in a single generate() call

# ------------------------------------------------------------------
# hmm1 (H=4096)
# ------------------------------------------------------------------
rm -f results/generated/comparison_hmm1_a${A}_generated.csv
python src/generate.py \
    --hmm_variant hmm1 \
    --hmm_model_path models/hmm_gpt2-large_bttf \
    --prompts_path "$PROMPTS" \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --a "$A" --num_generations "$NG" --generation_batch_size "$GBS" --max_len "$ML" --baseline

python src/generate.py \
    --hmm_variant hmm1 \
    --hmm_model_path models/hmm_gpt2-large_bttf \
    --prompts_path data/prompts.jsonl \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --num_generations 1 --max_len 5 \
    --dump_eap_path results/figures/eap_hmm1.npz

# ------------------------------------------------------------------
# hmm2 (H=64)
# ------------------------------------------------------------------
rm -f results/generated/comparison_hmm2_64_a${A}_generated.csv
python src/generate.py \
    --hmm_variant hmm2 \
    --hmm_model_path models/hmm2_gpt2-large_64_bttf \
    --prompts_path "$PROMPTS" \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --a "$A" --num_generations "$NG" --generation_batch_size "$GBS" --max_len "$ML" --baseline \
    --output_path results/generated/comparison_hmm2_64_a${A}_generated.csv

python src/generate.py \
    --hmm_variant hmm2 \
    --hmm_model_path models/hmm2_gpt2-large_64_bttf \
    --prompts_path data/prompts.jsonl \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --num_generations 1 --max_len 5 \
    --dump_eap_path results/figures/eap_hmm2_64.npz

# ------------------------------------------------------------------
# hmm2 (H=256)
# ------------------------------------------------------------------
rm -f results/generated/comparison_hmm2_256_a${A}_generated.csv
python src/generate.py \
    --hmm_variant hmm2 \
    --hmm_model_path models/hmm2_gpt2-large_256_bttf \
    --prompts_path "$PROMPTS" \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --a "$A" --num_generations "$NG" --generation_batch_size "$GBS" --max_len "$ML" --baseline \
    --output_path results/generated/comparison_hmm2_256_a${A}_generated.csv

python src/generate.py \
    --hmm_variant hmm2 \
    --hmm_model_path models/hmm2_gpt2-large_256_bttf \
    --prompts_path data/prompts.jsonl \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --num_generations 1 --max_len 5 \
    --dump_eap_path results/figures/eap_hmm2_256.npz

# ------------------------------------------------------------------
# chmm (uniform6)
# ------------------------------------------------------------------
rm -f results/generated/comparison_chmm_uniform6_a${A}_generated.csv
python src/generate.py \
    --hmm_variant chmm \
    --hmm_model_path models/chmm_gpt-2-large_uniform6_bttf \
    --prompts_path "$PROMPTS" \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --a "$A" --num_generations "$NG" --generation_batch_size "$GBS" --max_len "$ML" --baseline \
    --output_path results/generated/comparison_chmm_uniform6_a${A}_generated.csv

python src/generate.py \
    --hmm_variant chmm \
    --hmm_model_path models/chmm_gpt-2-large_uniform6_bttf \
    --prompts_path data/prompts.jsonl \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --num_generations 1 --max_len 5 \
    --dump_eap_path results/figures/eap_chmm_uniform6.npz

# ------------------------------------------------------------------
# chmm (uniform8)
# ------------------------------------------------------------------
rm -f results/generated/comparison_chmm_uniform8_a${A}_generated.csv
python src/generate.py \
    --hmm_variant chmm \
    --hmm_model_path models/chmm_gpt-2-large_uniform8_bttf \
    --prompts_path "$PROMPTS" \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --a "$A" --num_generations "$NG" --generation_batch_size "$GBS" --max_len "$ML" --baseline \
    --output_path results/generated/comparison_chmm_uniform8_a${A}_generated.csv

python src/generate.py \
    --hmm_variant chmm \
    --hmm_model_path models/chmm_gpt-2-large_uniform8_bttf \
    --prompts_path data/prompts.jsonl \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --num_generations 1 --max_len 5 \
    --dump_eap_path results/figures/eap_chmm_uniform8.npz

# ------------------------------------------------------------------
# chmm (quadratic_log)
# ------------------------------------------------------------------
rm -f results/generated/comparison_chmm_quadratic_log_a${A}_generated.csv
python src/generate.py \
    --hmm_variant chmm \
    --hmm_model_path models/chmm_gpt-2-large_quadratic_log_bttf \
    --prompts_path "$PROMPTS" \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --a "$A" --num_generations "$NG" --generation_batch_size "$GBS" --max_len "$ML" --baseline \
    --output_path results/generated/comparison_chmm_quadratic_log_a${A}_generated.csv

python src/generate.py \
    --hmm_variant chmm \
    --hmm_model_path models/chmm_gpt-2-large_quadratic_log_bttf \
    --prompts_path data/prompts.jsonl \
    --weights_path classifiers/coefficients_nontoxicity.csv \
    --num_generations 1 --max_len 5 \
    --dump_eap_path results/figures/eap_chmm_quadratic_log.npz

echo "==== GENERATION DONE ===="
ls -la results/generated/comparison_*_a${A}_generated.csv
ls -la results/figures/eap_*.npz
