#!/bin/bash
#SBATCH --job-name=score
#SBATCH --output=/home/c/caplim/back-to-the-future/output/score.out
#SBATCH --error=/home/c/caplim/back-to-the-future/output/score.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

nvidia-smi
source /home/c/caplim/back-to-the-future/.venv/bin/activate

mkdir -p results/evaluation

python src/score.py --input_csv results/generated/comparison_hmm1_a1.0_generated.csv --output_csv results/evaluation/comparison_hmm1_a1.0_scored.csv --toxicity_only
python src/score.py --input_csv results/generated/comparison_hmm2_a1.0_generated.csv --output_csv results/evaluation/comparison_hmm2_a1.0_scored.csv --toxicity_only
python src/score.py --input_csv results/generated/comparison_chmm_a1.0_generated.csv --output_csv results/evaluation/comparison_chmm_a1.0_scored.csv --toxicity_only
