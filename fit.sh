#!/bin/bash
#SBATCH --job-name=fitting_classifiers
#SBATCH --output=/home/c/caplim/back-to-the-future/output/fit.out
#SBATCH --error=/home/c/caplim/back-to-the-future/output/fit.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

nvidia-smi
source /home/c/caplim/back-to-the-future/.venv/bin/activate

python -m evaluations.classifiers.fit_nontoxicity
python -m evaluations.classifiers.fit_nontoxicity --b 1 --c 0 --output_path classifiers/coefficients_nontoxicity_notf.csv
python -m evaluations.classifiers.fit_nonpoliticalness
python -m evaluations.classifiers.fit_neural_baseline --attribute toxicity
python -m evaluations.classifiers.fit_role --train_data data/rolebench/train.jsonl --neg_ratio 1.0
