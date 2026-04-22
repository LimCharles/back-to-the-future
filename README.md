# TRACE fork: HMM vs SHMM vs CHMM for controllable detoxification

A research fork of [TRACE](https://github.com/yidouweng/trace) (Weng-Yidou et al., ICML 2025) that extends the decode-time guidance kernel to three HMM variants and compares them head-to-head on RealToxicityPrompts:

| Variant | Meaning | Hidden structure | Decode kernel |
|---|---|---|---|
| `hmm1` | First-order HMM (upstream baseline) | Dense `alpha_exp (H, H)` | [src/logits_processor.py](src/logits_processor.py) |
| `hmm2` | Second-order HMM (SOHMM / SHMM) | Dense `alpha_exp (H, H, H)` | [src/logits_processor_sohmm.py](src/logits_processor_sohmm.py) |
| `chmm` | Clone-hidden HMM | Sparse block transitions keyed by observed `(src_token, dst_token)` pairs | [src/logits_processor_chmm.py](src/logits_processor_chmm.py) |

Full routing, file-by-file layout, and integration edges live in [agent.md](agent.md). For a kernel smoke test that loads all three variants and scores one generation each, see [main.ipynb](main.ipynb).

## Pipeline at a glance

```
  fit classifiers  →  generate (× 6 variants)  →  score (Detoxify + GPT2-XL PPL + dist-n)  →  tables/plots
  classifiers/*.csv   results/generated/*.csv       results/evaluation/*.csv                   results/tables/ + results/figures/
```

The four driver scripts wire the full pipeline; each runs as a standalone SLURM batch (H100 partition) or inline via `bash`:

| Script | Does |
|---|---|
| [scripts/fit.sh](scripts/fit.sh) | Fits the attribute Lasso classifiers (nontoxicity + variants) |
| [scripts/gen_all.sh](scripts/gen_all.sh) | Generates 25 continuations per RTP_test prompt for every variant (hmm1, hmm2_{64, 256}, chmm_{uniform6, uniform8, quadratic_log}) |
| [scripts/score_all.sh](scripts/score_all.sh) | Scores every generated CSV with Detoxify + GPT-2-XL PPL + distinct-n |
| [scripts/analyze.sh](scripts/analyze.sh) | Builds Tables 1 & 6 and the three summary plots from the scored CSVs |

## 1. Setup

```bash
conda env create -f environment.yml   # GPU; auto-detects CUDA
conda activate trace
```

CPU-only: use `environment_cpu.yml`. Toxicity scoring uses [Detoxify](https://github.com/unitaryai/detoxify) locally, so no Perspective API key is needed.

## 2. Data & models

### Data

```bash
cd data/
wget https://github.com/yidouweng/trace/releases/download/v1.0.0/RTP_train.jsonl.tar.gz
wget https://github.com/yidouweng/trace/releases/download/v1.0.0/RTP_test.jsonl.tar.gz
tar -xzf RTP_train.jsonl.tar.gz
tar -xzf RTP_test.jsonl.tar.gz
cd ..
```

`data/prompts.jsonl` (12 demo prompts) is already tracked. `data/rolebench/` and `data/misra_news.json` ship for RoleBench and composition experiments.

### Models

Expected layout under `models/` (gitignored; see [agent.md §5](agent.md) for the full spec):

```
models/
├── hmm_gpt2-large_bttf/                    # hmm1, H=4096 (fork retrain)
├── hmm_gpt2-large_uncon_seq-len-32_4096_10M/  # hmm1, H=4096 (upstream paper checkpoint)
├── hmm2_gpt2-large_64_bttf/                 # hmm2, H=64
├── hmm2_gpt2-large_256_bttf/                # hmm2, H=256
├── chmm_gpt-2-large_uniform6_bttf/          # chmm, uniform-6 init
├── chmm_gpt-2-large_uniform8_bttf/          # chmm, uniform-8 init
└── chmm_gpt-2-large_quadratic_log_bttf/     # chmm, quadratic-log init
```

Upstream `hmm1` checkpoint:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='gwenweng/hmm-gpt2-large', \
                    local_dir='models/hmm_gpt2-large_uncon_seq-len-32_4096_10M')"
```

`hmm2` and `chmm` checkpoints come from our own training runs and from the [sukumar1612/Ctrl-G `CHMM_distillation`](https://github.com/sukumar1612/Ctrl-G/tree/sukumar/CHMM_distillation) branch respectively.

## 3. Fit classifiers

The attribute Lasso classifiers produce the `Coefficient` columns the decoder multiplies into the sigmoid-logit guidance. RTP_train already carries per-prompt toxicity labels, so `fit.py` can run directly; custom attributes are labelled first via zero-shot.

```bash
# Nontoxicity (standard): b=10, c=3 logit transform
python -m evaluations.classifiers.fit_nontoxicity
#   → classifiers/coefficients_nontoxicity.csv

# Nontoxicity without logit transform (for Table 2 ablation)
python -m evaluations.classifiers.fit_nontoxicity --b 1 --c 0 \
    --output_path classifiers/coefficients_nontoxicity_notf.csv

# Nonpoliticalness (for Table 5 composition)
python -m evaluations.classifiers.fit_nonpoliticalness
#   → classifiers/coefficients_nonpoliticalness.csv

# Per-character RoleBench classifiers
python -m evaluations.classifiers.fit_role
#   → classifiers/role/<character>.csv + classifiers/role/manifest.json

# DistilBERT neural baseline (Table 6)
python -m evaluations.classifiers.fit_neural_baseline --attribute toxicity
#   → classifiers/neural_classifier_toxicity.pt
```

For an arbitrary custom attribute (DeBERTa zero-shot labels → Lasso):

```bash
python src/score_attribute.py --attribute politics      # writes data/RTP_train_politics.jsonl
python src/fit.py --data_path data/RTP_train_politics.jsonl --attribute politics
```

[scripts/fit.sh](scripts/fit.sh) wraps the nontoxicity fit for SLURM.

## 4. Generate

Direct `src/generate.py` invocation:

```bash
python src/generate.py \
  --hmm_variant chmm \
  --hmm_model_path models/chmm_gpt-2-large_uniform6_bttf \
  --prompts_path data/prompts.jsonl \
  --weights_path classifiers/coefficients_nontoxicity.csv \
  --a 1.0 --max_len 20 --num_generations 5 --baseline
```

- `--hmm_variant` picks the decoder kernel: `hmm1 | hmm2 | chmm`.
- `--a` is the guidance strength (0 = no guidance, 1 = paper default, >1 = aggressive).
- `--baseline` runs unguided GPT-2 alongside TRACE into `results/generated/comparison_*_generated.csv`. Drop it for TRACE-only runs (`results/generated/detox_*_generated.csv`).
- `--dump_eap_path <path>.npz` writes first-step per-token expected-attribute-probability dumps; all three kernels share the same `.npz` schema for downstream plotting.

The full sweep over all six variants on RTP_test lives in [scripts/gen_all.sh](scripts/gen_all.sh).

## 5. Score

`src/score.py` runs locally: **Detoxify `'original'`** for toxicity, **GPT-2-XL** for perplexity, string-level distinct-n.

```bash
python src/score.py \
  --input_csv results/generated/comparison_chmm_uniform6_a1.0_generated.csv \
  --output_csv results/evaluation/comparison_chmm_uniform6_a1.0_scored.csv
```

Flags:
- `--toxicity_only` skips the fluency pass (much faster; sets `mean_fluency`/`dist-*` to `NA`).
- `--perp_model gpt2-large` drops perplexity model size if VRAM is tight.

Scored-CSV schema is in [evaluations/csv_schema.py](evaluations/csv_schema.py). Full sweep in [scripts/score_all.sh](scripts/score_all.sh); Detoxify scores are cached under `evaluations/.score_cache/` by SHA1 of continuation text so re-scoring is cheap.

## 6. Tables & plots

[scripts/analyze.sh](scripts/analyze.sh) runs the end-to-end table+plot chain:

```bash
bash scripts/analyze.sh
```

Individual entry points under [evaluations/](evaluations/). Full CLI reference: [evaluations/README.md](evaluations/README.md).

Key outputs (`results/tables/`, `results/figures/`):

- **`table1_detoxification.{json,csv}`** — per-variant TRACE + baseline: avg max-tox, prob(tox>0.5), dist-2/3, perplexity.
- **`fluency_toxicity_tradeoff.png`** — Figure 3 equivalent, hued by variant.
- **`hmm_capacity_vs_toxicity.png`** — H (log scale) vs avg max-tox across all six variants; optional twin-axis val-LL.
- **`transformation_distributions.png`** — Detoxify score transform + per-variant pre/post EAP histograms.

Additional tables available but not in the default driver:

- `table2_transformation_ablation` (hmm1-only logit-transform ablation)
- `table3_roles` (qualitative role-play side-by-side)
- `table4_timing` (fit + per-token inference ratio)
- `table5_composition` (nontoxicity × nonpoliticalness)
- `table6_factorizability` (Lasso vs neural CE)
- `table7_conditional_entropy` (from scored CSV)
- `table8_lm_judge` (Llama-3.3-70B judge; resumable)

## What's different from upstream TRACE

- **Two new decode kernels**: `hmm2` ([src/sohmm.py](src/sohmm.py) + [src/logits_processor_sohmm.py](src/logits_processor_sohmm.py)) and `chmm` ([src/chmm.py](src/chmm.py) + [src/logits_processor_chmm.py](src/logits_processor_chmm.py)). Upstream has only `hmm1`. CHMM is ported from the [`sukumar1612/Ctrl-G`](https://github.com/sukumar1612/Ctrl-G/tree/sukumar/CHMM_distillation) `CHMM_distillation` branch and adapted to the `(model, expectation_cache, a, tokenizer, dump_eap_path)` processor contract.
- **Reproducibility harness**: [evaluations/](evaluations/) (tables, plots, classifier fits, generation runner, LM-as-judge) — none of this existed upstream.
- **Local toxicity scoring**: Detoxify replaces the Perspective API. No rate limits, no key.
- **SLURM drivers**: [scripts/](scripts/) targets H100 partitions.

## Where to read next

- [agent.md](agent.md) — authoritative routing map (variant dispatch, file-by-file, integration edges). Start here when adding a variant or a new metric.
- [evaluations/README.md](evaluations/README.md) — per-table/plot CLI reference, caching/resumability notes.
- [main.ipynb](main.ipynb) — one-cell-per-step smoke test that loads each variant and scores a single generation.
- [tutorial.ipynb](tutorial.ipynb) — upstream pedagogical walkthrough, kept for reference. For the fork's end-to-end pipeline, use this README and the scripts above instead.

## Citation & license

MIT License. Upstream paper:

```bibtex
@inproceedings{yidou-weng2025trace,
  title={TRACE Back from the Future: A Probabilistic Reasoning Approach to Controllable Language Generation},
  author={Weng-Yidou, Gwen and Wang, Benjie and Van den Broeck, Guy},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```
