# TRACE Experiments

Model-agnostic evaluation harness that reproduces quantitative results from the TRACE paper. Supported variants: `hmm1` (first-order HMM) and `hmm2` (SOHMM / second-order HMM). `chmm` is **not implemented** — passing it to `src/generate.py` raises `NotImplementedError`.

## Directory Structure

```
evaluations/
├── metrics.py                              # Shared metric computation
├── generation_runner.py                    # Wraps src/generate.py + src/score.py
├── judge.py                                # LM-as-judge (Llama-3.3-70B-Instruct)
├── classifiers/
│   ├── fit_nontoxicity.py                  # Lasso classifier for nontoxicity
│   ├── fit_role.py                         # Per-character classifiers (RoleBench)
│   ├── fit_nonpoliticalness.py             # Nonpoliticalness classifier
│   └── fit_neural_baseline.py              # DistilBERT baseline (Table 6)
├── tables/
│   ├── table1_detoxification.py            # Full RTP eval (10k × 25)
│   ├── table2_transformation_ablation.py   # Logit transform ablation
│   ├── table4_timing.py                    # Wall-clock measurements
│   ├── table5_composition.py               # Nontoxicity + nonpoliticalness
│   ├── table6_factorizability.py           # CE loss: factorised vs neural
│   ├── table7_conditional_entropy.py       # Token-level conditional entropy
│   └── table8_lm_judge.py                  # LM-as-judge scores
├── plots/
│   ├── plot_fluency_toxicity_tradeoff.py   # Figure 3
│   ├── plot_hmm_quality_vs_toxicity.py     # Capacity-vs-detox scatter (across HMM sizes)
│   ├── plot_transformation_distributions.py # Score & EAP histograms
│   └── plot_role_quality_scatter.py        # Prompting vs TRACE per character
├── (outputs → results/tables/ and results/figures/ at repo root)
├── .score_cache/                           # SHA1-keyed Detoxify cache
└── .judge_cache/                           # SHA1-keyed judge cache
```

## Prerequisites

- The `--hmm_variant` flag on `src/generate.py` (already added).
- For the full 10k-prompt evaluation, download RealToxicityPrompts:
  ```bash
  # The DExperts 10k test split should be saved as data/rtp_10k.jsonl
  # Format: {"prompt": {"text": "..."}} per line
  ```
- For role-play evaluations, `ZenMoore/RoleBench` is downloaded automatically.
- For Table 8, access to `meta-llama/Llama-3.3-70B-Instruct` on a GPU with ≥80GB VRAM.

## Quick Start

### 1. Fit classifiers

```bash
# Nontoxicity (standard)
python -m evaluations.classifiers.fit_nontoxicity

# Nontoxicity without logit transform (for Table 2 ablation)
python -m evaluations.classifiers.fit_nontoxicity --b 1 --c 0 \
    --output_path classifiers/coefficients_nontoxicity_notf.csv

# Nonpoliticalness
python -m evaluations.classifiers.fit_nonpoliticalness

# Neural baseline (for Table 6)
python -m evaluations.classifiers.fit_neural_baseline --attribute toxicity
```

### 2. Run tables

```bash
# Table 1: full detoxification eval
python -m evaluations.tables.table1_detoxification \
    --prompts_path data/rtp_10k.jsonl --variants hmm1

# Table 2: transformation ablation
python -m evaluations.tables.table2_transformation_ablation --hmm_variant hmm1

# Table 4: timing
python -m evaluations.tables.table4_timing --hmm_variant hmm1

# Table 5: composition
python -m evaluations.tables.table5_composition --hmm_variant hmm1

# Table 6: factorizability
python -m evaluations.tables.table6_factorizability

# Table 7: conditional entropy
python -m evaluations.tables.table7_conditional_entropy --hmm_variant hmm1

# Table 8: LM judge (slowest — resumable)
python -m evaluations.tables.table8_lm_judge \
    --scored_csv results/evaluation/detox_hmm1_a1.0_scored.csv --hmm_variant hmm1
```

### 3. Generate plots

```bash
python -m evaluations.plots.plot_fluency_toxicity_tradeoff
# Pass one EAP .npz dump per variant (produced by src/generate.py --dump_eap_path)
python -m evaluations.plots.plot_transformation_distributions \
    --eap_dumps results/figures/eap_hmm1.npz results/figures/eap_hmm2_256.npz \
    --eap_labels hmm1 hmm2_256
# Capacity-vs-detox scatter across finished HMM checkpoints (no training-step sweep required)
python -m evaluations.plots.plot_hmm_quality_vs_toxicity \
    --models "hmm1:models/hmm_gpt2-large_bttf,hmm2:models/hmm2_gpt2-large_64_bttf,hmm2:models/hmm2_gpt2-large_256_bttf"
python -m evaluations.plots.plot_role_quality_scatter --role_results results/figures/role_eval.json
```

## Variant Flag

Every script accepts `--hmm_variant {hmm1,hmm2}`. The variant is:
- Recorded in every output JSON and filename
- Validated during plotting/aggregation — missing variant metadata fails loudly

Multiple variants can be evaluated side-by-side:
```bash
python -m evaluations.tables.table1_detoxification --variants hmm1,hmm2
```

`chmm` appears as a reserved choice in some argparse definitions but will raise
`NotImplementedError` at `src/generate.py` because there is no `src/chmm.py` and no
`utils.load_chmm_model`. Ignore `models/chmm_gpt2-large_bttf/` and any
`comparison_chmm_*` artifacts — they are pre-SOHMM-refactor detritus.

### Distinguishing hmm2 checkpoint sizes

`src/generate.py`'s default output path keys on `--hmm_variant` only, so two
`hmm2` checkpoints of different hidden sizes (e.g., H=64 vs H=256) will clobber
each other. When running side-by-side, either pass an explicit output filename
with `--output_csv`/`--scored_csv` or rename after the fact. The existing
`comparison_hmm2_64_a1.0_*.csv` and `comparison_hmm2_256_a1.0_*.csv` use the
`hmm2_<H>` convention adopted throughout this repo.

## Resumability

Tables 1, 5, and 8 are resumable:
- **Table 1**: checks for existing scored CSVs before generating
- **Table 5**: writes a checkpoint JSON after each row
- **Table 8**: `LocalJudge.score_batch()` checkpoints every N items

## Caching

- **Detoxify scores**: cached under `.score_cache/` by SHA1 of text
- **Judge scores**: cached under `.judge_cache/` by SHA1 of (prompt + continuations)
- Cache files are JSON, two-level directory structure (`{sha1[:2]}/{sha1}.json`)

## Output Format

Every script outputs:
- A **JSON file** in `results/tables/` with full hyperparameters, timestamps, and raw numbers
- A **CSV file** alongside (human-readable table rendering)
- Plots output **PNG** in `results/figures/` with any companion data JSON in `results/figures/`

JSON is the source of truth; CSV and PNG are renderings of it.
