# agent.md

Agent-oriented repository map for TRACE (Back-to-the-Future fork).

This document is optimized for autonomous agents that need fast, correct
routing across training, generation, scoring, table aggregation, and plot
rendering. It is denser and more prescriptive than [README.md](README.md).

---

## 1) What this repo is

- Base project: **TRACE** — HMM-guided controllable generation for toxicity reduction
  (Weng-Yidou et al., ICML 2025).
- This fork extends the runtime/evaluation surface to decode with multiple HMM
  variants.

**Supported variants (as of the current tree):**

| Variant | Meaning | Model class | Decode kernel |
|---|---|---|---|
| `hmm1` | first-order HMM | [src/hmm.py](src/hmm.py) (`HMM`) | [src/logits_processor.py](src/logits_processor.py) (`HmmGuidedLogitsProcessor`) |
| `hmm2` | second-order HMM / SOHMM / SHMM | [src/sohmm.py](src/sohmm.py) (`SOHMM`) | [src/logits_processor_sohmm.py](src/logits_processor_sohmm.py) (`SOHmmGuidedLogitsProcessor`) |
| `chmm` | **not implemented** | — | — |

`--hmm_variant chmm` raises `NotImplementedError` in [src/generate.py:126-131](src/generate.py#L126-L131).
Any [models/chmm_gpt2-large_bttf/](models/chmm_gpt2-large_bttf/) directory and any
`*_chmm_*.csv` artifacts under `results/` are pre-SOHMM-refactor detritus and
should be ignored until `src/chmm.py` + `utils.load_chmm_model` are added.

[tutorial.ipynb](tutorial.ipynb) is upstream pedagogical material, not the
authoritative source for fork-specific evaluation defaults.

---

## 2) Canonical runtime/output contracts

Pipeline: **generate → score → table → plot**. Each stage has a canonical
output directory — do not cross-pollute.

| Stage | Script | Default output |
|---|---|---|
| Generate | [src/generate.py](src/generate.py) | `results/generated/{comparison\|detox}_<tag>_a<a>_generated.csv` |
| Score | [src/score.py](src/score.py) | `results/evaluation/<stem>_scored.csv` |
| Table | `evaluations/tables/table*.py` | `results/tables/table*.{json,csv}` |
| Plot | `evaluations/plots/plot*.py` | `results/figures/*.png` + companion `*.json` **next to the PNG** |

**Generate filename convention.** `src/generate.py` uses `--hmm_variant` as the
only tag. Two `hmm2` checkpoints of different sizes will clobber each other
unless the caller renames the output or passes a distinct `--hmm_model_path`
with a wrapper script that renames after. Existing files in
[results/generated/](results/generated/) and [results/evaluation/](results/evaluation/) follow the **`hmm2_<H>`** convention (e.g.
`comparison_hmm2_64_a1.0_*.csv`, `comparison_hmm2_256_a1.0_*.csv`) — produced by manual
renaming after each run. Keep using this convention; the capacity plot
([evaluations/plots/plot_hmm_quality_vs_toxicity.py](evaluations/plots/plot_hmm_quality_vs_toxicity.py)) depends on it.

**Plot companion JSON.** Every plot writes its source-of-truth data JSON next
to its PNG under `results/figures/`. Do not write plot companion JSONs
anywhere under `evaluations/`.

If an agent changes any output path, update **four** surfaces together:
1. `src/*` defaults, 2. [evaluations/generation_runner.py](evaluations/generation_runner.py), 3. table/plot argparse defaults, 4. this file + [README.md](README.md) + [evaluations/README.md](evaluations/README.md).

---

## 3) High-level pipelines

### Pipeline A — Fit attribute weights (Lasso classifier)

1. Obtain / build an attribute-scored JSONL (`data/RTP_train*.jsonl`). Use
   [src/score_attribute.py](src/score_attribute.py) for zero-shot attribute
   labeling on arbitrary attributes.
2. Fit sparse negative-coefficient Lasso via
   [src/fit.py](src/fit.py) (zero intercept, optional logit
   transform governed by `--b`/`--c`).
3. Output CSV in [data/](data/) or [classifiers/](classifiers/) (schema:
   `Token ID,Coefficient`). Decoder reads from `--weights_path`.

Companion entry points under [evaluations/classifiers/](evaluations/classifiers/)
wrap `src/fit.py` for the paper-aligned attributes (nontoxicity,
nontoxicity-notf, nonpoliticalness, role, neural baseline).

### Pipeline B — Generate → Score

1. `src/generate.py` loads LM + HMM(/SOHMM) + weights, dispatches on
   `--hmm_variant`, writes `results/generated/*_generated.csv`.
2. `src/score.py` consumes the generated CSV and augments rows with
   `max_toxicity`, `any_toxicity_gt_0.5`, `mean_fluency`, `dist-1/2/3`
   (Detoxify + GPT-2-XL PPL + distinct-n).
3. Scored CSV lands in `results/evaluation/` (auto-inferred from input
   filename if `--output_csv` is omitted).

**Variant dispatch lives in [src/generate.py:94-131](src/generate.py#L94-L131):**

| Variant | Loader | Processor | `--no_decode_transform` | `--dump_eap_path` |
|---|---|---|---|---|
| `hmm1` | `utils.load_hmm_model` | `HmmGuidedLogitsProcessor` | honored | honored |
| `hmm2` | `utils.load_sohmm_model` | `SOHmmGuidedLogitsProcessor` | warning + ignored | honored |
| `chmm` | — | — | `NotImplementedError` | — |

### Pipeline C — Tables and plots

1. [evaluations/generation_runner.py](evaluations/generation_runner.py)
   (`GenerationRunner.generate_and_score`) is the Python wrapper around
   `generate.py` + `score.py`. All tables and plots go through it rather than
   shelling out directly.
2. [evaluations/tables/table*.py](evaluations/tables/) reads scored CSVs
   (or generates them on demand via the runner), computes aggregate numbers,
   and writes `{json,csv}` into `results/tables/`.
3. [evaluations/plots/plot*.py](evaluations/plots/) reads either table JSONs
   or raw artifacts (e.g. EAP `.npz` dumps) and writes PNG + companion JSON
   into `results/figures/`.

---

## 4) Core architecture notes for agents

- [src/hmm.py](src/hmm.py) — first-order HMM. Parameters: `alpha_exp (H,H)` in
  prob space, `beta (H,V)` log-space, `gamma_exp (1,H)` prob-space. Exposes
  `compute_forward_probability` (forward α over a prompt) and
  `compute_backward_expectation` (E[exp Σ w(x_i)] per state, per step).
  `stable_mvm` is the numerically stable log-domain matrix-vector multiply.
- [src/sohmm.py](src/sohmm.py) — second-order HMM. Parameters:
  `alpha_exp (H,H,H)` prob-space, `beta (H,V)` log-space,
  `gamma (H,H)` log-space. `compute_forward_probability` returns a pair-state
  log-α `(B, H, H)`; `compute_backward_expectation` returns
  `(T, H, H)` indexed by `(z_{t-1}, z_t)`. `loglikelihood(input_ids, batch_size)`
  is available.
- [src/logits_processor.py](src/logits_processor.py) — TRACE decode kernel
  for `hmm1`: `logit_adjustment` (with sigmoid-logit reshape) and
  `logit_adjustment_no_transform` (ablation path for Table 2).
  `_compute_eap_for_dump` emits first-step per-token EAP pre/post transform
  to `.npz` when `--dump_eap_path` is set.
- [src/logits_processor_sohmm.py](src/logits_processor_sohmm.py) — TRACE
  decode kernel for `hmm2`: `logit_adjustment_so` carries
  `log_alpha_prev (B,H,H)` and a scalar `product_generated_toxicity (B,)`
  across steps. `_compute_eap_for_dump_so` mirrors the hmm1 dump contract
  (same `.npz` keys `eap_pre_transform`/`eap_post_transform`), so the same
  downstream plot consumes both.
- [src/utils.py](src/utils.py) — shared loaders: `load_hmm_model`,
  `load_sohmm_model`, `load_weights`. The weights CSV loader enforces
  sequential `Token ID` starting at 0.
- `--a` is the decode-time guidance-strength knob. `a=0` is no guidance,
  `a=1` is the paper default, `a>1` is aggressive.
- Generation CSV schema: JSON-per-cell under `trace_gen_{k}` and/or
  `baseline_gen_{k}` columns. Legacy `gen_{k}` is also read by metrics.
- Scored CSV aggregate columns mix TRACE and baseline. Tables that need
  per-mode numbers recompute from the JSON cells — see
  `evaluations/tables/table1_detoxification._aggregate_mode` and
  `evaluations/plots/plot_hmm_quality_vs_toxicity._per_mode_aggregate`.

---

## 5) File-by-file map

### Root
- [.gitignore](.gitignore) — ignores envs/build artifacts/notebook state plus `results/**`, `classifiers/**`.
- [FAQ.md](FAQ.md) — operational troubleshooting (env, scoring, CUDA).
- [README.md](README.md) — human-first project intro and walkthrough.
- [agent.md](agent.md) — this file.
- [environment.yml](environment.yml) / [environment_cpu.yml](environment_cpu.yml) — conda envs (GPU / CPU).
- [pyproject.toml](pyproject.toml) — project metadata.
- [uv.lock](uv.lock) — locked dep graph.
- [fit.sh](fit.sh) — SLURM batch for classifier fitting (H100 partition).
- [score.sh](score.sh) — SLURM batch for scoring the three comparison variants.
  **The `chmm` line in this file currently invokes a scored CSV that targets
  the unimplemented variant — remove or replace with an hmm2-size-specific
  entry before running.**
- [main.ipynb](main.ipynb) — empty placeholder.
- [tutorial.ipynb](tutorial.ipynb) — upstream tutorial.

### Data inputs ([data/](data/))
- [data/prompts.jsonl](data/prompts.jsonl) — 12-prompt demo set.
- [data/coefficients.csv](data/coefficients.csv) — default token coefficients (toxicity).
- [data/custom_coefficients.csv](data/custom_coefficients.csv) — alt coefficients for custom attributes.
- [data/RTP_train.jsonl](data/RTP_train.jsonl) — RealToxicityPrompts train split (100k).
- [data/RTP_test.jsonl](data/RTP_test.jsonl) — RealToxicityPrompts test split (10k); also used by the optional `--val_data` path of the capacity plot.
- [data/misra_news.json](data/misra_news.json) — Misra News Category dataset (for nonpoliticalness).
- [data/rolebench/](data/rolebench/) — RoleBench JSONL splits.

### Models ([models/](models/))
- [models/hmm_gpt2-large_bttf/](models/hmm_gpt2-large_bttf/) — `hmm1`, H=4096 (our group's fork-specific retrain).
- [models/hmm_gpt2-large_uncon_seq-len-32_4096_10M/](models/hmm_gpt2-large_uncon_seq-len-32_4096_10M/) — `hmm1`, H=4096 (upstream paper checkpoint).
- [models/hmm2_gpt2-large_64_bttf/](models/hmm2_gpt2-large_64_bttf/) — `hmm2`, H=64.
- [models/hmm2_gpt2-large_256_bttf/](models/hmm2_gpt2-large_256_bttf/) — `hmm2`, H=256.
- [models/chmm_gpt2-large_bttf/](models/chmm_gpt2-large_bttf/) — **orphaned**. No `src/chmm.py`; `generate.py` raises. Keep the dir or delete it, but do not route evaluations through it.

### Classifiers ([classifiers/](classifiers/))
- [classifiers/coefficients_nontoxicity.csv](classifiers/coefficients_nontoxicity.csv) — standard Lasso nontoxicity (b=10, c=3).
- [classifiers/coefficients_nontoxicity_notf.csv](classifiers/coefficients_nontoxicity_notf.csv) — no-logit-transform variant for Table 2.
- [classifiers/coefficients_nonpoliticalness.csv](classifiers/coefficients_nonpoliticalness.csv) — Lasso nonpoliticalness for Table 5.
- [classifiers/neural_classifier_toxicity.pt](classifiers/neural_classifier_toxicity.pt) + [classifiers/neural_classifier_toxicity_metrics.json](classifiers/neural_classifier_toxicity_metrics.json) — DistilBERT baseline for Table 6.
- [classifiers/misra_news_political_scores.json](classifiers/misra_news_political_scores.json) — zero-shot scores for Misra News; input to `fit_nonpoliticalness`.
- [classifiers/role/](classifiers/role/) — per-character RoleBench classifiers.

### Results ([results/](results/) — gitignored)
- [results/generated/](results/generated/) — `_generated.csv` from `src/generate.py`.
- [results/evaluation/](results/evaluation/) — `_scored.csv` from `src/score.py`.
- [results/tables/](results/tables/) — `table*.{json,csv}` from the tables harness.
- [results/figures/](results/figures/) — `*.png` plus companion `*.json` / `*.npz` from plots.

### Core source ([src/](src/))
- [src/hmm.py](src/hmm.py) — first-order `HMM`; `stable_mvm`; forward α and backward E.
- [src/sohmm.py](src/sohmm.py) — second-order `SOHMM`; pair-state forward/backward; loglikelihood.
- [src/logits_processor.py](src/logits_processor.py) — `HmmGuidedLogitsProcessor`; compiled kernels; first-step EAP dump.
- [src/logits_processor_sohmm.py](src/logits_processor_sohmm.py) — `SOHmmGuidedLogitsProcessor`; compiled kernel; first-step EAP dump (same npz contract as hmm1).
- [src/utils.py](src/utils.py) — shared loaders for HMM/SOHMM/weights.
- [src/fit.py](src/fit.py) — Lasso coefficient fit (+ diagnostics plot).
- [src/score_attribute.py](src/score_attribute.py) — zero-shot attribute labeling for custom JSONL prep.
- [src/generate.py](src/generate.py) — generation entrypoint; variant dispatch at lines 94-131.
- [src/score.py](src/score.py) — scoring entrypoint (Detoxify + PPL + dist-n).
- [src/score_original.py](src/score_original.py) — **legacy**; retained for upstream comparison only.

### Evaluations harness ([evaluations/](evaluations/))
- [evaluations/__init__.py](evaluations/__init__.py) — package marker.
- [evaluations/README.md](evaluations/README.md) — harness usage + defaults.
- [evaluations/csv_schema.py](evaluations/csv_schema.py) — canonical scored-CSV schema + `validate_scored_csv`.
- [evaluations/device.py](evaluations/device.py) — device helper with fallback.
- [evaluations/metrics.py](evaluations/metrics.py) — shared metrics (parse, aggregate, dist-n, conditional entropy, cross-entropy).
- [evaluations/generation_runner.py](evaluations/generation_runner.py) — Python wrapper for generate.py + score.py with Detoxify cache.
- [evaluations/judge.py](evaluations/judge.py) — local LM-as-judge inference/caching (Table 8).

### Classifier fit scripts ([evaluations/classifiers/](evaluations/classifiers/))
- [evaluations/classifiers/fit_nontoxicity.py](evaluations/classifiers/fit_nontoxicity.py) — standard nontoxicity via `src/fit.py`.
- [evaluations/classifiers/fit_nonpoliticalness.py](evaluations/classifiers/fit_nonpoliticalness.py) — nonpoliticalness (composition).
- [evaluations/classifiers/fit_role.py](evaluations/classifiers/fit_role.py) — per-character RoleBench.
- [evaluations/classifiers/fit_neural_baseline.py](evaluations/classifiers/fit_neural_baseline.py) — DistilBERT (Table 6).

### Tables ([evaluations/tables/](evaluations/tables/) → `results/tables/`)
- [evaluations/tables/table1_detoxification.py](evaluations/tables/table1_detoxification.py) — full RTP sweep; resumable via scored-CSV check.
- [evaluations/tables/table2_transformation_ablation.py](evaluations/tables/table2_transformation_ablation.py) — train/decode transform ablation. **hmm1 only** (SOHMM kernel has no no-transform path).
- [evaluations/tables/table3_roles.py](evaluations/tables/table3_roles.py) — qualitative role-play side-by-side.
- [evaluations/tables/table4_timing.py](evaluations/tables/table4_timing.py) — fit + per-token inference timing.
- [evaluations/tables/table5_composition.py](evaluations/tables/table5_composition.py) — nontoxicity × nonpoliticalness composition.
- [evaluations/tables/table6_factorizability.py](evaluations/tables/table6_factorizability.py) — Lasso vs neural CE (no HMM needed).
- [evaluations/tables/table7_conditional_entropy.py](evaluations/tables/table7_conditional_entropy.py) — token-level conditional entropy from scored CSV.
- [evaluations/tables/table8_lm_judge.py](evaluations/tables/table8_lm_judge.py) — Llama-3.3-70B-Instruct judge, resumable.

### Plots ([evaluations/plots/](evaluations/plots/) → `results/figures/`)
- [evaluations/plots/plot_fluency_toxicity_tradeoff.py](evaluations/plots/plot_fluency_toxicity_tradeoff.py) — Figure 3; consumes `results/tables/table1_detoxification.json`.
- [evaluations/plots/plot_hmm_quality_vs_toxicity.py](evaluations/plots/plot_hmm_quality_vs_toxicity.py) — **repurposed** from training-step sweep to capacity sweep. Plots `H` (log-scale) vs `avg_max_tox` (TRACE-mode) for a user-supplied set of `variant:path` tuples; optional val-LL twin-axis.
- [evaluations/plots/plot_transformation_distributions.py](evaluations/plots/plot_transformation_distributions.py) — score/EAP histograms. Accepts `--eap_dumps` from either hmm1 or hmm2 (same `.npz` contract).
- [evaluations/plots/plot_role_quality_scatter.py](evaluations/plots/plot_role_quality_scatter.py) — prompting vs TRACE per RoleBench character.

---

## 6) Most important integration edges

- Variant dispatch is centralised in `src/generate.py:94-131`. Any new variant
  means (a) new source files, (b) a new loader in `src/utils.py`, (c) a new
  branch in the dispatch block, (d) updating every `--hmm_variant` argparse
  choices list across `evaluations/`.
- `src/generate.py` naming convention must stay in sync with
  `GenerationRunner.generate()` output-path inference.
- `src/score.py` default output location must stay in sync with
  `GenerationRunner.score()` and tables/plots expecting scored CSV reuse.
- Schema changes in `src/score.py` must be mirrored in
  `evaluations/csv_schema.py` + parsing in `evaluations/metrics.py`.
- `table1_detoxification.py` resumability depends on deterministic scored-CSV
  filenames under `results/evaluation/`.
- Plot companion JSON paths: always `results/figures/` — see Section 2.
- hmm2 EAP dump contract matches hmm1 (`.npz` keys
  `eap_pre_transform`/`eap_post_transform`); any change to the dump payload
  must be applied in both `_compute_eap_for_dump` (hmm1) and
  `_compute_eap_for_dump_so` (hmm2) simultaneously.

---

## 7) Fast operational checklist for agents

When changing pipeline behavior:
1. Update code defaults in `src/*` and `evaluations/*`.
2. Update docs: [README.md](README.md), [evaluations/README.md](evaluations/README.md), this file.
3. Run a syntax smoke check (`python -m py_compile`) across touched scripts.
4. Verify path contracts by grepping for stale path patterns
   (`evaluations/results/`, hard-coded absolute paths).
5. Run the minimal smoke chain end-to-end if generation behavior changed:
   `generate --num_generations 1 --max_len 5` → `score` → a cheap table or
   plot that consumes the scored CSV.

When adding a new HMM variant:
1. Add `src/<variant>.py` (model) and `src/logits_processor_<variant>.py` (kernel).
2. Add loader in `src/utils.py`.
3. Add branch in `src/generate.py` dispatch block + extend `--hmm_variant`
   choices (currently `hmm1|hmm2|chmm`).
4. Extend `--hmm_variant` choices in every argparse under `evaluations/`.
5. Update Sections 1, 3, 5 of this file and the tables-and-plots list in
   `evaluations/README.md`.
