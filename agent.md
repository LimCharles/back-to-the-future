# agent.md

Agent-oriented repository map for TRACE (Back-to-the-Future fork).

This document is intentionally denser than `README.md`: it is optimized for autonomous agents that need fast, correct routing across training, generation, scoring, and evaluation codepaths.

## 1) What this repo is

- Base project: TRACE (HMM-guided controllable generation for toxicity reduction).
- This fork extends the runtime/evaluation surface to support multiple decoding variants: `hmm1`, `hmm2`, and `chmm` (including your CHMM / higher-order experimentation track).
- `tutorial.ipynb` is part of original upstream tutorial flow and should be treated as pedagogical, not the authoritative source for fork-specific evaluation defaults.

## 2) Canonical runtime/output contracts (current)

- **Generation outputs**: `results/generated/*_generated.csv`
- **Scored outputs**: `results/evaluation/*_scored.csv`
- **Evaluation table outputs (JSON/CSV)**: `results/tables/`
- **Evaluation figure outputs (PNG/PDF/JSON sidecars)**: `results/figures/`

If an agent changes output paths, update all four surfaces together:
1) `src/*` defaults, 2) `evaluations/generation_runner.py`, 3) table/plot script argparse defaults, 4) docs.

## 3) High-level pipelines

### Pipeline A — Train attribute weights (toxicity or custom attribute)
1. Build / obtain attribute-scored JSONL (`data/RTP_train*.jsonl`) via `src/score_attribute.py` or external dataset.
2. Fit sparse linear token coefficients with `src/fit.py` (negative coefficients, zero intercept).
3. Coefficients written as CSV (`Token ID,Coefficient`) used by decoder guidance (`src/generate.py`).

### Pipeline B — Generate + score
1. `src/generate.py` loads LM + HMM + weights; emits generated CSV in `results/generated/`.
2. `src/score.py` computes toxicity (Detoxify), optional fluency/perplexity, diversity metrics.
3. Scored CSV written to `results/evaluation/` by default.

### Pipeline C — Repro evaluations harness
1. `evaluations/generation_runner.py` orchestrates generation/scoring subprocesses and caching.
2. `evaluations/tables/table*.py` compute benchmark summaries (JSON/CSV) under `results/tables/`.
3. `evaluations/plots/plot*.py` consume table outputs/raw artifacts and emit plots under `results/figures/`.

## 4) Core architecture notes for agents

- `src/hmm.py` defines HMM state dynamics, forward recursion, and backward expectation caches.
- `src/logits_processor.py` is the decode-time control kernel:
  - computes expected attribute-adjusted continuation utility per token,
  - reweights LM token probabilities,
  - supports transform/no-transform ablations (`decode_transform`).
- `a` is the decode guidance strength knob used across generation/evals.
- Generation CSV schema is JSON-per-cell in `trace_gen_*` / `baseline_gen_*` (or legacy `gen_*`) columns.
- Scoring augments rows with `max_toxicity`, `any_toxicity_gt_0.5`, `mean_fluency`, `dist-1/2/3`.

## 5) File-by-file map (all tracked files)

### Root and environment
- `.gitignore` — Ignore policy for envs/build artifacts/notebook state.
- `FAQ.md` — Operational troubleshooting (env, scoring anomalies, CUDA/runtime issues).
- `README.md` — Human-first project intro and end-to-end walkthrough.
- `agent.md` — This agent-focused operational map.
- `environment.yml` — GPU-focused conda environment with CUDA-aware setup.
- `environment_cpu.yml` — CPU-only conda environment.
- `pyproject.toml` — Project metadata and dependency declarations.
- `uv.lock` — Locked dependency graph for reproducible installs.
- `fit.sh` — SLURM batch script for classifier fitting workflow.
- `score.sh` — SLURM batch script for scoring generated CSVs to `results/evaluation/`.
- `main.ipynb` — Empty placeholder notebook (non-authoritative).
- `tutorial.ipynb` — Upstream tutorial notebook; useful for onboarding, but separate from fork-specific refactor decisions.

### Data inputs
- `data/prompts.jsonl` — Default prompt source for generation/evaluation quick runs.
- `data/coefficients.csv` — Default fitted token coefficients used for toxicity control.
- `data/custom_coefficients.csv` — Alternate coefficients for custom attribute experiments.

### Existing run logs/artifacts
- `output/fit.out` — Stdout capture from historical fitting run.
- `output/fit.err` — Stderr capture from historical fitting run.
- `output/score.out` — Stdout capture from historical scoring run.
- `output/score.err` — Stderr capture from historical scoring run.

### Core source (`src/`)
- `src/hmm.py` — HMM module + numerically stable ops (`stable_mvm`), forward probability and backward expectation routines, token-weight application.
- `src/logits_processor.py` — TRACE logits processor; compiled adjustment kernels; optional decode transform ablation; optional first-step EAP dump.
- `src/utils.py` — Shared loaders/helpers (HMM + coefficient loading and utilities reused by scripts).
- `src/fit.py` — Lasso-based coefficient fitting pipeline from attribute-labeled JSONL; includes transformation controls and diagnostics plot output.
- `src/score_attribute.py` — Zero-shot attribute scoring utility to produce training-ready JSONL for arbitrary attributes.
- `src/generate.py` — Main generation entrypoint; supports variants `hmm1/hmm2/chmm`, baseline mode, batching, and writes `_generated.csv` to `results/generated/`.
- `src/score.py` — Main scoring entrypoint; reads generated CSV, computes toxicity/fluency/diversity, writes `_scored.csv` to `results/evaluation/` by default.
- `src/score_original.py` — Legacy scorer retained for comparison/backward compatibility.

### Evaluations package scaffolding
- `evaluations/__init__.py` — Package marker.
- `evaluations/README.md` — Evaluation harness usage docs, defaults, and script catalog.
- `evaluations/csv_schema.py` — Canonical scored-CSV schema validation (contract for tables/plots).
- `evaluations/device.py` — Device selection helper with fallback policy.
- `evaluations/metrics.py` — Shared metrics: parsing, toxicity aggregations, distinct-n, entropy, etc.
- `evaluations/generation_runner.py` — Programmatic subprocess wrapper around `src/generate.py` + `src/score.py`; default scored outputs now target `results/evaluation/`; maintains Detoxify cache.
- `evaluations/judge.py` — Local LM-as-judge inference/caching utility (Table 8 pipeline).

### Evaluation classifier fit scripts
- `evaluations/classifiers/__init__.py` — Package marker.
- `evaluations/classifiers/fit_nontoxicity.py` — Builds nontoxicity classifier using shared fit pipeline.
- `evaluations/classifiers/fit_nonpoliticalness.py` — Builds nonpoliticalness classifier (composition experiments).
- `evaluations/classifiers/fit_role.py` — Trains per-character RoleBench role classifiers.
- `evaluations/classifiers/fit_neural_baseline.py` — DistilBERT neural baseline for factorizability comparison.

### Evaluation tables (defaults -> `results/tables/`)
- `evaluations/tables/__init__.py` — Package marker.
- `evaluations/tables/table1_detoxification.py` — Full detox sweep over variants and `a`; resumable; now checks scored CSVs in `results/evaluation/`.
- `evaluations/tables/table2_transformation_ablation.py` — Train/decode transform ablation matrix.
- `evaluations/tables/table3_roles.py` — Qualitative role-play table; now aligns generate fallback paths with `results/generated/`.
- `evaluations/tables/table4_timing.py` — Training/inference timing comparisons.
- `evaluations/tables/table5_composition.py` — Attribute composition (e.g., toxicity + politics) experiments.
- `evaluations/tables/table6_factorizability.py` — Factorized linear vs neural baseline CE comparison.
- `evaluations/tables/table7_conditional_entropy.py` — Conditional entropy computation from scored CSVs.
- `evaluations/tables/table8_lm_judge.py` — LM-as-judge evaluation with checkpointing.

### Evaluation plots (defaults -> `results/figures/`)
- `evaluations/plots/__init__.py` — Package marker.
- `evaluations/plots/plot_fluency_toxicity_tradeoff.py` — Figure from Table 1 JSON (default input now under `results/tables/`).
- `evaluations/plots/plot_hmm_quality_vs_toxicity.py` — Checkpoint quality vs toxicity trend plots.
- `evaluations/plots/plot_transformation_distributions.py` — Distribution visualizations (raw/transformed/fitted scores + optional EAP).
- `evaluations/plots/plot_role_quality_scatter.py` — Prompting-vs-TRACE role quality scatter.

## 6) Most important integration edges

- `src/generate.py` naming convention must stay in sync with `GenerationRunner.generate()` output-path inference.
- `src/score.py` default scoring location must stay in sync with `GenerationRunner.score()` and table scripts expecting scored CSV reuse.
- Any schema-affecting change in `src/score.py` should be mirrored in `evaluations/csv_schema.py` + parsing logic in `evaluations/metrics.py`.
- `table1_detoxification.py` resumability depends on deterministic path conventions for scored files.

## 7) Fast operational checklist for agents

When changing pipeline behavior:
1. Update code defaults (`src/*`, `evaluations/*`) and docs (`README.md`, `evaluations/README.md`, `agent.md`).
2. Run syntax smoke checks (`py_compile`) across touched scripts.
3. Verify path contracts by searching for stale path patterns.
4. Keep tutorial compatibility notes explicit if behavior diverges from notebook assumptions.

