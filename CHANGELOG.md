# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Changed
- Renamed the project from the working name `declib` to `brier` (package `brier`, base exception `BrierError`).
### Removed
- `datasets` from the `bench` extra (BANKING77 is read from its CSV directly).
### Fixed
- `brier check` no longer crashes when an error message contains characters the console cannot encode (e.g. `Ġ` on a Windows cp1252 console).
- L2 benchmark (`docs/results.md`, Qwen3-1.7B on banking20, 300 labels): accuracy 0.807 (L1: 0.668), ECE 0.035, NLL 0.727; label curve at 100/200/300. Colab notebooks `m5_4_l2_banking20_benchmark.ipynb` (full run incl. optional Qwen3-4B) and `m5_4b_l2_rerun.ipynb` (L2 only).
- L2 temperature: Platt (1999) targets for two classes; plain-NLL temperature scaling for more, refitted against one-vs-rest Platt targets only when it hits its lower bound (separable OOF scores). Smoothing every fit made 20-class heads badly underconfident (banking20 L2: ECE 0.46 at 81 % accuracy).
- mypy no longer pins `python_version = "3.10"`, which failed on Python 3.12+ where numpy 2.5 stubs use `type` statements.
### Added
- Wider model matrix: Granite-MoE (a mixture of experts) passes the integration suite; LFM2 (hybrid convolution), DeepSeek-R1-Distill, Falcon3, Phi-4-mini, SmolLM3, OLMoE (a second mixture of experts) and Mistral-7B pass `brier check` and the integration suite on an A100 (`notebooks/m6_3_compatibility_matrix.ipynb`); findings in `docs/COMPATIBILITY.md`. CI: weekly real-model run with the newest dependencies, unit tests on Windows and macOS, a numpy 1.24 / Python 3.10 job, and a test that runs the README examples.
- Calibration artifacts record the backend's precision (`model.dtype`, schema version 3, ADR-0007); `Decider.load` refuses an artifact fitted at another dtype. Version-1 and version-2 artifacts still load (no precision recorded, so no check).
- `brier check <model-id>` (and `brier.check.check_backend`): conformance report for a model before calibrating: label tokens per question type, batched-vs-single consistency, hidden-state access, a 14-item sanity task, and the levels the model supports; `--json`, exit code 1 on failure. The `brier` console script.
- README quickstart (runs as written), results summary and related-work section; `docs/related_work.md` (sources of the idea and methods, differences from AnyJev); pdoc API reference (dev dependency `pdoc`).
- Repository scaffold, specification, methods, roadmap and CI.
- Official Apache-2.0 licence text in `LICENSE`.
- Committed `uv.lock`; refreshed pre-commit hooks; CI audit no longer audits the unpublished project itself.
- `Choice`, `Noul`, `Score` question types with validation, the `Decision` result type and the `brier.errors` hierarchy.
- `brier._math`: stable float64 `logsumexp` and `norm` (log-softmax).
- `brier.prompts`: Choice/Noul/Score templates, HTML-escaped state, option rotation (ADR-0004).
- `Backend` protocol and deterministic `FakeBackend` (content signal + position bias + label prior).
- `raw` readout (with Score letter fallback) and `L0` debiasing: rotations, geometric-mean combine, batch prior (ADR-0005).
- `Decider` with `decide`, `decide_batch` and `fit_prior` (raw and L0), per-call limits (`max_questions=32`, `max_batch=64`) and provenance in `Decision.meta`.
- `brier.metrics`: accuracy, NLL, Brier, ECE (equal-mass and equal-width), flip rate, AURC, coverage at risk, MAE and RPS for Score, with seeded percentile and paired bootstrap CIs.
- Benchmark: `python -m brier.bench run` with task `banking20` (BANKING77, CC-BY-4.0, pinned and SHA-256 verified), raw/L0 with flip rate, bootstrap and paired CIs, result JSON plus per-item `.npz`.
- L1 benchmark (`docs/results.md`, Qwen3-1.7B on banking20): temperature T = 4.51 cuts ECE from 0.299 (L0) to 0.094 and NLL from 3.84 to 1.20, answers unchanged.
- Colab notebook `notebooks/m4_l1_banking20_benchmark.ipynb` (raw, L0, L1 on banking20, M4 acceptance check).
- Benchmark: level `L1` (`--levels raw,L0,L1`), temperature fitted on the calibration split; `calibration` and `L1_minus_L0` in the result JSON.
- Benchmark: level `L2` with label budgets (`--l2-budgets 100,200,300`) and a hidden-state cache (`--cache-dir`; no dataset text stored), `l2_curve` in the result JSON.
- L2 in `Decider`: `fit_head(states, question, labels, layers=None)` and `level="L2"`; heads saved in artifacts with schema version 2 (ADR-0006; version 1 still loads); L2 temperature fitted against Platt-smoothed targets so separable data no longer yields near-0/1 probabilities.
- `brier.heads.select`: L2 selection by stratified 5-fold OOF NLL over layers × α × {ridge, LDA}, refit on all labels, OOF temperature (`fit_temperature(log_t_bounds=...)`).
- `brier.heads`: L2 ridge head (dual and primal) and shrinkage-LDA head (Ledoit–Wolf γ from the Gram matrix, Woodbury when n < d) on standardised hidden states.
- `Decider.save(path)` / `Decider.load(path, backend)`: calibration round trip with identical decisions; load refuses artifacts from another model, revision or prompt template.
- `brier.artifacts`: calibration artifacts (`artifact.json` + `arrays.npz`, ADR-0003) with strict, untrusted-input loading: SHA-256, size caps incl. zip-bomb check, `allow_pickle=False`, model/revision/template-hash match; `prompts.template_hash()`.
- `Decider.fit_temperature(states, question, labels)` and `level="L1"`; `Decision.meta["temperature"]` records the fitted T.
- `brier.calibrate.temperature`: L1 temperature scaling — `apply_temperature` and `fit_temperature` (golden-section search on log T in [−3, 3], endpoints checked, ≥ 50 labelled items).
- First benchmark (`docs/results.md`, Qwen3-1.7B on banking20): L0 cuts flip rate from 36.4 % to 20.1 % and improves accuracy, NLL, Brier, ECE and AURC with paired CIs excluding zero.
- Colab notebook `notebooks/m3_4_banking20_benchmark.ipynb` for the M3.4 benchmark (Qwen3-1.7B, pinned brier commit and model revision, timing run, results table, download).
- Model compatibility matrix (`docs/COMPATIBILITY.md`): integration suite passes on Qwen3, SmolLM2, TinyLlama, OLMo-2 and Gemma 3 (`BRIER_TEST_MODELS=all`); `HFBackend(attn_implementation=...)` option (Gemma 3 needs `"eager"`).
- `HFBackend` (`brier.backends.hf`): safetensors-only loading with `trust_remote_code=False`, chat template with special-token-safe user content, left-padded batching with explicit position ids, strict label tokens, hidden states via hooks that stop after the deepest layer, and a hard `max_prompt_tokens` cap (default 9,216, at most the model context) raising `InputTooLargeError`. Requires `transformers>=5`.
### Security
- GitHub Actions and pre-commit hooks pinned to full commit SHAs.
