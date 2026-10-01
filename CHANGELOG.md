# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Changed
- Renamed the project from the working name `declib` to `brier` (package `brier`, base exception `BrierError`).
### Removed
- `datasets` from the `bench` extra (BANKING77 is read from its CSV directly).
### Fixed
- mypy no longer pins `python_version = "3.10"`, which failed on Python 3.12+ where numpy 2.5 stubs use `type` statements.
### Added
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
- First benchmark (`docs/results.md`, Qwen3-1.7B on banking20): L0 cuts flip rate from 36.4 % to 20.1 % and improves accuracy, NLL, Brier, ECE and AURC with paired CIs excluding zero.
- Colab notebook `notebooks/m3_4_banking20_benchmark.ipynb` for the M3.4 benchmark (Qwen3-1.7B, pinned brier commit and model revision, timing run, results table, download).
- Model compatibility matrix (`docs/COMPATIBILITY.md`): integration suite passes on Qwen3, SmolLM2, TinyLlama, OLMo-2 and Gemma 3 (`BRIER_TEST_MODELS=all`); `HFBackend(attn_implementation=...)` option (Gemma 3 needs `"eager"`).
- `HFBackend` (`brier.backends.hf`): safetensors-only loading with `trust_remote_code=False`, chat template with special-token-safe user content, left-padded batching with explicit position ids, strict label tokens, hidden states via hooks that stop after the deepest layer, and a hard `max_prompt_tokens` cap (default 9,216, at most the model context) raising `InputTooLargeError`. Requires `transformers>=5`.
### Security
- GitHub Actions and pre-commit hooks pinned to full commit SHAs.
