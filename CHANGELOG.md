# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `python -m brier.bench rotations` (ROADMAP M7.2): runs every L0 rotation once, stores the per-rotation log-probabilities and scores L0 with `m` evenly spaced rotations (accuracy, ECE, NLL, flip rate with bootstrap CIs), to measure how many rotations are needed. `brier.bench.rotations.subset_probs` scores any subset offline from the stored arrays.

## [0.1.1] - 2026-10-06

### Added
- Standard logging (SPEC §8): one stdlib logger per module under `brier` with a `NullHandler`; INFO records for model loading, `fit_prior` / `fit_temperature` / `fit_head` results, saved and loaded calibrations and benchmark progress; DEBUG batch progress. State text is never logged. `python -m brier.bench` now shows its progress.
- `HFBackend` warns when a model is loaded without a pinned revision.

### Changed
- `docs/related_work.md`: corrected the AnyJev comparison (AnyJev also selects the L2 layer and head type by out-of-fold NLL).
- README: logo, `brier check` GIF, six-family results chart, and a tagline that no longer implies a single forward pass.

## [0.1.0] - 2026-10-03

First public release.

### Added
- **Typed questions** `Choice`, `Noul` (yes/no) and `Score`, answered as probability
  distributions read from next-token log-probabilities (no generation, no fine-tuning). Every
  `Decision` records the level that produced it.
- **Correction levels** in `Decider` (`decide`, `decide_batch`):
  - `raw`: label-token readout (Score falls back to lettered options when digits are not single
    tokens);
  - `L0`, no labels: cyclic option rotations with a geometric-mean combine, plus an optional
    batch prior from unlabelled states (`fit_prior`; ADR-0005);
  - `L1`: temperature scaling (`fit_temperature`, at least 50 labels);
  - `L2`: hidden-state heads (`fit_head`, at least 60 labels and 5 per answer): ridge and
    Ledoit–Wolf shrinkage LDA, layer / solver / alpha chosen by stratified 5-fold out-of-fold
    NLL; L2 temperature with Platt (1999) targets for two classes and plain temperature
    scaling beyond (METHODS.md).
- **Calibration artifacts**: `Decider.save` / `Decider.load` (`artifact.json` + `arrays.npz`,
  never model weights), loaded as untrusted input (SHA-256, size caps, no pickle) and bound to
  the model id, revision, dtype and prompt templates (ADR-0003, 0006, 0007).
- **`HFBackend`** for Hugging Face causal LMs: safetensors only, `trust_remote_code` never set,
  special-token-safe prompts, batched readout and hidden states. Chat models are prompted through
  their chat template, base models through a plain-text prompt (ADR-0008).
- **`brier check <model-id>`**: a conformance report before calibrating (label tokens, batching,
  hidden states, a 14-item sanity task, supported levels); `--json`.
- **Benchmark** `python -m brier.bench run` (banking20 from BANKING77, CC-BY-4.0, pinned and
  SHA-256 verified): all levels, bootstrap and paired CIs, flip rate, L2 label curves, per-item
  outputs without state text. Results in `docs/results.md`.
- **Model compatibility**: integration suite and `brier check` pass on 15 models from Qwen3,
  SmolLM2/3, TinyLlama, OLMo-2, Gemma 3, Granite-MoE, LFM2, DeepSeek-R1-Distill, Falcon3,
  Phi-4-mini, OLMoE and Mistral, including two mixtures of experts and two base models
  (`docs/COMPATIBILITY.md`).
- Documentation: README quickstart (run in CI), API reference on GitHub Pages, SPEC, METHODS,
  ARCHITECTURE, EVALUATION, THREAT_MODEL, related work and eight ADRs.

### Security
- GitHub Actions and pre-commit hooks pinned to full commit SHAs; CodeQL, `pip-audit` and
  Dependabot in CI.

[Unreleased]: https://github.com/mohdUwaish59/brier/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mohdUwaish59/brier/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mohdUwaish59/brier/releases/tag/v0.1.0
