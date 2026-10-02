# Roadmap

Each milestone ends in something usable. Work top to bottom; one task per commit or PR.
Tick `[x]` when the acceptance criteria are met and all gates pass.

## M0 — Repository foundation
- [x] **M0.1** Add the official Apache-2.0 text to `LICENSE` (from apache.org), fill the copyright line in `NOTICE`.
- [x] **M0.2** `uv lock` and commit `uv.lock`; `uv run pre-commit autoupdate` and `uv run pre-commit install`. CI green on Python 3.10–3.13.
- [x] **M0.3** Pin every GitHub Action in `.github/workflows/*` to a full commit SHA (keep the version as a comment).
- [ ] **M0.4** Enable in GitHub settings (human task): private vulnerability reporting, branch protection on `main` (CI required), Dependabot alerts, PyPI trusted publisher.
  *Accept:* CI passes on a PR; `pre-commit run --all-files` is clean.

## M1 — Core types and the maths (no model yet)
- [x] **M1.1** `errors.py`, `questions.py` (Choice/Noul/Score + validation per SPEC §2), `decision.py`.
- [x] **M1.2** `_math.py`: `logsumexp`, `log_softmax`, `norm`. Property tests: output sums to 1, stable for inputs of ±1e4.
- [x] **M1.3** `prompts.py`: render prompt, escape `<state>` delimiters inside state text, rotate options. Snapshot tests of the rendered prompt.
- [x] **M1.4** `backends/base.py` Protocol + `backends/fake.py` (deterministic logits with a configurable position bias and label prior, for tests).
- [x] **M1.5** `readout.py` (raw) and `debias.py` (rotations, combine, prior) exactly per METHODS.md.
  *Accept:* hypothesis tests for equivariance; with FakeBackend's position bias, L0 flip rate < raw flip rate.

## M2 — Hugging Face backend
- [x] **M2.1** `backends/hf.py`: load with `trust_remote_code=False`, `use_safetensors=True`, pinned `revision`; chat template; left padding; batched `label_logprobs`. Tokenise user content with special-token parsing disabled (a state containing `[INST]`/`<|im_end|>`-like text must not yield special token ids; test it).
- [x] **M2.2** `label_token_ids` with strict single-token check (raises `TokenizationError`).
- [x] **M2.3** `hidden_states` for chosen layers, stopping the forward pass after the deepest requested layer.
- [x] **M2.4** Integration tests on a small model (`@pytest.mark.integration`): batched equals unbatched (|Δ| < 1e-4 in fp32), padding side doesn't change results.
- [x] **M2.5** Model compatibility: run the integration suite on several model families
  (Qwen3, SmolLM2, TinyLlama/Llama-2 template, OLMo-2, Gemma 3 if accessible; Llama 3.2 when
  the human grants access). Fix what breaks in `HFBackend` (chat templates, label tokenisation,
  pad tokens, thinking modes). Document a compatibility table and how base models (no chat
  template) are handled.
  *Accept:* integration tests pass on ≥ 3 families with pinned revisions; table in `docs/COMPATIBILITY.md`.

## M3 — Decider (raw + L0) and metrics → **release 0.1.0a1**
- [x] **M3.1** `decider.py` with `decide`, `decide_batch`, `fit_prior`; input limits per SPEC §5. Apply the token limit to the rendered (escaped) prompt, not the raw state: escaping can grow it up to 5×. `HFBackend` already enforces a hard `max_prompt_tokens` cap; an exact per-state token limit needs a Backend token-count method (ADR first). Per ADR-0005: L0 applies the prior only if fitted, records that in `meta`, and keeps Score's prior off by default.
- [x] **M3.2** `metrics.py` + bootstrap CIs, each metric checked against hand-computed examples.
- [x] **M3.3** `bench/` task `banking20`, runner, result JSON + per-item `.npz`.
- [x] **M3.4** First benchmark on Qwen3-1.7B: raw vs L0 table in `docs/results.md`.
  *Accept:* L0 flip rate lower than raw with non-overlapping CIs.

## M4 — L1 and artifacts
- [x] **M4.1** `calibrate/temperature.py` (bounded search, no scipy). Test: recovers a known T on synthetic data.
- [x] **M4.1b** `Decider.fit_temperature(states, question, labels)` and `level="L1"` (labels typed like `Decision.answer`; prior refit discards the temperature).
- [x] **M4.1c** Benchmark runner level `L1`: T fitted on the calibration split, applied to test L0; `L1_minus_L0` paired comparison.
- [x] **M4.2** `artifacts.py`: JSON + `.npz` (allow_pickle=False), schema version, template hash, model id + revision, SHA-256 of the npz in the JSON; refuse to load on mismatch.
- [x] **M4.3** `Decider.save/load`; round-trip test gives identical decisions.
  *Accept:* L1 ECE lower than L0 on banking20 (paired CI). **Met:** 0.299 → 0.094, paired diff −0.206 [−0.213, −0.192] (`docs/results.md`).

## M5 — L2 hidden-state heads → **release 0.1.0**
- [x] **M5.1** `heads/ridge.py` (dual + primal), `heads/lda.py` (Ledoit–Wolf + Woodbury). Tests against a naive reference implementation on small random data.
- [x] **M5.2** `heads/select.py` stratified k-fold OOF selection over layers × α × solver, then temperature.
- [x] **M5.2b** `Decider.fit_head(states, question, labels, layers=None)` and `level="L2"`; store the selected head (layer, solver, alpha, arrays, T, OOF metrics) in artifacts (schema_version 2, ADR first).
- [x] **M5.3** Feature caching in the benchmark (compute hidden states once per model/task).
- [x] **M5.4** Benchmark L2 label curve (100/200/300 labels: L2 needs ≥ 5 per class, i.e. ≥ 100 on banking20's 20 classes) on Qwen3-1.7B and 4B. *Done on Qwen3-1.7B (`docs/results.md`); the 4B run is optional and not yet done.*
- [ ] **M5.5** Docs: README quickstart, API reference (mkdocs or pdoc), `docs/results.md`. Tag 0.1.0. *Docs done (README, `docs/related_work.md`, pdoc); the 0.1.0 tag is the maintainer's.*

## Later (each needs an ADR first)
- Certified abstention: split-conformal sets and Learn-then-Test thresholds.
- One state, many questions: shared-prefix KV reuse.
- Ordinal (cumulative-logit) heads for Score.
- More than 26 options (shortlist + rotated comparison).
- vLLM backend.
