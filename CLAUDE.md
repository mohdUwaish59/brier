# CLAUDE.md — declib

`declib` (working name) is an open-source Python library that turns open causal LLMs
into **calibrated decision models**. The caller gives it a *state* (free text) and
*typed questions* (Choice / Noul / Score). It returns a probability distribution per
question, read from next-token logits or hidden states, with no text generation and
no fine-tuning.

Read these before any non-trivial change:
- `docs/SPEC.md`: what the library must do (public API, behaviour, limits)
- `docs/ARCHITECTURE.md`: module layout and interfaces
- `docs/METHODS.md`: the maths for every correction level (source of truth)
- `docs/EVALUATION.md`: metrics, datasets and benchmark protocol
- `docs/ROADMAP.md`: milestones and tasks. **Work is driven from here.**
- `docs/THREAT_MODEL.md`: security rules
- `docs/adr/`: decisions already made. Don't re-litigate them without a new ADR.

## Commands

```bash
uv sync --all-extras                  # install everything (dev, hf, bench)
uv run pytest                         # unit tests (CPU, fast, no network)
uv run pytest -m integration          # real-model tests (opt-in, needs a model download)
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pre-commit run --all-files     # every gate at once
```

A change is not done until `ruff`, `ruff format --check`, `mypy src` and `pytest` all pass.

## Workflow for every task

1. Pick the task from `docs/ROADMAP.md` (first unchecked task of the current milestone,
   unless the user names one). Restate its acceptance criteria.
2. Plan briefly: list the files you will touch. If the plan conflicts with SPEC,
   METHODS or an ADR, stop and ask.
3. **Write tests first** (unit tests using `FakeBackend`), then implement.
4. Run all gates. Fix the code, never the test's expectations, unless the test is wrong.
   If you think a test is wrong, say why.
5. Update docs affected by the change, add a line under `## [Unreleased]` in
   `CHANGELOG.md`, and tick the task in `ROADMAP.md`.
6. Commit on a feature branch with a Conventional Commit message
   (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`). One task per commit or PR.
7. For changes to maths or security-relevant code, ask the `math-verifier` or
   `security-auditor` subagent to review before committing.

## Code standards

- Python ≥ 3.10, full type hints, `mypy --strict` clean. Public API has NumPy-style docstrings.
- **The core is numpy-only.** `torch`/`transformers` are imported only inside
  `declib/backends/hf.py` (lazy import). Nothing else may import them.
- Calibration and metric maths: pure functions, `float64`, log-space where
  probabilities multiply, `scipy.special`-style stable logsumexp (implement in `_math.py`).
- No global mutable state. Randomness only through an explicit `numpy.random.Generator` / seed argument.
- Validate inputs at the public boundary and raise the library's own exceptions (`declib.errors`).
  Never `assert` for runtime validation.
- Every output carries the `level` that produced it (`raw`, `L0`, `L1`, `L2`).
- Keep functions small; prefer dataclasses (frozen) for value objects.
- Tests: `pytest`, `hypothesis` for invariants (probabilities sum to 1, rotation
  equivariance, temperature preserves argmax). Unit tests must not download anything.

## Hard rules

- Never set `trust_remote_code=True`. Load weights with safetensors only.
- Never use `pickle`, `torch.load` without `weights_only=True`, `eval`, `exec`, or
  `np.load(..., allow_pickle=True)`. Artifacts are JSON + `.npz` (see ADR-0003).
- No network access in library code except model/tokenizer loading that the user
  explicitly requests. No telemetry. Do not log state text by default.
- Do not add runtime dependencies without an ADR. Dev dependencies need a one-line justification in the PR.
- Do not weaken, skip or delete tests to make CI pass.
- Do not edit `LICENSE`, `SECURITY.md` or release workflows unless the task is about them.
- Do not push, publish, or rewrite git history. The human does that.
- Do not read `.env`, tokens or credentials.
- If requirements are ambiguous, ask instead of guessing.
