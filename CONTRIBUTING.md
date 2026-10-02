# Contributing

Thanks for helping! Bug reports, model results, docs and code are all welcome. For anything
larger than a small fix, open an issue first so we can agree on the approach.

## Set up

```bash
git clone https://github.com/mohdUwaish59/brier && cd brier
uv sync --all-extras          # https://docs.astral.sh/uv/
uv run pre-commit install
uv run pytest                 # unit tests: CPU, fast, no downloads
```

Before you open a PR, every gate must pass:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Easy ways to contribute

- **Add your model to the compatibility table.** Run
  `brier check <model-id> --revision <commit-sha>`, then open a PR adding a row to
  `docs/COMPATIBILITY.md` with the output. If the model is small enough, also add it to `MODELS`
  in `tests/integration/test_hf_backend.py` and run
  `BRIER_TEST_MODELS=<key> uv run pytest -m integration`.
- **Report a model that fails `brier check`.** Open an issue with the full output (check it for
  local paths first).
- **Add a benchmark task.** Tasks live in `src/brier/bench/tasks.py`; datasets must have a
  clear licence and be pinned (commit + SHA-256), like BANKING77. Describe the protocol in
  `docs/EVALUATION.md`.
- **Improve the docs**: anything that confused you is worth fixing.

Issues labelled *good first issue* are a good place to start.

## How the project works

- Read `docs/SPEC.md` (what the library does), `docs/ARCHITECTURE.md` (modules) and
  `docs/METHODS.md` (the maths) before larger changes.
- Decisions are recorded as ADRs in `docs/adr/`. Changes to the Backend interface, the artifact
  format, the prompt templates, or new runtime dependencies need a new ADR first.
- **The core is numpy-only**: `torch` and `transformers` are imported only in
  `brier/backends/hf.py`.
- **Write tests first.** Unit tests use `FakeBackend` and never download anything; real-model
  tests are marked `@pytest.mark.integration`.
- Maths must match `docs/METHODS.md`; update it in the same PR if you change a method.
- Security rules (`docs/THREAT_MODEL.md`): never `trust_remote_code`, safetensors only, no
  pickle, `eval` or `exec`, and calibration files are untrusted input.
- Use [Conventional Commit](https://www.conventionalcommits.org/) messages (`feat:`, `fix:`,
  `docs:`, `test:`, ...), one change per PR, and fill in the PR template. Add a line to
  `CHANGELOG.md` under `## [Unreleased]`.

`CLAUDE.md` holds the same rules in the form AI coding assistants read. AI-assisted
contributions are welcome; you are responsible for reviewing and understanding every line you
submit.

By contributing you agree that your work is licensed under Apache-2.0, and you agree to follow
the [Code of Conduct](CODE_OF_CONDUCT.md).
