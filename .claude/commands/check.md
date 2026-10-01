---
description: Run every quality gate and report
---
Run, in order, and stop at the first failure to report it clearly:
1. `uv run ruff format --check .`
2. `uv run ruff check .`
3. `uv run mypy src`
4. `uv run pytest -q --cov=declib --cov-report=term-missing`
Report a table of gate → pass/fail, the coverage total, and any file below 90 % coverage in `src/declib` (excluding `backends/hf.py` and `bench/`).
