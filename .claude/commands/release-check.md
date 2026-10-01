---
description: Verify the repo is ready to tag a release
argument-hint: <version, e.g. 0.1.0>
---
Prepare release $ARGUMENTS (do not tag, push or publish):
1. All gates pass (`/check`) and `uv run pytest -m integration` passes if a model is available.
2. `__version__` and `pyproject.toml` version equal $ARGUMENTS.
3. Move CHANGELOG `Unreleased` entries under `## [$ARGUMENTS] - <today>`.
4. `uv build` succeeds; inspect the sdist/wheel file list for stray files (no tests data, no secrets, no results/).
5. README quickstart runs as written.
6. Security-auditor subagent passes.
Report a checklist and the exact commands the human should run to tag and release.
