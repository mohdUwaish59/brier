---
description: Full review of the current branch before merging
---
Review all changes on the current branch versus `main`:
1. Run the code-reviewer subagent.
2. Run the math-verifier subagent if any file in readout.py, debias.py, calibrate/, heads/ or metrics.py changed.
3. Run the security-auditor subagent if any file in backends/, artifacts.py, prompts.py, pyproject.toml, uv.lock or .github/ changed.
Merge their findings into one prioritised list with a final verdict.
