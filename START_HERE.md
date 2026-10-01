# Start here (for you, not for the repo — delete after setup)

## One-time setup
1. Pick the final package name (check it's free on PyPI; avoid "Jev"). To rename:
   `grep -rl declib . | xargs sed -i 's/declib/<newname>/g'` and rename `src/declib`.
2. Create the GitHub repo, then copy these files in and commit:
   `git init && git add . && git commit -m "chore: scaffold"`.
3. Install uv (https://docs.astral.sh/uv/) and run `uv lock && uv sync --all-extras`.
4. Do task M0.4 by hand in GitHub settings.

## Working with Claude Code
- Start a session in the repo root. Claude Code reads `CLAUDE.md` automatically.
- First prompt:
  > Read CLAUDE.md and every file in docs/. Summarise the plan in 10 lines and list anything
  > ambiguous or contradictory. Then run /next-task M0.1.
- After that, run one task per session: `/next-task` → review the diff yourself → `/review`
  → push and open a PR. Keep sessions short; the docs carry the context between them.
- Run `/check` any time. Run `/release-check 0.1.0` before tagging.

## What you should still review yourself
- Every change to `docs/METHODS.md`, `docs/SPEC.md` and ADRs (the agent follows these blindly).
- Benchmark numbers before putting them in the README.
- Anything the reviewers flag as a decision for the human.
