# Contributing

Thanks for helping! Quick start:

```bash
git clone https://github.com/mohdUwaish59/brier && cd brier
uv sync --all-extras
uv run pre-commit install
uv run pytest
```

- Discuss larger changes in an issue first. Changes to the Backend interface, artifact
  format, prompt template or new runtime dependencies need an ADR (`docs/adr/`).
- Write tests first; keep the core numpy-only (see `CLAUDE.md` → Code standards,
  which apply to humans too).
- Maths must match `docs/METHODS.md`; update it in the same PR if you change a method.
- Use Conventional Commit messages and fill in the PR template.
- By contributing you agree your work is licensed under Apache-2.0.

AI-assisted contributions are welcome. You are responsible for reviewing and
understanding every line you submit.
