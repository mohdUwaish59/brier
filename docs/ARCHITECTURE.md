# Architecture

```
src/declib/
  __init__.py          # public exports + __version__
  errors.py            # DeclibError and subclasses
  questions.py         # Choice, Noul, Score (frozen dataclasses + validation)
  decision.py          # Decision result type
  prompts.py           # prompt rendering, state delimiting, rotations of option order
  _math.py             # logsumexp, log_softmax, safe normalisation (float64)
  readout.py           # raw: label-token log-probs -> distribution
  debias.py            # L0: rotation combine, batch / content-free prior
  calibrate/
    temperature.py     # L1: fit/apply temperature (no scipy; bounded 1-D search)
  heads/
    ridge.py           # L2 dual/primal ridge head
    lda.py             # L2 shrinkage LDA head
    select.py          # k-fold out-of-fold model selection (layer, alpha, solver, T)
  artifacts.py         # save/load calibration state (JSON + npz, schema-versioned)
  decider.py           # orchestration; the only class most users touch
  metrics.py           # accuracy, NLL, Brier, ECE, flip rate, risk-coverage, bootstrap CI
  backends/
    base.py            # Backend Protocol
    fake.py            # deterministic backend for tests
    hf.py              # transformers backend (lazy torch import)
  bench/
    tasks.py           # dataset loaders -> (state, question, label) items
    run.py             # benchmark runner + result JSON writer
    __main__.py        # CLI
tests/
  unit/                # CPU, FakeBackend, fast, no network
  integration/         # real small model, marked @pytest.mark.integration
```

## Backend interface (the only boundary to model code)

```python
class Backend(Protocol):
    model_id: str
    revision: str | None
    num_layers: int

    def label_token_ids(self, labels: Sequence[str]) -> list[int]:
        """Token id of each label as it appears right after the answer prefix.
        Raise TokenizationError if any label is not a single, distinct token."""

    def label_logprobs(self, prompts: Sequence[str], token_ids: Sequence[int]) -> np.ndarray:
        """(n_prompts, n_labels) float64 log-probs of the given tokens at the next position
        (full-vocab log_softmax, then gathered). Batched internally."""

    def hidden_states(self, prompts: Sequence[str], layers: Sequence[int]) -> np.ndarray:
        """(n_prompts, n_layers, d) float32 residual-stream state at the last prompt token.
        May stop the forward pass after max(layers)."""
```

Everything after these calls is numpy and runs on CPU. This makes all maths unit-testable
with `FakeBackend` and lets future backends (vLLM, SGLang) be added without touching the core.

## Data flow (L0 Choice)

state + Choice → `prompts.render` × K rotations → `backend.label_logprobs` →
map rotated positions back to options → `debias.combine` (mean of log-probs, renormalise) →
`debias.apply_prior` → `Decision(level="L0")`.

## Prompt layout (state first; keeps prefix sharing possible later)

```
<chat template: system> You answer multiple-choice questions about the text in <state> tags.
<user>
<state>
{escaped state}
</state>
Question: {question}
A. {option}
B. {option}
...
Answer with the letter only.
<assistant> Answer:        # generation prompt; label tokens read at the next position
```
The exact template string lives in `prompts.py` and is covered by snapshot tests.
Changing it is a breaking change for stored artifacts (artifacts store a template hash).
