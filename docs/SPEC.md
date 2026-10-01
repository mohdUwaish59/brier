# Specification — brier v0.1

## 1. Purpose

Given an open causal LM, a **state** (arbitrary text) and one or more **typed
questions**, return for each question a calibrated probability distribution over
its allowed answers, in one or a few forward passes and with no generation.

Non-goals for v0.1: training or fine-tuning, closed/API-only models, generation,
more than 26 options per Choice, multi-label questions, vLLM/SGLang serving.

## 2. Question types

| Type | Answers | Output |
|---|---|---|
| `Choice(text, options, name)` | 2–26 unique option strings | `probs: dict[str, float]`, `answer: str` |
| `Noul(text, name)` | yes / no | `p_yes: float`, `answer: bool` |
| `Score(text, levels, name, labels=None)` | integer levels `1..L`, `2 ≤ L ≤ 10`, optional level descriptions | `probs` over levels, `expected: float`, `answer: int` (mode) |

Validation (raise `QuestionError`): empty text; duplicate or empty options; option
count outside range; option strings longer than 500 chars; line breaks in option or
level-label strings; duplicate `name` within one call.

## 3. Public API (target)

```python
from brier import Decider, Choice, Noul, Score
from brier.backends.hf import HFBackend

d = Decider(HFBackend("Qwen/Qwen3-1.7B", revision="<commit-sha>"))
qs = [
    Choice(
        "Which team should handle this?", ["billing", "technical", "sales", "other"], name="route"
    ),
    Noul("Is the customer asking for a refund?", name="refund"),
    Score("How urgent is this?", levels=5, name="urgency"),
]
res = d.decide("My card was charged twice, fix it now!", qs, level="L0")
res["route"].answer, res["route"].probs, res["route"].level  # "billing", {...}, "L0"
```

- `Decider.decide(state, questions, level="L0") -> dict[str, Decision]`
- `Decider.decide_batch(states, questions, level=...) -> list[dict[str, Decision]]`
- `Decider.fit_prior(states, questions)`: label-free prior estimate for L0 (unlabelled states).
  Priors are keyed by the whole question (text, options/levels, name), so a changed
  question never reuses a stale prior; Score is skipped unless `Decider(score_prior=True)`.
- `Decider.fit_temperature(states, question, labels)`: L1, one question per call, ≥ 50 states.
  Labels are typed like `Decision.answer`: option string (Choice), `bool` (Noul), int level
  `1..L` (Score). `T` is fitted on top of L0 (with its prior, if fitted) and keyed by the whole
  question; refitting that question's prior discards its temperature. `level="L1"` raises
  `NotFittedError` for any question without a temperature.
- `Decider.fit_head(states, question, labels, layers=None)`: L2
- `Decider.save(path)` / `Decider.load(path, backend)`: calibration artifacts, never weights

Requesting a level that has not been fitted raises `NotFittedError`. It never
silently falls back to a lower level.
Exception: `L0` without `fit_prior` is rotation-only and does not raise (ADR-0005).

`Decision` (frozen dataclass): `name, type, probs, answer, level, confidence`
(max prob), `meta` (model id, revision, layer for L2, n_forward).
`probs` is keyed by option string (Choice), `"yes"`/`"no"` (Noul) or `"1"`..`"L"` (Score),
in question order. `answer` and `confidence` are derived from `probs` (ties go to the
first-listed answer); Noul adds `p_yes`, Score adds `expected`. Construction rejects
probabilities that are non-finite, negative or do not sum to 1 (±1e-9).

## 4. Correction levels

| Level | Labels | Summary (details in METHODS.md) |
|---|---|---|
| `raw` | 0 | Softmax restricted to label tokens at last prompt position |
| `L0` | 0 | Cyclic rotations of options + geometric-mean combine + batch prior correction |
| `L1` | 50–500 | Temperature scaling on L0 log-probs |
| `L2` | 100–300 per question | Closed-form head on hidden state of a middle layer, temperature fitted out-of-fold |

## 5. Limits and guarantees

- Probabilities returned always sum to 1 (±1e-9) and contain no NaN.
- `L1` never changes the argmax of `L0`.
- `raw` and `L0` are deterministic for a given backend, model revision and dtype.
- Default limits (configurable): state ≤ 8,192 tokens, ≤ 32 questions per call,
  batch ≤ 64. Exceeding them raises `InputTooLargeError`. No silent truncation.
- The library makes **no claim** that decisions are safe on adversarial input
  (see THREAT_MODEL.md).

## 6. Benchmark CLI

`python -m brier.bench run --model <id> --task banking20 --levels raw,L0,L1,L2 --out results/`
writes a JSON result file (schema in EVALUATION.md).
