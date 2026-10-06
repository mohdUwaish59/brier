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
- `Decider.fit_head(states, question, labels, layers=None)`: L2. At least 60 labelled states
  and 5 per answer; labels typed like `Decision.answer`. Reads the hidden state at the last
  prompt token (prompt rendered without rotation; Score with digits) for the candidate layers
  (default every 2nd block from 40 % to 90 % depth), selects layer, solver and alpha by
  out-of-fold NLL and fits the L2 temperature on the out-of-fold scores (Platt targets for two
  classes, plain NLL beyond; METHODS.md, L2). Warns if
  the temperature hits its lower bound. `level="L2"` raises `NotFittedError` for any question
  without a head. Heads are saved in artifacts (schema version 2+, ADR-0006).
- `Decider.save(path)` / `Decider.load(path, backend)`: calibration artifacts, never weights.
  `save` writes every fitted prior and temperature plus `prior_strength` (ADR-0003). `load`
  validates the artifact as untrusted input and refuses it unless it was fitted on the
  backend's exact model id, revision and precision (`dtype`, schema version 3, ADR-0007) with
  the current prompt templates; decisions after a round trip are identical. Re-fit a
  calibration to use it at another precision. `revision=None` matches by name only and does not pin weights, so
  pass a commit SHA. The SHA-256 checks the arrays file against its JSON; it does not prove
  who wrote the artifact, so only load artifacts from sources you trust.

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

## 7. Conformance check

`brier check <model-id> [--revision SHA] [--dtype ...] [--device ...] [--batch-size N]
[--attn-implementation ...] [--json]` (also `python -m brier check`) loads the model through
`HFBackend` and reports, on built-in inputs only:

- `revision`: warns if the revision is not pinned;
- `labels.choice` / `labels.noul` / `labels.score`: the labels `A`-`Z`, `Yes`/`No` and 10
  Score levels are single, distinct tokens (Score may use the letter fallback);
- `batch_consistency`: label probabilities from a batched and a single-prompt pass differ by
  at most 0.02 (padding or masking bugs move them by more);
- `hidden_states`: the default L2 candidate layers are reachable and finite;
- `sanity`: 14 built-in items (Choice, Noul, Score): raw and L0 accuracy, order flips under a
  reversed option list, ms per prompt. Warns below 60 % L0 accuracy (a weak model), never fails
  on accuracy alone.

It prints which levels the model supports (`raw`/`L0`/`L1` need the label checks, `L2` needs
hidden states). Exit code 0 if no check failed (warnings allowed), 1 if the model could not be
loaded or a check failed. The Python API is `brier.check.check_backend(backend)`. Failure
details quote the exception text, which can contain local paths or URLs: review a report
before posting it publicly.

## 8. Logging

brier logs through the standard library: one logger per module under `brier` (for example
`brier.decider`, `brier.backends.hf`), with a `NullHandler` on `brier`. The library never adds
other handlers, sets levels or prints; the application decides what to show, e.g.
`logging.basicConfig(level=logging.INFO)` or `logging.getLogger("brier").setLevel(...)`.

- **INFO:** a model loaded (id, revision, dtype, device, prompt format, layer count);
  `fit_prior`, `fit_temperature` and `fit_head` finished (question name, item count, fitted T,
  chosen L2 layer / solver / alpha, out-of-fold accuracy, time); a calibration saved or loaded
  (path, schema version, question count, model); benchmark progress per level.
- **DEBUG:** batch progress inside the backend.
- **Never logged:** state text, rendered prompts or labels (THREAT_MODEL T6). Individual
  decisions are not logged at INFO.
- **Warnings** (`warnings.warn`) are for things to act on: an L2 temperature at its lower bound,
  and a model loaded without a pinned revision.

`python -m brier.bench` is an application and shows INFO logs; `brier check` prints its own
report.
