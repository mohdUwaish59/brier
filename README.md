# brier

> Calibrated typed decisions from open LLMs: choice, yes/no and score questions
> answered with real probabilities, read from the model in a forward pass, without generation
> or fine-tuning.

**Status: alpha.** The API may still change before 1.0. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Why

Reading an LLM's answer from generated text is slow and needs parsing. Raw next-token
probabilities skip that, but they are biased by option order and badly calibrated. brier reads
the answer distribution directly and corrects it in levels. Every decision records the level
that produced it.

| Level | Labels needed | What it fixes | Forward passes |
|---|---|---|---|
| `raw` | 0 | baseline readout of the label tokens | 1 |
| `L0` | 0 (optional unlabelled pool) | option-order bias, label prior bias | K for a K-option Choice, else 1 |
| `L1` | ≥ 50 | over/under-confidence (one temperature) | as L0 |
| `L2` | ≥ 60, ≥ 5 per answer | accuracy and calibration via a hidden-state head | 1 |

## Results

banking20 (20 BANKING77 intents), Qwen3-1.7B, 2,603 test items, 95 % bootstrap CIs.
L1 and L2 use 300 labels. Full tables, protocol and caveats: [`docs/results.md`](docs/results.md).

| | raw | L0 | L1 | L2 |
|---|---|---|---|---|
| Accuracy ↑ | 0.626 | 0.668 | 0.668 | **0.807** [0.793, 0.823] |
| ECE ↓ | 0.357 | 0.299 | 0.094 | **0.035** [0.028, 0.050] |
| NLL ↓ | 6.541 | 3.845 | 1.204 | **0.727** [0.672, 0.781] |

One model, one task so far: treat these as a sanity check, not a general claim.

## Install

```bash
pip install "brier[hf] @ git+https://github.com/mohdUwaish59/brier"   # until the PyPI release
```

Python ≥ 3.10. The core needs only numpy; the `hf` extra adds torch and transformers.

## Quickstart

```python
from brier import Choice, Decider, Noul
from brier.backends.hf import HFBackend

d = Decider(HFBackend("Qwen/Qwen3-1.7B", revision="<commit-sha>"))
route = Choice("Which team?", ["billing", "technical", "sales"], name="route")
refund = Noul("Is this a refund request?", name="refund")

res = d.decide("My card was charged twice, please fix it now!", [route, refund], level="L0")
print(res["route"].answer, res["route"].probs)  # top option and all option probabilities
print(res["refund"].p_yes, res["refund"].level)  # probability of yes, and "L0"
```

Calibrate with your own data, save the calibration, and reuse it:

```python
d.fit_prior(unlabelled_states, [route])  # L0 prior, no labels
d.fit_temperature(labelled_states, route, route_labels)  # L1, >= 50 labels
d.fit_head(labelled_states, route, route_labels)  # L2, >= 60 labels, >= 5 per answer
d.decide(state, [route], level="L2")

d.save("calibration/")  # JSON + .npz, never model weights
d = Decider.load("calibration/", HFBackend("Qwen/Qwen3-1.7B", revision="<commit-sha>"))
```

`load` refuses a calibration made for another model, revision or prompt template. Pin
`revision` to a commit SHA: a model name alone does not pin weights.

Questions:
- **`Choice(text, options, name=...)`**: one of K options.
- **`Noul(text, name=...)`**: yes/no; `p_yes` is the probability of yes.
- **`Score(text, levels, name=...)`**: an integer rating `1..levels`; `expected` is the mean.

Works with Hugging Face chat models that ship safetensors weights. Check yours before
calibrating:

```text
$ brier check Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca

  PASS  revision           c1899de289a04d12100db370d81485cdf75e47ca
  PASS  labels.choice      A-Z: single, distinct tokens
  PASS  labels.noul        Yes, No: single, distinct tokens
  PASS  labels.score       10 levels as lettered options (digit labels are not single tokens)
  PASS  batch_consistency  max |p_batched - p_single| = 5.3e-06 (limit 2e-02)
  PASS  hidden_states      7 candidate layers (12-24 of 28), d = 1024
  PASS  sanity             accuracy raw 79% / L0 79% on 14 items, order flips raw 50% / L0 17%, ...

Supported levels: raw, L0, L1, L2
```

Tested families are listed in [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Documentation

- [`docs/SPEC.md`](docs/SPEC.md): behaviour and limits
- [`docs/METHODS.md`](docs/METHODS.md): the maths of every level, with references
- [`docs/EVALUATION.md`](docs/EVALUATION.md) and [`docs/results.md`](docs/results.md): benchmark protocol and results
- [`docs/related_work.md`](docs/related_work.md): where the idea and the methods come from
- API reference, built from the docstrings into `site/`:
  `uv run pdoc brier brier.backends.hf brier.backends.base brier.backends.fake brier.errors brier.metrics -d numpy -o site`

## Security

Decisions on untrusted text can be manipulated by prompt injection. Don't use them as the only
gate for destructive actions. brier never runs model code (`trust_remote_code` is never set),
loads weights from safetensors only, and validates calibration files as untrusted input. Report
vulnerabilities privately; see [SECURITY.md](SECURITY.md) and
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Related work

brier follows the idea of TypeSafe AI's Jev and of
[AnyJev](https://github.com/nokia-applied-research/AnyJev) (Nokia Applied Research): typed
questions answered from next-token probabilities, corrected in levels. It is an independent
implementation, not affiliated with either. The correction methods come from published work
(Zheng et al. 2024; Zhou et al. 2024; Guo et al. 2017; Platt 1999; Ledoit & Wolf 2004).
[`docs/related_work.md`](docs/related_work.md) sets out what brier shares with these projects
and where it differs.

## License

Apache-2.0
