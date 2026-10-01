# brier

> Calibrated typed decisions from open LLMs: choice, yes/no and score questions
> answered with real probabilities, in one forward pass, without generation or fine-tuning.

**Status: pre-alpha, under active development.** See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Why

Reading an LLM's answer from generated text is slow and needs parsing, and raw
next-token probabilities are biased by option order and badly calibrated. brier reads the
answer distribution directly and corrects it in levels:

| Level | Labels needed | What it fixes |
|---|---|---|
| `raw` | 0 | baseline readout |
| `L0` | 0 | option-order bias, label prior bias |
| `L1` | ~100 | over/under-confidence (temperature) |
| `L2` | ~100–300 per question | accuracy and calibration via a hidden-state head |

## Quickstart (target API)

```python
from brier import Decider, Choice, Noul
from brier.backends.hf import HFBackend

d = Decider(HFBackend("Qwen/Qwen3-1.7B"))
res = d.decide(
    "My card was charged twice, please fix it now!",
    [
        Choice("Which team?", ["billing", "technical", "sales"], name="route"),
        Noul("Is this a refund request?", name="refund"),
    ],
    level="L0",
)
print(res["route"].answer, res["route"].probs)
```

Works with Hugging Face chat models that ship safetensors weights; tested families are listed
in [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Install

```bash
pip install "brier[hf]"   # after the first release
```

## Security

Decisions on untrusted text can be manipulated by prompt injection. Don't use them as the
only gate for destructive actions. Report vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## Related work

Inspired by TypeSafe AI's Jev and by [AnyJev](https://github.com/nokia-applied-research/AnyJev).
Not affiliated with either. Methods and references: [`docs/METHODS.md`](docs/METHODS.md).

## License

Apache-2.0
