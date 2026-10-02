# Benchmark results

Protocol: [`EVALUATION.md`](EVALUATION.md). Every value is a point estimate with a 95 %
percentile bootstrap CI (1,000 seeded resamples); level differences use a paired bootstrap on
the same items. Raw result files live in [`benchmarks/`](benchmarks/).

## banking20 — Qwen3-1.7B, raw vs L0 (M3.4)

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B` @ `70d244c`, bfloat16, CUDA (Colab A100) |
| brier | commit `3d717e3` (notebook `notebooks/m3_4_banking20_benchmark.ipynb`) |
| Data | BANKING77 train CSV (CC-BY-4.0, SHA-256 `b06e26ac…`), top-20 intents, split seed 0 |
| Items | 2,603 test, 500 unlabelled prior pool (L0 prior), 20-option Choice question |
| Runtime | raw 34 s, L0 796 s (incl. prior fit; 20 rotations × 2 option orders) |

| Metric | raw | L0 | L0 − raw (paired) |
|---|---|---|---|
| Flip rate ↓ | 0.364 [0.347, 0.383] | **0.201** [0.186, 0.217] | **−0.164** [−0.181, −0.146] |
| Accuracy ↑ | 0.626 [0.607, 0.644] | **0.668** [0.650, 0.687] | **+0.043** [+0.030, +0.054] |
| NLL ↓ | 6.541 [6.162, 6.901] | **3.845** [3.594, 4.101] | **−2.696** [−2.924, −2.446] |
| Brier ↓ | 0.726 [0.690, 0.762] | **0.617** [0.584, 0.652] | **−0.109** [−0.130, −0.085] |
| ECE (15 equal-mass bins) ↓ | 0.357 [0.338, 0.375] | **0.299** [0.282, 0.317] | **−0.057** [−0.070, −0.044] |
| ECE (equal-width) ↓ | 0.357 [0.339, 0.376] | **0.299** [0.282, 0.317] | −0.057 [−0.070, −0.045] |
| AURC ↓ | 0.196 [0.180, 0.215] | **0.146** [0.132, 0.160] | **−0.050** [−0.061, −0.038] |
| Coverage at risk ≤ 5 % (in-sample) ↑ | 0.154 [0.040, 0.186] | 0.151 [0.123, 0.304] | −0.003 [−0.038, +0.165] |

**M3.4 acceptance:** L0 flip rate below raw with non-overlapping CIs — **pass**
(L0 upper bound 0.217 < raw lower bound 0.347).

### Reading

- **Order bias is real and L0 removes much of it.** Raw changes its answer on 36 % of items
  when the option list is reversed; L0 on 20 %. L0 is not fully order-invariant on a real
  model: reversal is not one of the cyclic rotations it averages over, and the two orders get
  separately fitted priors.
- **L0 improves every proper scoring rule** (NLL, Brier) and accuracy, with paired CIs that
  exclude zero.
- **Probabilities are still overconfident.** ECE ≈ 0.30 and NLL ≈ 3.8 at 67 % accuracy mean
  confidences run well above accuracy. This is what L1 (temperature scaling, M4) targets.
- **Coverage at 5 % risk is unstable** at this accuracy: it is set by the few most confident
  items, hence the wide, skewed CIs. Treat it as inconclusive here.

## banking20 — Qwen3-1.7B, raw vs L0 vs L1 (M4)

Same model revision, data, split and hardware as M3.4; brier commit `3988cb7` (notebook
`notebooks/m4_l1_banking20_benchmark.ipynb`). L1 fits a temperature on L0 predictions for the
300-item labelled calibration split and applies it to the 2,603 test items. Fitted
**T = 4.51** (reversed option order: 4.20). Raw and L0 numbers reproduce the M3.4 run exactly.

| Metric | raw | L0 | L1 | L1 − L0 (paired) |
|---|---|---|---|---|
| ECE (15 equal-mass bins) ↓ | 0.357 | 0.299 | **0.094** [0.081, 0.112] | **−0.206** [−0.213, −0.192] |
| ECE (equal-width) ↓ | 0.357 | 0.299 | **0.087** [0.073, 0.103] | **−0.213** [−0.220, −0.201] |
| NLL ↓ | 6.541 | 3.845 | **1.204** [1.148, 1.266] | **−2.641** [−2.840, −2.451] |
| Brier ↓ | 0.726 | 0.617 | **0.470** [0.447, 0.494] | **−0.147** [−0.161, −0.134] |
| AURC ↓ | 0.196 | 0.146 | **0.139** [0.126, 0.153] | **−0.007** [−0.009, −0.005] |
| Coverage at risk ≤ 5 % (in-sample) ↑ | 0.154 | 0.151 | 0.180 [0.142, 0.363] | +0.030 [−0.003, +0.141] |
| Accuracy | 0.626 | 0.668 | 0.668 | 0.000 (by design) |
| Flip rate | 0.364 | 0.201 | 0.201 | 0.000 (by design) |

**M4 acceptance:** L1 ECE lower than L0 with a paired CI below zero — **pass**.

### Reading

- **L0's probabilities were ~4.5× too sharp.** One temperature fitted on 300 labelled items
  cuts ECE from 0.30 to 0.09 and NLL from 3.8 to 1.2; both option orders give a similar T.
- **Not fully calibrated yet.** ECE ≈ 0.09 means confidence still misses accuracy by about
  9 points on average; a single global temperature cannot fix miscalibration that differs
  between intents (the target of L2 heads or richer calibrators).
- **Answers are unchanged:** accuracy and flip rate equal L0 because a temperature never
  changes the argmax. AURC improves slightly because confidence is the max probability of the
  rescaled distribution, which can reorder items.
- **Reproducible:** raw and L0 match the M3.4 run to every digit on a fresh Colab machine.

## banking20 — Qwen3-1.7B, L2 hidden-state head (M5.4)

Same model revision, data and split as above; L2 heads per label budget, each budget a seeded
class-balanced subsample of the 300-item calibration split; selection over layers
(every 2nd block, 40–90 % depth) × ridge α × LDA by 5-fold out-of-fold NLL. brier commit
`faa2cc9` (L2-only notebook `notebooks/m5_4b_l2_rerun.ipynb`), raw / L0 / L1 from the M4 run.

| Metric | raw | L0 | L1 | **L2 (300 labels)** |
|---|---|---|---|---|
| Accuracy ↑ | 0.626 | 0.668 | 0.668 | **0.807** [0.793, 0.823] |
| ECE (15 equal-mass bins) ↓ | 0.357 | 0.299 | 0.094 | **0.035** [0.028, 0.050] |
| NLL ↓ | 6.541 | 3.845 | 1.204 | **0.727** [0.672, 0.781] |
| Brier ↓ | 0.726 | 0.617 | 0.470 | **0.284** [0.264, 0.302] |
| AURC ↓ | 0.196 | 0.146 | 0.139 | **0.065** [0.054, 0.077] |
| Coverage at risk ≤ 5 % (in-sample) ↑ | 0.154 | 0.151 | 0.180 | **0.612** [0.541, 0.663] |
| Flip rate ↓ | 0.364 | 0.201 | 0.201 | **0.192** [0.177, 0.207] |

Label curve (L2 head chosen per budget):

| Labels | Layer | Solver | α | T | Accuracy | ECE | NLL |
|---|---|---|---|---|---|---|---|
| 100 | 22 | ridge | 1000 | 0.107 | 0.730 [0.713, 0.747] | 0.063 [0.051, 0.079] | 0.978 [0.919, 1.034] |
| 200 | 24 | ridge | 100 | 0.138 | 0.788 [0.773, 0.804] | 0.033 [0.031, 0.054] | 0.811 [0.759, 0.865] |
| 300 | 22 | ridge | 100 | 0.128 | 0.807 [0.793, 0.823] | 0.035 [0.028, 0.050] | 0.727 [0.672, 0.781] |

L2 costs one forward pass per decision (68 s wall time for the whole L2 run on an A100).

### Reading

- **L2 improves every metric at once.** With 300 labels accuracy rises from 0.668 to 0.807
  (+13.9 points; the two 95 % CIs do not overlap) while ECE falls to 0.035 and NLL to
  0.727 — better calibrated than L1 at much higher accuracy. 61 % of decisions can be automated at ≤ 5 % in-sample error (L1: 18 %).
- **More labels help, with diminishing returns:** 100 → 200 → 300 labels gives accuracy
  0.73 → 0.79 → 0.81 and ECE 0.063 → 0.033 → 0.035. Selection settles on ridge at blocks 22–24
  of 28 (≈ 80–85 % depth).
- **Calibration fix (M5.4b).** The first M5.4 run reached the same 0.807 accuracy but ECE 0.461:
  the L2 temperature was fitted against smoothed targets that capped confidence near 0.46 with
  20 classes. L2 now uses Platt targets for two classes and plain temperature scaling beyond
  (METHODS.md, L2); the numbers above are from the fixed version.
- **Flip rate barely moves** (0.20 → 0.19): L2 reads one prompt in the given option order, so it
  does not average position bias out the way L0 does.


## banking20 — six model families (M6.5)

Same task, split, seed, prompts and labels as above (2,603 test items; L1 and L2 fitted on the
300-item calibration split; 500 unlabelled items for the L0 prior); every model in bfloat16 on an
A100 with a pinned revision. Qwen3-1.7B is the run above; the five others were run with
`notebooks/m6_5_cross_family_benchmark.ipynb` at brier `355f90d`.

**Point estimates only.** The other models' result files, with bootstrap CIs and NLL, were lost
when the Colab runtime disconnected; the numbers below are the run's printed output, kept
verbatim in `docs/benchmarks/m6_5_cross_family_log.txt`. With 2,603 test items the 95 % CIs
on accuracy are about ±0.015 (see the Qwen3-1.7B tables). Mistral-7B-v0.3 did not finish and is
not reported.

Accuracy ↑

| Model | raw | L0 | L1 | L2 (300 labels) |
|---|---|---|---|---|
| Qwen3-1.7B (chat) | 0.626 | 0.668 | 0.668 | 0.807 |
| Falcon3-1B-**Base** (plain prompt) | 0.155 | 0.601 | 0.601 | 0.804 |
| LFM2-1.2B (hybrid conv.) | 0.200 | 0.617 | 0.617 | 0.802 |
| SmolLM3-3B | 0.660 | 0.746 | 0.746 | 0.834 |
| Phi-4-mini (3.8B) | 0.711 | 0.786 | 0.786 | 0.859 |
| OLMoE-1B-7B (MoE) | 0.257 | 0.686 | 0.686 | 0.819 |

ECE ↓

| Model | raw | L0 | L1 | L2 (300 labels) |
|---|---|---|---|---|
| Qwen3-1.7B (chat) | 0.357 | 0.299 | 0.094 | 0.035 |
| Falcon3-1B-**Base** (plain prompt) | 0.119 | 0.519 | 0.162 | 0.072 |
| LFM2-1.2B (hybrid conv.) | 0.336 | 0.280 | 0.064 | 0.036 |
| SmolLM3-3B | 0.190 | 0.084 | 0.030 | 0.039 |
| Phi-4-mini (3.8B) | 0.186 | 0.116 | 0.022 | 0.057 |
| OLMoE-1B-7B (MoE) | 0.068 | 0.472 | 0.098 | 0.084 |

Flip rate ↓

| Model | raw | L0 | L1 | L2 (300 labels) |
|---|---|---|---|---|
| Qwen3-1.7B (chat) | 0.364 | 0.201 | 0.201 | 0.192 |
| Falcon3-1B-**Base** (plain prompt) | 0.990 | 0.371 | 0.371 | 0.136 |
| LFM2-1.2B (hybrid conv.) | 0.972 | 0.415 | 0.415 | 0.191 |
| SmolLM3-3B | 0.279 | 0.123 | 0.123 | 0.162 |
| Phi-4-mini (3.8B) | 0.275 | 0.099 | 0.099 | 0.117 |
| OLMoE-1B-7B (MoE) | 0.945 | 0.316 | 0.316 | 0.193 |

### Reading

- **Zero labels, every family: L0 raises accuracy and cuts order flips.** The gains are largest
  where the raw readout is near chance: Falcon3-1B-Base 0.155 → 0.601, LFM2 0.200 → 0.617,
  OLMoE 0.257 → 0.686. Their raw flip rates of 0.95–0.99 show what went wrong: the raw answer
  follows the option *position*, not the content. Rotating the options removes most of that.
- **300 labels: L2 lands every model between 0.80 and 0.86 accuracy**, from raw accuracies that
  range from 0.155 to 0.711. A 1B **base** model (no instruction tuning, plain-text prompt)
  reaches 0.804, on par with the 1.7B chat model's 0.807.
- **Calibration: L1 lowers ECE relative to L0 on every model** (to 0.022–0.162), and L2 reaches
  0.035–0.084. On SmolLM3 and Phi-4-mini, L1 is better calibrated than L2 (0.030 vs 0.039,
  0.022 vs 0.057) while L2 is more accurate: choose by what matters for your use.
- **Low raw ECE can be misleading.** Falcon3-1B-Base and OLMoE have raw ECE of 0.119 and 0.068
  at near-chance accuracy: they are unconfident *and* wrong, which ECE alone scores as well
  calibrated. L0 makes them confident (ECE 0.519 and 0.472) and L1 brings that back down (0.162
  and 0.098). Read ECE together with accuracy.
- **L2 does not always flip less than L0**: on SmolLM3 and Phi-4-mini it flips slightly more
  (0.162 vs 0.123, 0.117 vs 0.099). L2 reads one prompt in the given order and does not average
  out position bias the way L0 does.
- **Limits:** one task, one seed, point estimates without intervals for five of the six models.
