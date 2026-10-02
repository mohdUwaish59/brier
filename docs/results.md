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

L2 costs one forward pass per decision (68 s for 2 × 2,903 prompts incl. feature extraction).

### Reading

- **L2 improves every metric at once.** With 300 labels accuracy rises from 0.668 to 0.807
  (+13.9 points, paired CI [+12.1, +15.6] in the full M5.4 run, which used the same selected
  head) while ECE falls to 0.035 and NLL to 0.727 — better calibrated than L1 at much higher
  accuracy. 61 % of decisions can be automated at ≤ 5 % in-sample error (L1: 18 %).
- **More labels help, with diminishing returns:** 100 → 200 → 300 labels gives accuracy
  0.73 → 0.79 → 0.81 and ECE 0.063 → 0.033 → 0.035. Selection settles on ridge at blocks 22–24
  of 28 (≈ 80–85 % depth).
- **Calibration fix (M5.4b).** The first M5.4 run reached the same 0.807 accuracy but ECE 0.461:
  the L2 temperature was fitted against smoothed targets that capped confidence near 0.46 with
  20 classes. L2 now uses Platt targets for two classes and plain temperature scaling beyond
  (METHODS.md, L2); the numbers above are from the fixed version.
- **Flip rate barely moves** (0.20 → 0.19): L2 reads one prompt in the given option order, so it
  does not average position bias out the way L0 does.
