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
