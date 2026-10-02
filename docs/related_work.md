# Related work

brier did not invent its idea or its methods. This page sets out where they come from, what
brier shares with the projects closest to it, and where it differs.

## The idea: Jev and AnyJev

**Jev** (TypeSafe AI) framed LLM decisions as *typed questions* (a choice, a yes/no, a score)
answered from next-token probabilities rather than generated text.
**[AnyJev](https://github.com/nokia-applied-research/AnyJev)** (Nokia Applied Research,
Apache-2.0) is an open implementation of that idea for open models, with the levels
raw → L0 → L1 → L2.

brier started from the same idea and uses the same level names and the same banking20 task,
so that results can be compared. It is an independent implementation, written from the
published methods below and not from either project's code, and it is not affiliated with
either. The AnyJev source was first read after brier's L0–L2 were built (M5.3); that reading
informed the diagnosis of a calibration bug (M5.4b, see METHODS.md), not the design.

## The methods: published work

| Level | Method | Source |
|---|---|---|
| raw | softmax over the answer-label tokens | standard practice |
| L0 | cyclic option rotations, geometric-mean combine | Zheng et al. 2024, arXiv:2309.03882 |
| L0 | batch prior correction | Zhou et al. 2024, arXiv:2309.17249 |
| L0 | content-free prior (optional) | Zhao et al. 2021, arXiv:2102.09690 |
| L1 | temperature scaling | Guo et al. 2017, arXiv:1706.04599 |
| L2 | ridge and LDA probes on intermediate hidden states | Skean et al. 2025, arXiv:2502.02013, among others |
| L2 | Ledoit–Wolf covariance shrinkage | Ledoit & Wolf 2004 |
| L2 | smoothed targets for two-class temperature fits | Platt 1999 |

AnyJev builds on the same literature, so the two libraries agree on much of the maths.

## Where brier differs from AnyJev

Taken from AnyJev's public repository when it was read; their project may have changed since.

| | AnyJev | brier |
|---|---|---|
| L2 head family | caller picks one (diff-means, LDA, ridge, reduced-rank) | ridge and LDA, chosen by out-of-fold NLL |
| L2 layer | fixed by the caller | selected over a layer grid |
| LDA shrinkage | fixed values | Ledoit–Wolf, estimated from the data |
| L2 folds | random k-fold | stratified k-fold |
| L2 temperature | plain NLL | Platt targets for two classes; plain NLL beyond, with a Platt fallback for separable scores |
| Model loading | allows `trust_remote_code` | never runs model code; safetensors only |
| Prompt | state inserted as is | state escaped and tokenised with special tokens disabled |
| Calibration files | — | JSON + `.npz`, validated as untrusted input |
| Evaluation | larger task suite; pooled ECE as the headline | one task so far; bootstrap CIs, paired comparisons, per-question ECE, 2,603 test items |
| Model families | Qwen | five families tested (docs/COMPATIBILITY.md) |

AnyJev has much that brier does not: shared-prefix KV caching, certified adaptive stopping of
rotations, a vLLM backend, model truncation for serving L2, online and label-free head
adaptation, pretrained heads and a broad benchmark suite. Those are AnyJev's contributions;
brier does not reproduce them. If it ever adds something comparable, the design will be its
own, recorded in an ADR, and credited here.

## Numbers

On banking20 with Qwen3-1.7B and 300 labels, AnyJev's committed L2 result (200 test items) is
accuracy 0.785, ECE 0.045, NLL 0.733; brier's (2,603 test items, docs/results.md) is accuracy
0.807, ECE 0.035, NLL 0.727. The splits, prompts and test sizes differ, and 200 test items give
about ±0.06 on accuracy, so this shows the two are **on par**; it does not show either is
better. A fair comparison needs AnyJev run on brier's exact split, which is planned
(docs/EVALUATION.md).
