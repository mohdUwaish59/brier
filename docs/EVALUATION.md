# Evaluation protocol

## Metrics (`brier.metrics`, all pure numpy, tested against hand-computed cases)

| Metric | Definition |
|---|---|
| Accuracy | mean `argmax p == y` |
| NLL | mean `−log p_y` (clip p ≥ 1e-12) |
| Brier | mean `Σ_k (p_k − 1[k=y])²` |
| ECE | top-label, **15 equal-mass bins**: `Σ_m (|B_m|/N) |acc(B_m) − conf(B_m)|`; also report equal-width for comparison |
| Flip rate | share of items whose argmax changes when the option list is reversed (mapped back) |
| Risk–coverage | sort by confidence; AURC; coverage at risk ≤ α. **Label it "in-sample"** when the threshold is chosen on the same items |
| Score questions | MAE of expected level, ranked probability score (RPS) |

Conventions (as implemented): labels are option/level indices `0..K-1`. Equal-mass ECE bins
split items sorted by confidence (stable for ties); equal-width bins are `(m/n, (m+1)/n]`;
empty bins are skipped. AURC is the mean selective risk over coverages `k/N`, items sorted by
descending confidence (stable for ties); coverage at risk uses the same order, so a cut
may split a group of tied confidences. RPS is normalised by `K − 1` so it lies in `[0, 1]`.

Every metric is reported with a 95 % percentile bootstrap CI (1,000 resamples, seeded).
Comparisons between levels use **paired** bootstrap on the same items.

## Datasets

| Task id | Source | Setup |
|---|---|---|
| `banking20` | BANKING77 (`PolyAI/banking77` on HF) | 20 most frequent intents from the train split. Fixed seeded splits: 500 unlabelled prior pool, 300 labelled calibration/train, ≥ 1,000 test. Intents shown as their label names with `_` → space |
| `banking20-noul` | same | Noul per item: "Is this about {intent}?" balanced yes/no |
| later | CLINC150, a Score task | added in later milestones |

Check and record each dataset's licence in `bench/tasks.py` before adding it.

## Models

Development: `Qwen/Qwen3-1.7B`. Headline: `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`.
Add a second model family before claiming generality. Always pin the HF `revision` (commit sha).

## Result file (JSON, one per run)

```json
{
  "schema_version": 1,
  "brier_version": "…", "git_commit": "…",
  "model": {"id": "…", "revision": "…", "dtype": "bfloat16"},
  "env": {"python": "…", "torch": "…", "transformers": "…", "device": "…"},
  "task": "banking20", "split_seed": 0, "n_test": 1000,
  "levels": {"raw": {"accuracy": [v, lo, hi], "ece": [...], "...": "..."}, "L0": {}, "L1": {}, "L2": {}},
  "timing": {"forward_passes_per_decision": {}, "wall_seconds": {}}
}
```

Per-item predictions are saved next to it as `.npz` so metrics can be recomputed on CPU.

## Sanity reference

Optionally run AnyJev (pinned version) on the same splits as a reference point.
Expect raw < L0 on flip rate and L1 < L0 on ECE. If your raw/L0 numbers are far from
AnyJev's on the same model and task, investigate the prompt and tokenization first.
