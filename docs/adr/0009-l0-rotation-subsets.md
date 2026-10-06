# ADR-0009: Optional L0 rotation subsets; artifact schema version 4

Status: accepted

## Context
L0 asks a K-option Choice question once per cyclic rotation of its options (K forward passes),
so position bias cancels exactly (METHODS.md, L0). M7.2 measured evenly spaced subsets on
banking20 (`docs/results.md`): on Qwen3-1.7B, 2 rotations already match full-L0 accuracy and
calibration at a tenth of the cost, while on the strongly position-biased Falcon3-1B-Base
accuracy keeps rising up to all 20. A user asked for the cheaper option after the 0.1.0 launch.
The L0 prior and the L1 temperature are fitted on L0 outputs, so they depend on the rotations
used.

## Decision
1. **`Decider(..., rotations=None)`.** `None` (the default) keeps all K rotations. An int `m`
   in `[1, 26]` uses `m` evenly spaced shifts `floor(j K / m)`, `j = 0 … m − 1`, for every
   Choice question at L0 and L1, in `fit_prior` and `fit_temperature` alike. A question with
   `K ≤ m` uses all K. `Decision.meta["n_forward"]` reports the passes actually used; Noul,
   Score, raw and L2 are unaffected. `evenly_spaced` moves to `brier.debias` (core).
2. **Artifacts record it: `schema_version` 4** adds a top-level `rotations` (`null` or an int
   in `[1, 26]`). `Decider.load` restores it, like `prior_strength`. Versions 1–3 load with
   `rotations = null` (all K), which is what they were fitted with.
3. **The default stays all K.** Fewer rotations are a speed/quality trade-off whose cost
   depends on the model; `brier check`'s order-flip rate and the `rotations` benchmark show it.

## Consequences
- Users can cut L0 cost K/m times; the docs say when that is safe (low raw flip rate) and when
  it is not (strongly biased models).
- Artifacts written by this version are not readable by older brier versions ("unsupported
  schema_version"). A calibration fitted with `m` rotations is always applied with `m`, because
  the value travels with it.
- This is a fixed, evenly spaced subset chosen by the caller, not adaptive stopping (AnyJev's
  feature, credited in `docs/related_work.md`).
