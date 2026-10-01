# ADR-0005: L0 works without a fitted prior; prior strength λ defaults to 1.0

Status: accepted

## Context
METHODS.md defines L0 as option rotations plus batch prior correction, and asks for λ's
default to be recorded in an ADR. SPEC.md says an unfitted level raises `NotFittedError`,
but the quickstart calls `level="L0"` without `fit_prior`.

## Decision
- L0 always applies option rotations (Choice). Prior correction is applied only when a
  prior has been fitted for that question (`fit_prior`); otherwise L0 is rotation-only
  and does not raise. `Decision.meta` records whether a prior was applied (M3.1).
- λ defaults to 1.0 and is configurable in `[0, 1]`. Revisit after the M3.4 benchmark.
- Score: prior correction stays off by default (METHODS.md).

## Consequences
The quickstart works as written. Callers who want the full L0 must call `fit_prior`
and can check `meta`. Batch Calibration only partly cancels an additive logit bias
(it divides by the mean probability), so a residual label bias can remain.
