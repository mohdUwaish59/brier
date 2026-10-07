# ADR-0007: Artifact schema version 3 records the model's precision

Status: accepted (schema version 4, adding L0 rotations, is in ADR-0009)

## Context
Calibration is a property of the exact numbers a model produces. The same weights run in
`bfloat16` and in `float32` give different logits and hidden states: small differences for
raw/L0/L1, but an L2 head fitted on one precision's hidden states can be miscalibrated on
another. Today an artifact is bound to the model id, revision and prompt template
(ADR-0003, ADR-0006), but not to the precision, so a calibration fitted on a GPU in bf16 loads
silently on a CPU in fp32. ADR-0003 says adding fields bumps `schema_version`.

## Decision
1. **`schema_version` 3.** `save` always writes 3. `load` accepts 1, 2 and 3.
2. **`model.dtype`** joins `model.id` and `model.revision` in `artifact.json`: the backend's
   precision string (`HFBackend.dtype`: `"float32"`, `"bfloat16"` or `"float16"`) or `null`
   for a backend that does not expose one. Validation: `null` or a string of 1–32 characters
   from `[a-z0-9_.+-]` (room for later quantisation tags such as `"bfloat16+int4"`, which
   would need their own ADR).
3. **Binding:** for a version-3 artifact, `load` requires the recorded `dtype` to equal the
   backend's `dtype` (`getattr(backend, "dtype", None)`), exactly like the revision check, and
   raises `ArtifactError` otherwise. Version-1 and version-2 artifacts carry no precision, so
   the check is skipped for them; they keep loading as before.
4. `dtype` is an optional attribute of a backend, not part of the `Backend` protocol: it must
   be a string or absent (`Decider` raises `BrierError` otherwise). Custom backends without it
   record `null` and must load with a backend that also has none.

## Consequences
- Calibrations can no longer be reused across precisions by accident; re-fitting per
  precision is the supported path. Device (CPU vs CUDA) is not recorded: at a fixed precision
  its numerical differences are far below calibration error.
- Artifacts written by this version are not readable by older brier versions, which fail with
  "unsupported schema_version".
- Like the revision check, the precision binding guards against accidents, not tampering: the
  JSON is not signed, so whoever edits it can write any `dtype` (or relabel it as version 2).
  Where an artifact comes from remains the user's responsibility (THREAT_MODEL T2).
