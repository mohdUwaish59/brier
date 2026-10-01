# ADR-0003: Calibration artifacts are JSON + npz, never pickle

Status: accepted

## Decision
An artifact is a directory with `artifact.json` (schema_version, brier version,
model id + revision, prompt template hash, per-question config: prior, temperature,
head metadata, SHA-256 of arrays file) and `arrays.npz` (numeric arrays only), loaded
with `allow_pickle=False`. Loading fails with `ArtifactError` on schema, hash, model or
template mismatch.

## Consequences
Artifacts are safe to share and inspect. Adding fields requires bumping `schema_version`
and a migration or a clear error.

## Implementation notes (M4.2, schema_version 1)
- `artifact.json` keys: `schema_version`, `brier_version`, `model {id, revision}`,
  `template_hash` (`prompts.template_hash()`), `prior_strength` (λ used to apply priors),
  `arrays_sha256`, `questions` (full question + `prior: bool` + `temperature`).
  Prior arrays live in `arrays.npz` as `prior_<i>`, `i` = question index; no file or array
  name is ever read from the JSON.
- L2 head metadata is not in schema 1; adding it bumps `schema_version`.
- Loading also: opens each file once and requires a regular non-symlink file read with a size
  cap; rejects unknown keys; requires exactly the referenced npz members (no duplicates), each
  stored/deflated, unencrypted and ≤ 4 KB, with total uncompressed size under the cap; parses
  each `.npy` header itself and requires `<f8`, C order and shape `(K,)` before reading any
  data (never `np.load`); requires priors strictly positive and summing to 1, and temperatures
  in `[1e-6, 1e6]`; converts any unexpected exception to `ArtifactError`.
