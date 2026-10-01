# ADR-0003: Calibration artifacts are JSON + npz, never pickle

Status: accepted

## Decision
An artifact is a directory with `artifact.json` (schema_version, declib version,
model id + revision, prompt template hash, per-question config: prior, temperature,
head metadata, SHA-256 of arrays file) and `arrays.npz` (numeric arrays only), loaded
with `allow_pickle=False`. Loading fails with `ArtifactError` on schema, hash, model or
template mismatch.

## Consequences
Artifacts are safe to share and inspect. Adding fields requires bumping `schema_version`
and a migration or a clear error.
