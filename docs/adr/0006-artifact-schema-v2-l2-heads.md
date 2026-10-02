# ADR-0006: Artifact schema version 2 stores L2 heads

Status: accepted (schema version 3, adding the model dtype, is in ADR-0007)

## Context
L2 (`heads/select.py`) produces a fitted head per question: a layer, a solver (ridge or LDA),
an alpha (ridge only), a standardiser, a linear map to class scores, an L2 temperature and
out-of-fold diagnostics. `Decider.save/load` must round-trip it so L2 decisions are identical
after loading, like L0/L1 (ADR-0003, M4.3). Schema 1 has no place for it, and ADR-0003 says
adding fields bumps `schema_version`. Heads are also much larger than priors: `d × C` floats,
about 330 KB for `d = 2048, C = 20`, so the 4 KB per-array cap used for priors cannot apply.

## Decision
1. **`schema_version` 2.** `save` always writes 2. `load` accepts 1 and 2; a version-1 file is
   read as version 2 with no heads. Any other version is rejected (unchanged).
2. **One optional `head` object per question entry** in `artifact.json` (`null` when absent):
   `layer` (int, ≥ 0), `solver` (`"ridge"` or `"lda"`), `alpha` (number for ridge, `null` for
   LDA), `temperature` (the L2 T, separate from the L1 `temperature`), `temperature_at_bound`
   (bool), `oof_nll` and `oof_accuracy` (numbers, diagnostics only). The candidate grid is not
   stored.
3. **Both head types are stored in one affine form** — `scores = ((h − mean) / scale) @ weights
   + bias` — as four npz arrays per question `i`: `head_<i>_mean` `(d,)`, `head_<i>_scale`
   `(d,)`, `head_<i>_weights` `(d, C)`, `head_<i>_bias` `(C,)`. Ridge maps to (W, Ȳ), LDA to
   (Σ_γ⁻¹Mᵀ, intercept). Array names are derived from the question index, never read from JSON.
4. **Validation (THREAT_MODEL T2), same hardened path as priors:** `.npy` headers parsed by
   hand before any allocation; dtype `<f8`, C order; `d` must agree across the four arrays and
   be ≤ 65,536; `C` must equal the question's answer count; every value finite; `scale > 0`.
   The 4 KB per-member cap stays for priors; head members are capped by their validated shape
   (at most `65,536 × 26 × 8` bytes plus header) and the whole artifact by `max_bytes`
   (default 100 MB, unchanged). The temperature must lie in L2's range `[e^-7, e^7]`.
5. **Model binding:** the existing model id, revision and template-hash checks apply (hidden
   states depend on the rendered prompt). `Decider.load` additionally rejects a head whose
   `layer` is not below `backend.num_layers`. A hidden-size mismatch surfaces as `BrierError`
   on the first L2 decision.

## Consequences
- Version-1 artifacts keep loading. Artifacts written by this version are not readable by
  older brier versions, which fail with a clear "unsupported schema_version" error.
- Artifacts with L2 heads are larger (hundreds of KB per question) but stay far below the cap.
- Loaded heads are reconstructed as immutable read-only arrays; L2 decisions after a round
  trip are identical (tested like M4.3).

## Implementation notes (security audit)
- At most 1,024 questions per artifact (bounds parsing work; each head is four npz members).
- Head `scale` must be ≥ 1e-6, the floor fitting uses; `.npy` shape dimensions must be exact
  ints; non-finite or oversized numbers in head JSON raise `BrierError` → `ArtifactError`.
