# Methods (source of truth for all maths)

Notation: question with K options `o_0 … o_{K-1}`; `z` = log-probs of the K label
tokens gathered from the full-vocabulary log-softmax; `N` = number of items.
All calibration maths runs in float64. `norm(v)` = `v - logsumexp(v)` (log-space renormalisation).

## raw

`log p = norm(z)`, i.e. softmax restricted to the K label tokens.
Label tokens: `A`…`Z` for Choice, `Yes`/`No` for Noul, digits `1`…`L` for Score
(fallback to letters if any digit is not a single token). The backend must verify each
label is a single, distinct token *in the position it is read* (leading-space variants
matter). Otherwise it raises `TokenizationError`. Don't guess.

## L0 — order debiasing (no labels)

**Rotations (Choice only).** For shift `s ∈ S ⊆ {0,…,K-1}` display order is
position `j` shows option `(j + s) mod K`. Given position log-probs `z^(s)`,
option `k` sat at position `(k - s) mod K`, so

    ℓ^(s)_k = norm(z^(s))_{(k - s) mod K}

Combine by geometric mean (mean of log-probs) and renormalise:

    log p̄_k = norm( (1/|S|) Σ_s ℓ^(s)_k )

Default `S` = all K shifts (every option visits every position once).
Property to test: permuting the *input* option list permutes the output identically
(equivariance), and with all K shifts the result is invariant to the starting order.

**Batch prior correction (Zhou et al., ICLR 2024 — Batch Calibration).**
From an unlabelled pool of states for the same question:

    π_k = (1/N) Σ_i p̄_{i,k}          (clip π_k ≥ 1e-6, renormalise)
    log q_{i,k} = norm( log p̄_{i,k} − λ log π_k )

`λ ∈ [0,1]` (config, default chosen by evaluation, not hard-coded; start at 1.0 and
record the choice in an ADR). `π` is stored in the artifact after `fit_prior`.
Documented weakness: when the *true* label distribution is skewed, this removes real
signal. Report accuracy on skewed questions separately.

**Content-free prior (Zhao et al., ICML 2021), optional.** Estimate `π` from states
`"N/A"`, `""`, `"[MASK]"` instead of a pool. Useful when no unlabelled pool exists.

**Noul:** no rotation (two labels); prior correction only.
**Score:** never rotated (order is meaningful); prior correction optional, off by default.

## L1 — temperature scaling (Guo et al., ICML 2017)

    log q^T = norm( log q / T )

Fit `T` on labelled items by minimising mean NLL over `t = log T ∈ [-3, 3]` with a
bounded golden-section search (the NLL is unimodal in `t` for practical data; also
evaluate the endpoints and pick the best). One `T` per question. `T` never changes the argmax.
Minimum labelled items: 50 (else `InsufficientDataError`).

## L2 — hidden-state head

Features: residual-stream vector `h ∈ R^d` at the last prompt token after block `b`.
Standardise with training mean/std (stored), `x = (h − μ)/σ` (σ floored at 1e-6).

**Dual ridge (n < d, the usual case).** One-hot targets `Y ∈ R^{n×C}` centred per column:

    A = (X Xᵀ + α I_n)^{-1} (Y − Ȳ)       W = Xᵀ A       scores s = x W + Ȳ
Use the primal form `W = (XᵀX + αI_d)^{-1}Xᵀ(Y−Ȳ)` when n ≥ d. Solve with
`np.linalg.solve` / Cholesky, never an explicit inverse.

**Shrinkage LDA.** Class means `m_c`, pooled within-class covariance `Σ`,
shrunk `Σ_γ = (1−γ)Σ + γ (tr Σ / d) I` with Ledoit–Wolf γ. When n < d, compute
`Σ_γ^{-1} m_c` via the Woodbury identity. Scores:
`s_c = xᵀΣ_γ^{-1}m_c − ½ m_cᵀΣ_γ^{-1}m_c + log π_c`.

*Implementation notes (M5.1):* `σ` is the population standard deviation (ddof 0). The pooled
within-class covariance is `Σ = ZᵀZ / n` with residuals `Z = X − M[y]` (the 1/n estimate the
Ledoit–Wolf formula is defined on). Ledoit–Wolf: `μ = tr Σ / d`, `δ² = ‖Σ − μI‖²_F`,
`β̄² = (1/n²) Σ_k ‖z_k z_kᵀ − Σ‖²_F`, `γ = min(β̄², δ²) / δ²`, all computed from the n×n Gram
matrix `Z Zᵀ`. `π_c` are training class frequencies; every class must appear. With n < d and
`γ ≈ 0` the shrunk covariance is singular and fitting raises.

**Probabilities.** `log p = norm(s / T)` with `T` fitted on out-of-fold scores.

**Selection.** Stratified k-fold (k=5, seeded) over the grid
`layers × α ∈ {1e-2,…,1e4} (log grid) × solver ∈ {ridge, lda}`; choose by
out-of-fold NLL, then refit on all labels. Store the choice and OOF metrics in the artifact.
Default layer grid: every 2nd block from 40 % to 90 % depth.
Minimum: 5 labels per class and 60 total (else `InsufficientDataError`).

*Implementation notes (M5.2):* folds deal each class's shuffled items round-robin; `k` must not
exceed the smallest class count, so every fold holds every class. α grid: `{1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4}` for ridge; LDA has no α
and is tried once per layer. A candidate's OOF NLL is measured **after** fitting its own
temperature on its OOF scores, so ridge (scores are not log-probabilities) and LDA compete
fairly. L2 temperatures are searched over `t = log T ∈ [−7, 7]` (L1 keeps `[−3, 3]`):
ridge scores differ by ~0.1–1 and need T ≪ e^-3, LDA scores can differ by hundreds. If T
lands on a bound (OOF scores nearly separable) the selection records `temperature_at_bound`.
The reported OOF NLL/accuracy are the minimum over the grid, so they are optimistic
(winner's curse): a diagnostic, not a held-out estimate. A
candidate whose head cannot be fitted (e.g. degenerate LDA on one layer) scores `inf`.

## References

- Zhao et al. 2021, *Calibrate Before Use*, arXiv:2102.09690
- Zheng et al. 2024, *LLMs Are Not Robust Multiple Choice Selectors*, arXiv:2309.03882
- Zhou et al. 2024, *Batch Calibration*, arXiv:2309.17249
- Guo et al. 2017, *On Calibration of Modern Neural Networks*, arXiv:1706.04599
- Ledoit & Wolf 2004, *A well-conditioned estimator for large-dimensional covariance matrices*
- Skean et al. 2025, *Layer by Layer*, arXiv:2502.02013 (intermediate layers)
