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

*Subset rotations (M7.2, benchmark only):* `m` evenly spaced shifts
`S_m = {floor(j K / m) : j = 0, …, m − 1}` cost `m` forward passes instead of `K`. The
combine is unchanged (geometric mean over `S_m`), and the batch prior is fitted on the pool
with the same `S_m`. With `m < K` an option no longer visits every position, so position bias
cancels only partly; `python -m brier.bench rotations` measures how much of the gain remains.
`Decider(rotations=m)` uses the same subsets (ADR-0009); the default is all K. Results on
banking20 are in `docs/results.md` (M7.2).
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

**Probabilities.** `log p = norm(s / T)` with `T` fitted on out-of-fold scores. With two
classes `T` is always fitted against Platt (1999) targets: small, near-separable binary sets
otherwise give near-0/1 probabilities (on a real 60-label refund/tracking task, plain NLL put
98.9 % "refund" on a tracking request). With more classes `T` is fitted by plain NLL
(temperature scaling, Guo 2017), and only if that fit lands on its lower bound — separable OOF
scores, where plain NLL would drive `T → 0` — is it refitted against Platt targets taken
one-vs-rest: `(n_c + 1)/(n_c + 2)` for an item's class `c`,
the remaining `1/(n_c + 2)` spread evenly over the other `C − 1` classes (for `C = 2` exactly
Platt's `(N₊+1)/(N₊+2)` and `1/(N₋+2)`; only Platt's targets are used, the model stays a
single temperature with no bias term). Candidates are compared by plain OOF NLL.

*Correction (M5.4b):* earlier versions smoothed every fit. `(n_c + 1)/(n_c + C)` capped the
target at 16/35 ≈ 0.46 with C = 20 and 15 labels per class (banking20 L2: ECE 0.46 at 81 %
accuracy); one-vs-rest targets on every fit still biased `T` upward with many classes (in
simulation, C = 20 with 5 labels per class: ECE 0.17 vs 0.02 for plain NLL). Smoothing is
therefore always used for two classes and, beyond two, only when plain NLL hits the bound.
Known residual risk: small, easy tasks with 3–5 classes and very few labels can still come out
somewhat overconfident.

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
lands on a bound the selection records `temperature_at_bound` (lower bound: score gaps too
small to sharpen enough, and `fit_head` warns; upper bound: no signal, near-uniform output).
The reported OOF NLL/accuracy are the minimum over the grid, so they are optimistic
(winner's curse): a diagnostic, not a held-out estimate. A
candidate whose head cannot be fitted (e.g. degenerate LDA on one layer) scores `inf`.

## References

- Zhao et al. 2021, *Calibrate Before Use*, arXiv:2102.09690
- Zheng et al. 2024, *LLMs Are Not Robust Multiple Choice Selectors*, arXiv:2309.03882
- Zhou et al. 2024, *Batch Calibration*, arXiv:2309.17249
- Guo et al. 2017, *On Calibration of Modern Neural Networks*, arXiv:1706.04599
- Platt 1999, *Probabilistic Outputs for Support Vector Machines* (smoothed targets)
- Ledoit & Wolf 2004, *A well-conditioned estimator for large-dimensional covariance matrices*
- Skean et al. 2025, *Layer by Layer*, arXiv:2502.02013 (intermediate layers)
