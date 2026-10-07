# ADR-0010: Shared-prefix KV caching for rotations and multiple questions

Status: accepted

## Context
Every brier prompt starts with the same text for a given state: the chat template (or the
plain-text system line), then `<state> … </state>`; only the question, the option list and the
answer prefix differ (ARCHITECTURE.md puts the state first for exactly this reason). Yet the
backend recomputes the whole prompt every time:

- L0 asks one Choice question K times, once per rotation of its options;
- `decide` with several questions asks about the same state once per question (K times each
  at L0).

With state length S and question-plus-options length Q, an L0 decision costs about `K (S + Q)`
tokens of forward pass; with the state's key/value cache reused it costs about `S + K Q`. For
long states (tickets, agent traces, documents) that is several times cheaper; for short states
such as banking20's (about 20 tokens of state against 100+ tokens of options) it saves little.
M7.2 (ADR-0009) reduces K; this ADR reduces the cost of each rotation and question. AnyJev
implements shared-prefix caching too (credited in `docs/related_work.md`); this design is
brier's own.

## Decision
1. **Exactness is defined on token ids, not strings.** Within one backend call, prompts are
   grouped by the longest common prefix of the token ids `HFBackend.encode` already produces.
   No prompt string, template, label or `prompts.template_hash()` changes, so calibration
   artifacts are unaffected. Tokenisation stays as today (special-token text in the state is
   split, THREAT_MODEL T3), and `max_prompt_tokens` applies to each full prompt.
2. **Phase A, no interface change:** `brier.debias.l0_logprobs` sends all `N × K` rotated
   prompts of a question in **one** `label_logprobs` call instead of K calls (the label tokens
   are the same for every rotation). The contract of `label_logprobs` is unchanged, so every
   backend keeps working; a backend that shares prefixes inside a call gets the saving.
3. **Phase B, an optional backend method** for several questions per state:
   `label_logprobs_multi(prompts, token_ids_per_prompt) -> list[array]`, one label set per
   prompt (questions differ in their labels). `Decider.decide_batch` uses it when the backend
   has it and falls back to one call per question otherwise. It is not part of the `Backend`
   protocol; `FakeBackend` implements it so the fallback and the fast path are both tested.
4. **`HFBackend(prefix_cache=False)`, opt-in at first.** When on: for each group of at least
   two prompts whose shared token prefix is at least 32 tokens, run the prefix once, reuse its
   cache for the group's tails (right-padded, the label log-probs read at each tail's last
   real token), and use the current path for everything else. Only models whose cache can be
   reused exactly are eligible (standard full-attention caches); models with sliding-window
   attention (Gemma 3) or non-attention state (LFM2's convolutions) fall back to the current
   path until an integration test proves them exact.
5. **Correctness bar:** cached and uncached results must agree to `1e-4` in log-probability in
   float32 on every family in the integration suite, and within the existing batch-consistency
   tolerance (`0.02` in probability) in bfloat16. `brier check` reports a `prefix_cache` line:
   exact (with the measured difference and speed-up), fallback, or off.
6. **The default changes only with evidence:** `prefix_cache` becomes `"auto"` (on where
   eligible and worthwhile) only after a long-state benchmark shows a real speed-up with exact
   results; that switch is recorded as an amendment to this ADR.

## Consequences
- L0 and multi-question decisions get cheaper on long states; banking20 numbers do not change
  (the results must be identical, which the benchmark can check).
- `HFBackend` becomes more complex (grouping, cache reuse, right-padded tails); the grouping
  logic is a pure function with its own unit tests, and the cache path is covered by
  cached-vs-uncached integration tests per family.
- Peak memory per group grows with the group size; batches are capped by `batch_size` as
  today.
- Backends other than `HFBackend` need no change (phase A) and may implement phase B later.
