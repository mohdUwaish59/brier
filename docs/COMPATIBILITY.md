# Model compatibility

`HFBackend` works with any Hugging Face causal LM that has a **chat template** and
**safetensors** weights (`trust_remote_code` is never enabled).

**Check your own model first:**

```bash
brier check <model-id> --revision <commit-sha>
```

It reports whether the labels, batching and hidden states work, which levels the model
supports, and a quick sanity score (SPEC §7). The integration suite runs on every family below
with pinned revisions:

```bash
BRIER_TEST_MODELS=all uv run pytest -m integration      # or e.g. qwen3,gemma3
```

| Key | Model (pinned revision) | Template | Label tokens after `Answer:` | Score labels | Notes |
|---|---|---|---|---|---|
| `qwen3` | `Qwen/Qwen3-0.6B` @ `c1899de` | ChatML + empty `<think>` block | `ĠA`, `ĠYes` (byte-level BPE) | letters | Thinking disabled with `enable_thinking=False` |
| `smollm2` | `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc15` | ChatML | `ĠA`, `ĠYes` | letters | Weak answers (model size), works |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea` | Zephyr (`<\|user\|>`, `</s>`) | `▁A`, `▁Yes` (SentencePiece) | letters | Llama-2 tokenizer; weak answers |
| `olmo2` | `allenai/OLMo-2-0425-1B-Instruct` @ `48d788e` | Tülu (`<\|user\|>`, `<\|assistant\|>`) | `ĠA`, `ĠYes` | letters | Weak answers |
| `gemma3` | `google/gemma-3-1b-it` @ `dcc83ea` | `<start_of_turn>`; system text merged into the user turn | `▁A`, `▁Yes` (SentencePiece) | letters | Use `attn_implementation="eager"` (see below) |
| `granitemoe` | `ibm-granite/granite-3.1-1b-a400m-instruct` @ `0da7a48` | Granite (`<\|start_of_role\|>`) | `ĠA`, `ĠYes` | letters | **Mixture of experts**; weak answers (`brier check`: L0 64 % on the sanity set) |
| `lfm2` | `LiquidAI/LFM2-1.2B` @ `40f3da0` | ChatML | `ĠA`, `ĠYes` | letters | **Hybrid convolution + attention**; sanity L0 71 % |
| `r1distill` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae` | DeepSeek (`<｜User｜>`); always opens `<think>` | `ĠA`, `ĠYes` | letters | **Reasoning model: thinking cannot be disabled** (see below); sanity L0 57 % |
| `falcon3` | `tiiuae/Falcon3-1B-Instruct` @ `28ba225` | `<\|system\|>` / `<\|user\|>` | `ĠA`, `ĠYes` | letters | Llama architecture; sanity L0 64 % |
| `phi4mini` | `microsoft/Phi-4-mini-instruct` @ `cfbefac` | Phi (`<\|end\|>`) | `ĠA`, `ĠYes` | letters | 3.8B; sanity raw 100 % / L0 93 % |
| `smollm3` | `HuggingFaceTB/SmolLM3-3B` @ `a07cc9a` | ChatML + metadata system block; empty `<think>` block | `ĠA`, `ĠYes` | letters | Thinking disabled with `enable_thinking=False`; **template inserts today's date** (see below); sanity 93 % |
| `olmoe` | `allenai/OLMoE-1B-7B-0125-Instruct` @ `b89a7c4` | Tülu-style (`<\|user\|>`) | `ĠA`, `ĠYes` | **digits** (` 10` is one token) | **Mixture of experts** (7B total); sanity 93 % |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c70` | `[INST]`; system text merged into the user turn | `▁A`, `▁Yes` (SentencePiece) | letters | 7B; sanity 100 % |

All pass: template split, special-token injection blocked, batched = unbatched within
1e-4 (fp32), padding-invariant, strict label tokens, hidden states equal a full forward pass,
early stop after the deepest layer, size limits, and `Decider` raw/L0 end to end.

The first five families and `granitemoe` were run on a CPU; `lfm2` to `mistral7b` on an
A100 in fp32 (`notebooks/m6_3_compatibility_matrix.ipynb`). `brier check` passed on all of
them. On the GPU one test failed for every model because the test built its reference input
on the CPU, and `olmoe` failed two tests that assumed ` 10` is never a single token; both
were test bugs, fixed in `51ae987`.

**Weekly CI** (`.github/workflows/integration.yml`) runs `brier check` and this suite on
`qwen3`, `smollm2` and `granitemoe` with the newest allowed torch / transformers / numpy.

## Findings

- **Score usually uses lettered options.** Most tokenizers split digits off the space after
  `"Answer:"`, so METHODS.md's letter fallback is the normal path. OLMoE is the exception:
  ` 1` … ` 10` are single tokens there, so its Score questions use digits.
- **Reasoning models whose template always thinks.** DeepSeek-R1-Distill's template opens a
  `<think>` block after the user turn and ignores `enable_thinking=False`, so brier reads the
  answer inside an open thinking block. It works (all checks pass) but the readout is off the
  model's normal answer path, which likely costs accuracy (sanity L0 57 %). Closing the block
  before `Answer:` would change the prompt template (ADR-0004); it is on the roadmap.
- **Templates that insert the date.** SmolLM3's template writes today's date into the system
  prompt, so the exact prompt, and therefore the probabilities, change slightly from day to
  day. A calibration fitted on one day is applied to slightly different prompts later; the
  effect should be small, but re-check calibration if it matters.
- **Gemma 3 and attention kernels.** With the default SDPA kernel, batched and unbatched
  log-probs differ by up to 1.9e-4 (fp32); with `attn_implementation="eager"` by 5e-5.
  Numerical, not a padding bug; pass `attn_implementation="eager"` for Gemma 3.
  Gemma models may require accepting Google's licence on Hugging Face (and a logged-in
  token) before download; the default test model (`qwen3`) is ungated.
- **Gated models.** `meta-llama/Llama-3.2-*` need the licence accepted on Hugging Face
  and a logged-in token; not yet in the matrix.
- **Answer quality varies by model size.** Small models (SmolLM2-360M, TinyLlama, OLMo-2-1B)
  route the quickstart example wrongly. That is model quality, not an adapter failure; the
  sanity-answer tests run only for models marked capable.

## Not supported (yet)

- **Base models without a chat template** raise `BrierError`. A plain-text fallback would be
  a new prompt template and needs an ADR (ADR-0001).
- Models that need `trust_remote_code=True` or ship only `.bin`/pickle weights.
