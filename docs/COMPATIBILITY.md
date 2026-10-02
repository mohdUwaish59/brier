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

All six pass: template split, special-token injection blocked, batched = unbatched within
1e-4 (fp32), padding-invariant, strict label tokens, hidden states equal a full forward pass,
early stop after the deepest layer, size limits, and `Decider` raw/L0 end to end.

**Pending a GPU run** (`notebooks/m6_3_compatibility_matrix.ipynb`; registered in
`tests/integration/test_hf_backend.py` with pinned revisions): `lfm2` (LFM2-1.2B, hybrid
convolution/attention), `r1distill` (DeepSeek-R1-Distill-Qwen-1.5B, reasoning template),
`falcon3` (Falcon3-1B), `phi4mini` (Phi-4-mini, 3.8B), `smollm3` (SmolLM3-3B, thinking mode),
`olmoe` (OLMoE-1B-7B, mixture of experts), `mistral7b` (Mistral-7B-v0.3).

**Weekly CI** (`.github/workflows/integration.yml`) runs `brier check` and this suite on
`qwen3`, `smollm2` and `granitemoe` with the newest allowed torch / transformers / numpy.

## Findings

- **Score always uses lettered options.** No tested tokenizer encodes `" 1"` after
  `"Answer:"` as one token (digits are split off the space), so METHODS.md's letter fallback
  is the normal path, not an edge case.
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
