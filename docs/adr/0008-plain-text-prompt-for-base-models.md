# ADR-0008: Plain-text prompt for models without a chat template

Status: accepted

## Context
`HFBackend` wraps every user message in the model's chat template (ADR-0004) and raises
`BrierError` when the tokenizer has none. Base (pre-trained, not instruction-tuned) models
usually ship no chat template — SmolLM2-360M, OLMo-2-1B, Falcon3-1B-Base, Granite-MoE base,
Pythia — so brier cannot run on them at all, although reading a next-token distribution after
`Answer:` is exactly the kind of completion a base model is trained for.

## Decision
1. **Format chosen by the tokenizer, not by the caller.** If the tokenizer has a chat template,
   the chat path is used, unchanged. If it has none, the **plain** format is used:

       {bos?}{SYSTEM}\n\n{user message}\n{ANSWER_PREFIX}

   `SYSTEM`, the user message from `prompts.render` and `ANSWER_PREFIX` are the same strings as
   on the chat path, so `prompts.template_hash()` is unchanged. `{bos?}` is whatever special
   tokens the tokenizer itself puts in front of a text when asked to add special tokens (none
   for most current tokenizers; `<s>` for Llama-2-style ones).
2. **No override.** Because the format follows from the tokenizer, it is fixed by the model id
   and revision, which artifacts already bind (ADR-0003). An option to force a format would let
   one model id produce two different inputs, and artifacts would then have to record the
   format too (a schema change). Add it only with that schema change.
3. `HFBackend.prompt_format` is `"chat"` or `"plain"`; `brier check` reports it.
4. The same protections apply on both paths: the user message is tokenised with special-token
   parsing disabled (THREAT_MODEL T3), and the full prompt is checked against
   `max_prompt_tokens`.

## Consequences
- Any causal LM with safetensors weights and the standard decoder layout can be used,
  including base models. Base models are often weaker zero-shot readers; `brier check`'s sanity
  task and the L1/L2 levels show and correct how much.
- Models whose layers are not at `model.get_decoder().layers` (e.g. GPT-2) remain
  unsupported; that is a separate problem (layer discovery).
- A tokenizer that gains or loses a chat template in a new revision changes format with it;
  artifacts are bound to the revision, so a pinned revision is unaffected.
