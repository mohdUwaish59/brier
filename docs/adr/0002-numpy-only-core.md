# ADR-0002: numpy-only core, model code behind a Backend protocol

Status: accepted

## Context
Calibration maths must be testable fast on CPU and independent of the inference engine.

## Decision
The core depends only on numpy. torch/transformers live in the `hf` extra and are
imported lazily inside `backends/hf.py`. All model access goes through the three
`Backend` methods (ARCHITECTURE.md).

## Consequences
Unit tests run without a GPU or model download. New engines (vLLM, SGLang) are new
backends only. Some operations (e.g. prefix KV reuse) will need new Backend methods, via a new ADR.
