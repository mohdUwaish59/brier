# ADR-0004: Prompt templates for Noul and Score, and state escaping

Status: accepted

## Context
ARCHITECTURE.md fixed the Choice template only. Noul and Score need templates, and
"escape `<state>` delimiters" did not say how. The `Backend` protocol takes plain strings.

## Decision
- `prompts.render` returns the **user message**. The backend wraps it in the chat
  template with `prompts.SYSTEM` as system message and `prompts.ANSWER_PREFIX`
  (`"Answer:"`) as the start of the assistant turn. The Backend protocol is unchanged.
- Noul mirrors Choice: `Question: …` then `Answer Yes or No.`; labels `Yes`/`No`.
- Score: `Question: …`, then either `1. {label}` … `L. {label}` or
  `Scale: 1 (lowest) to L (highest).`, then `Answer with the number only.`; labels `1`..`L`.
  When a digit is not a single token (always for L = 10), Score is rendered as lettered
  options (`A. {label or level}` …, `Answer with the letter only.`) and read on `A`…
- The state is escaped with `html.escape(state, quote=False)` (`&`, `<`, `>`), so it
  cannot open or close a `<state>` block in any case or spacing variant. Question and
  option text are caller-controlled and are not escaped.

## Consequences
Exact strings are snapshot-tested; changing them is a breaking change for artifacts
(which store a template hash, M4.2) and needs a new ADR. Escaping is not a security
boundary against prompt injection (THREAT_MODEL T3).
