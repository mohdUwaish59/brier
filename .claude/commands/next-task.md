---
description: Implement the next roadmap task end to end (test-first)
argument-hint: [optional task id, e.g. M1.3]
---
Implement roadmap task $ARGUMENTS (if empty: the first unchecked task in docs/ROADMAP.md).

1. Read CLAUDE.md, the task, and the SPEC/METHODS/ARCHITECTURE sections it touches. Restate the acceptance criteria.
2. Create a branch `task/<id>-<short-name>`.
3. Use the test-writer subagent to write failing tests first.
4. Implement the smallest code that passes them, following the code standards.
5. Run `/check`. Fix until everything is green.
6. If maths changed, run the math-verifier subagent. If backends/artifacts/prompts/deps/CI changed, run the security-auditor subagent. Then run the code-reviewer subagent. Address findings.
7. Update docs, CHANGELOG (Unreleased) and tick the task in ROADMAP.md.
8. Commit with a Conventional Commit message. Do not push. Summarise what changed and anything the human must decide.
