# Security policy

## Supported versions
Only the latest release receives security fixes while the project is pre-1.0.

## Reporting a vulnerability
Please **do not** open a public issue. Use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability"). You should get an acknowledgement within
7 days. Please include a minimal reproduction and the affected version.

## Scope
In scope: code execution or data leakage through model loading, calibration artifacts,
the benchmark CLI, or dependencies we pin. See `docs/THREAT_MODEL.md`.

Out of scope: decisions being manipulable by adversarial text in the state (prompt
injection). This is a documented limitation, not a vulnerability. Don't use a decision
as the sole gate for destructive or irreversible actions.
