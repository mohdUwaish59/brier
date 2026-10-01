# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- Repository scaffold, specification, methods, roadmap and CI.
- Official Apache-2.0 licence text in `LICENSE`.
- Committed `uv.lock`; refreshed pre-commit hooks; CI audit no longer audits the unpublished project itself.
- `Choice`, `Noul`, `Score` question types with validation, the `Decision` result type and the `declib.errors` hierarchy.
- `declib._math`: stable float64 `logsumexp` and `norm` (log-softmax).
- `declib.prompts`: Choice/Noul/Score templates, HTML-escaped state, option rotation (ADR-0004).
### Security
- GitHub Actions and pre-commit hooks pinned to full commit SHAs.
