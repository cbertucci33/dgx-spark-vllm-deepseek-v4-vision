# Changelog

All notable public releases are recorded here. The plugin package keeps the `0.1.1` ABI version because release 2.0 does not change the model/plugin registration contract.

## 2.0 — 2026-08-24

Release tag: `2.0`

### Added

- Fail-closed DSpark `lm_head` resolution and compatible head sharing for multimodal targets.
- Draft quantization and target-attention-backend propagation.
- Prefix-cache replay backport for DSpark sliding-attention state.
- Fail-closed vision tile-layout and tower-state validation.
- Two-node image-ID consistency checks before either TP rank starts.
- Deterministic DSpark acceptance collection with counter-reset, contamination, sample-size, and per-position validation.
- Explicit CUDA-graph capture sizing for the configured DSpark batch shape.
- Deterministic wheel/release checks and expanded deployment-contract tests.

### Changed

- Removed unsupported historical DSpark environment variables that did not control the pinned runtime.
- Hardened worker-first startup, bounded readiness checks, and image pinning.

### Fixed

- DSpark draft-head lookup for the custom multimodal wrapper.
- Prefix-cache hits no longer omit the final draft window needed to repopulate DSpark state.
- Vision layout now fails closed on missing or contradictory tile metadata instead of silently selecting an unsafe layout.

### Deliberately excluded

Release 2.0 does **not** contain the optional expanded `VLLM_DSPARK_SLIDING_WINDOW` override.

## 1.0 — 2026-08-21

Initial public DGX Spark vLLM deployment overlay for the DeepSeek V4 Flash abliterated vision model.
