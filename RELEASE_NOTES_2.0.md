# 2.0 release notes

Release 2.0 is the reviewed dual-DGX-Spark deployment for the DeepSeek V4 Flash abliterated vision checkpoint. It preserves the pinned Anemll `0.1.1` runtime ABI while fixing DSpark integration, prefix-cache behavior, multimodal validation, and distributed deployment identity.

## Highlights

### Correct DSpark head sharing for multimodal targets

The pinned runtime assumed the target model exposed `lm_head` directly. The custom multimodal wrapper keeps the language head on its unwrapped language model. Release 2.0 applies a fail-closed patch that:

- resolves the head from the unwrapped language model with a compatible outer-wrapper fallback;
- verifies vocabulary-compatible head sharing;
- propagates the target attention backend;
- selects the draft quantization configuration explicitly;
- refuses unknown source bytes instead of patching a drifting runtime.

### Prefix-cache replay fix

When prefix caching skips cached prompt tokens, the DSpark sliding-attention state still needs the final draft window from the target model. Release 2.0 backports the relevant cache-manager and scheduler behavior and verifies the exact before/after source hashes.

### Fail-closed vision layout

Release 2.0 treats checkpoint tile metadata as authoritative. A supplied override must match it; missing metadata requires an explicit verified legacy value. Incomplete or incompatible vision-tower state stops startup rather than leaving randomly initialized or semantically inconsistent vision modules active.

### Two-node artifact identity

The cluster launcher resolves the image selected by each node's own environment file before launch. It refuses to start either TP rank unless both tags resolve to the same content-addressed image ID. An optional expected image ID provides an additional deployment pin.

### Honest DSpark measurements

The acceptance collector aggregates all matching Prometheus series and rejects:

- counter resets;
- concurrent traffic contamination;
- non-finite or inconsistent counters;
- insufficient draft samples;
- malformed per-position acceptance shapes.

The default 20% floor is a catastrophic-regression smoke threshold, not a performance target.

## Public release boundary

The repository publishes source provenance, deterministic build inputs, and validation procedures. It intentionally excludes machine-specific image IDs, hostnames, addresses, paths, logs, and deployment measurements. Every rebuilt image must be identified by its own content-addressed ID and requalified on the target environment.

## Build and validation

Run the deterministic local gate:

```bash
bash scripts/release-check.sh
```

Then build the overlay from the digest-pinned base resolved through `SOURCE_PINS.json`:

```bash
export DSPARK_VLLM_BASE_IMAGE="$(python3 scripts/resolve_dgx_spark_base.py)"
docker build \
  --build-arg DSPARK_VLLM_BASE_IMAGE="$DSPARK_VLLM_BASE_IMAGE" \
  -t anemll-dsv4-vision:0.1.1-dspark-headfix1 .
```

A rebuild is a new artifact and normally has a different image ID. Synchronize it across ranks, pin the resulting ID if desired, and repeat every live qualification gate before calling that rebuilt deployment verified healthy.

## Not included

Release 2.0 does not include the optional expanded DSpark draft-window override (`VLLM_DSPARK_SLIDING_WINDOW`).
