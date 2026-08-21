# Reproducible model composition

## Inputs

Use the exact repositories, revisions, filenames, sizes, and SHA-256 values recorded in `SOURCE_PINS.json`.

The three independent input trees are:

1. the complete abliterated language-model repository;
2. a vision-assets directory containing `tower/deepencoder_v2_tower.safetensors` and `adapter/merged-004800-5af0c5.pt`;
3. a runtime-artifacts directory containing exactly the pinned `wheels/dsv4_vision_vllm-0.1.1-py3-none-any.whl`. The publishable plugin source belongs in the separate GitHub repository, not in this fail-closed assembler input tree.

## Command

```bash
uv run --python 3.12 python scripts/assemble_hf_model.py \
  --source-model /path/to/DeepSeek-V4-Flash-0731-abliterated \
  --vision-assets /path/to/vision-assets \
  --runtime-artifacts /path/to/runtime-artifacts \
  --release-metadata model-release \
  --source-pins SOURCE_PINS.json \
  --destination /path/to/DeepSeek-V4-Flash-0731-abliterated-vision
```

The destination and its `.building` staging sibling must not already exist.

## Composition algorithm

1. Reject symlinks in every input tree and reject any overlap among inputs, destination, and staging paths.
2. Parse `model.safetensors.index.json`, require its SHA-256 to match `SOURCE_PINS.json`, require every referenced shard to exist, and verify the index plus every referenced shard against the pinned `LANGUAGE_WEIGHTS.sha256` manifest.
3. Verify the pinned vision files by exact byte size and SHA-256.
4. Verify the runtime wheel by SHA-256.
5. Copy the language repository without rewriting its model shards.
6. Set `architectures` to `DeepseekV4VisionForCausalLM` and add `vision_config` to `config.json`.
7. Copy the vision assets under `vision/`, write `vision_config.json` and `preprocessor_config.json`, and copy the runtime artifact under `runtime/`.
8. Copy the eight allowed release-metadata files (`LICENSE`, `README.md`, `ASSEMBLY.md`, `THIRD_PARTY_NOTICES.md`, `VALIDATION.md`, `LICENSES/Apache-2.0.txt`, `LICENSES/MIT-DeepSeek.txt`, and `LICENSES/MIT-FlyCockpit.txt`) over the inherited language-only documentation.
9. Require the post-copy language weight index to equal the pre-copy index exactly.
10. Reject output symlinks, validate the copied shard graph again, write `MANIFEST.sha256`, and atomically rename staging to destination.

## What this does not do

- It does not average, interpolate, concatenate, or otherwise blend language-model tensors.
- It does not re-quantize the language model.
- It does not add vision tensors to `model.safetensors.index.json`; the custom runtime loads the tower and projector from their explicit paths.
- It does not grant redistribution rights for any upstream component.

## Verification

```bash
python -m pytest -q \
  tests/test_assemble_hf_model.py \
  tests/test_deepencoderv2_mask.py \
  tests/test_wrapper_interfaces.py

sha256sum -c MANIFEST.sha256
```

For the live multimodal gate, use `tests/live_multimodal_smoke.py` against the started two-node server and require its text plus two-image OCR contract to pass.
