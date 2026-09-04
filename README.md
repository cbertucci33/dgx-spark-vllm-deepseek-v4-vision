# DGX Spark vLLM deployment for DeepSeek V4 Vision

A two-node vLLM deployment for serving our **[DeepSeek-V4-Flash-0731 Abliterated Vision](https://huggingface.co/cbert33/DeepSeek-V4-Flash-0731-abliterated-vision)** model on NVIDIA DGX Spark / GB10 systems through an OpenAI-compatible API.

This repository is built specifically for that Hugging Face model. Stock Transformers and stock vLLM do not load its custom multimodal architecture correctly; use this pinned Anemll DSpark overlay and the launch flow below.

**Current release:** **2.0** (`2.0`). It adds the reviewed DSpark head-sharing, prefix-cache replay, multimodal-layout, two-node image-identity, and acceptance-measurement fixes described in [`RELEASE_NOTES_2.0.md`](RELEASE_NOTES_2.0.md). The optional expanded DSpark draft-window override is deliberately not included.

> **Repository shape:** this is not a vendored copy of all vLLM source. It is the publishable overlay that installs on the pinned Anemll DGX Spark runtime. The base image contains the tested vLLM fork; this repository contains the multimodal plugin, model assembler, deployment profiles, and validation suite. `SOURCE_PINS.json` identifies the upstream repository by numeric GitHub ID and immutable revision without embedding its legacy product identifier.

## Quick start: launch the model

You need two DGX Spark / GB10 nodes with Docker, NVIDIA Container Toolkit, a working high-speed interconnect, passwordless SSH from the head to the worker, and the complete Hugging Face model directory available at the same path on both nodes.

```bash
git clone https://github.com/cbertucci33/dgx-spark-vllm-deepseek-v4-vision.git
cd dgx-spark-vllm-deepseek-v4-vision

export DSPARK_VLLM_BASE_IMAGE="$(python3 scripts/resolve_dgx_spark_base.py)"
docker build \
  --build-arg DSPARK_VLLM_BASE_IMAGE="$DSPARK_VLLM_BASE_IMAGE" \
  -t anemll-dsv4-vision:0.1.1-dspark-headfix1 .

cp deployments/anemll-vision/config/head.example.env \
   deployments/anemll-vision/config/head.env
cp deployments/anemll-vision/config/worker.example.env \
   deployments/anemll-vision/config/worker.env
```

Edit `head.env` and `worker.env` with the actual model path, fabric addresses/interfaces, worker SSH target, repository path, and image tag for both nodes. Do not copy the example network values blindly. Synchronize this repository and the canonical [DeepSeek V4 Flash 0731 abliterated vision v2 model](https://huggingface.co/cbert33/DeepSeek-V4-Flash-0731-abliterated-vision-v2) to the worker, then launch from the head node:

```bash
cd deployments/anemll-vision
./start-cluster.sh config/head.env config/worker.env
```

Verify the OpenAI-compatible endpoint:

```bash
curl http://HEAD_MANAGEMENT_IP:8000/v1/models
```

The launcher starts the worker rank first, starts rank 0, and waits for bounded API readiness. Stop both ranks with:

```bash
./stop-cluster.sh config/head.env
```

See [Configure two nodes](#configure-two-nodes), [Launch](#launch), and [API smoke test](#api-smoke-test) below for the full deployment and validation flow.

## Pinned release contract

| Setting | Public value |
|---|---|
| Runtime base | Digest-pinned Anemll DGX Spark image resolved from `SOURCE_PINS.json` |
| Runtime digest | `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8` |
| vLLM runtime version | `0.25.2.dev0+g752a3a504.d20260714` |
| Model ID | `DeepSeek-V4-Flash-0731-Vision` |
| Main KV cache | `nvfp4_ds_mla`, block size 256 |
| Speculation | native DSpark / EAGLE3 |

Machine-specific image IDs, host details, local paths, and deployment measurements are intentionally excluded. Rebuild the overlay, synchronize its content-addressed image ID across ranks, and run the complete live validation sequence on the target environment.

## Starting point and what we changed

We started from [`FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks@7cb2047`](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks/tree/7cb20472e0f007a0626bd22ed9f5e22a8825c7e1), which in turn targets the Anemll DSpark vLLM runtime. This repository turns that starting point into a pinned, reproducible release for the Cebeuq abliterated checkpoint:

- A vLLM general plugin registering `DeepseekV4VisionForCausalLM` without patching the installed vLLM package.
- DeepEncoderV2 image preprocessing, aspect-aware tiling, projector loading, and image-embedding splicing.
- Preservation of raw input token IDs for DeepSeek V4 hash-MoE routing.
- Explicit delegation to the pinned native DeepSeek V4 implementation for DSpark/EAGLE3 behavior.
- A fail-closed runtime patch that aliases the DSpark draft `lm_head` from the unwrapped language model for multimodal targets, inherits the DeepSeek target attention backend, selects the draft quantization config, and validates vocabulary-compatible weight sharing.
- A fail-closed Anemll backport that preserves the DSpark sliding-attention window across prefix-cache hits by replaying the final draft window through the target model.
- A live DSpark smoke gate that aggregates all exact Prometheus series, rejects counter resets and concurrent traffic, validates per-position acceptance shape, requires a minimum draft sample, and records acceptance rate and length. Its default 20% floor catches catastrophic regressions; it is not the expected-performance target.
- A Transformers 5.13.1-compatible additive attention-mask path with deterministic tests.
- A fail-closed Hugging Face model assembler with source/hash pins and atomic output.
- Two-node worker-first launch scripts and bounded health checks.
- Live text plus two-image OCR validation.

## Recommended usage

- Use this repository with [`cbert33/DeepSeek-V4-Flash-0731-abliterated-vision`](https://huggingface.co/cbert33/DeepSeek-V4-Flash-0731-abliterated-vision); stock vLLM does not provide this custom multimodal architecture.
- Start with the supplied two-node `TP=2` profile on GB10 systems and re-measure cache capacity before changing context, concurrency, graph mode, or cache format.
- Keep the worker-first startup order and run the text plus genuine-image smoke test before connecting clients.
- Treat model outputs as untrusted. This is an abliterated checkpoint with reduced refusal behavior; operators are responsible for safeguards, legal compliance, and downstream use.

## Upstream lineage

| Component | Source |
|---|---|
| DGX Spark vLLM base | [Anemll repository ID `1301198905`](https://api.github.com/repositories/1301198905), revision `47503f8` |
| Vision plugin lineage | [`FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks@7cb2047`](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks/tree/7cb20472e0f007a0626bd22ed9f5e22a8825c7e1) |
| Language checkpoint | [`cebeuq/DeepSeek-V4-Flash-0731-abliterated@21bd923`](https://huggingface.co/cebeuq/DeepSeek-V4-Flash-0731-abliterated/tree/21bd923c2574d9edcd7b914885024ce72fd5c076) |
| Vision assets | [`FlyCockpit/DeepSeek-V4-Flash-0731-vision@d8efc7d`](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision/tree/d8efc7dfaceee965164d95952e2498b60fee323c) |

Exact revisions, image digests, artifact hashes, and asset sizes are machine-readable in [`SOURCE_PINS.json`](SOURCE_PINS.json).

## Repository layout

```text
plugin/                         vLLM general plugin source
runtime-patches/                 fail-closed patches for the pinned vLLM runtime
scripts/assemble_hf_model.py    independent-model assembler
tests/                          unit and live multimodal validation
deployments/anemll-vision/      two-node compose and launch scripts
model-release/                  HF model card, assembly notes, notices, checklist
Dockerfile                      source-built overlay on the pinned Anemll image
SOURCE_PINS.json                immutable provenance and artifact hashes
CHANGELOG.md                     public release history
RELEASE_NOTES_2.0.md             2.0 changes and qualification record
```

Local `.env` profiles, rollback profiles, wheels, caches, and machine-specific release evidence are excluded by `.gitignore`.

## Prerequisites

- Two NVIDIA GB10 systems with Docker and NVIDIA Container Toolkit.
- A working high-speed inter-node fabric supported by NCCL.
- The same complete model directory on both nodes.
- Passwordless SSH from the head node to the worker for the supplied launcher, or equivalent manual orchestration.
- Enough local storage for the model and container image.

Do not copy the example network interface names blindly. Determine the actual fabric interfaces, RoCE devices, GID, and IPs on each host.

## Build the overlay image

The plugin is installed from source during the image build; no prebuilt wheel is required in Git:

```bash
export DSPARK_VLLM_BASE_IMAGE="$(python3 scripts/resolve_dgx_spark_base.py)"
docker build \
  --build-arg DSPARK_VLLM_BASE_IMAGE="$DSPARK_VLLM_BASE_IMAGE" \
  -t anemll-dsv4-vision:0.1.1-dspark-headfix1 \
  .
```

`scripts/resolve_dgx_spark_base.py` queries the pinned numeric GitHub repository ID, verifies the returned ID, and constructs the complete immutable Anemll DGX Spark image reference using the tag and registry digest in `SOURCE_PINS.json`. The public Dockerfile deliberately contains no legacy product identifier; the build argument is required and has no default.

## Assemble the model repository

The assembler composes—rather than averages—the language checkpoint and vision assets. It leaves the original language weight index unchanged and verifies every language shard, the overlay shard, and the index against `LANGUAGE_WEIGHTS.sha256`.

```bash
uv run --python 3.12 python scripts/assemble_hf_model.py \
  --source-model /path/to/DeepSeek-V4-Flash-0731-abliterated \
  --vision-assets /path/to/vision-assets \
  --runtime-artifacts /path/to/runtime-artifacts \
  --release-metadata model-release \
  --source-pins SOURCE_PINS.json \
  --destination /path/to/DeepSeek-V4-Flash-0731-abliterated-vision
```

Prepare the structured runtime-artifacts directory with:

```bash
rm -rf /tmp/dsv4-runtime-artifacts
mkdir -p /tmp/dsv4-runtime-artifacts/wheels
SOURCE_DATE_EPOCH=1787270400 uv build --wheel --python 3.12 \
  --out-dir /tmp/dsv4-runtime-artifacts/wheels ./plugin
```

The fixed build epoch makes the wheel reproducible; `scripts/release-check.sh` builds it twice and checks the SHA-256 recorded in `SOURCE_PINS.json`.

See [`model-release/ASSEMBLY.md`](model-release/ASSEMBLY.md) for the exact composition and verification contract.

## Configure two nodes

Copy the publish-safe templates:

```bash
cp deployments/anemll-vision/config/head.example.env \
   deployments/anemll-vision/config/head.env
cp deployments/anemll-vision/config/worker.example.env \
   deployments/anemll-vision/config/worker.env
```

Edit both files. At minimum set:

- head and worker fabric IPs;
- NCCL/RoCE devices and socket interfaces;
- model, Hugging Face cache, and temporary-storage paths;
- worker SSH target and repository path;
- the image tag built above.

`start-cluster.sh` resolves `DSPARK_VLLM_IMAGE` on both nodes and aborts before either rank starts unless the content-addressed Docker image IDs match. `DSPARK_VLLM_IMAGE_ID` is an optional additional pin; when set, both nodes must match it. Rebuilds produce new IDs, so update the pin deliberately after a verified build. For a published image, also prefer an immutable registry digest over a mutable tag.

The vision loader fails closed rather than serving randomly initialized or
layout-incompatible vision modules:

- `DSV4_VISION_TOWER` and `DSV4_VISION_ADAPTER` must identify existing files;
- checkpoint `config.tiles` is authoritative;
- `DSV4_VISION_TILES`, when set, must agree with checkpoint metadata;
- legacy adapters without tile metadata require an explicit, verified
  `DSV4_VISION_TILES` value.

The Compose profiles intentionally expose only runtime settings consumed by the
pinned implementation. Unsupported historical `VLLM_DSPARK_*` and
`VLLM_DSV4_DSPARK_*` pseudo-controls are omitted.

The qualified release values are already represented in the templates:

```text
MAX_MODEL_LEN=800000
MAX_NUM_SEQS=2
GPU_MEMORY_UTILIZATION=0.88
SERVED_MODEL_NAME=DeepSeek-V4-Flash-0731-Vision
```

Synchronize the repository and model directory to both nodes before launch.

## Launch

From the deployment directory on the head node:

```bash
cd deployments/anemll-vision
./start-cluster.sh config/head.env config/worker.env
```

The launcher first proves that the image tag resolves to the same content-addressed image on both nodes. It then starts the headless worker, waits for its container, starts rank 0, and polls the API with a bounded readiness timeout. The speculative profile explicitly captures the full `MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)` CUDA-graph shape so Anemll does not truncate a 12-token maximum to its default 8-token capture bucket.

Stop both ranks with:

```bash
./stop-cluster.sh config/head.env
```

## API smoke test

```bash
curl http://HEAD_MANAGEMENT_IP:8000/v1/models
```

Text request:

```bash
curl http://HEAD_MANAGEMENT_IP:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash-0731-Vision",
    "chat_template_kwargs": {"thinking": false},
    "messages": [{"role": "user", "content": "Reply with exactly: VISION TEXT PATH OK"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

For image validation, run [`tests/live_multimodal_smoke.py`](tests/live_multimodal_smoke.py) with two independently generated test images. The contract requires the text response plus both OCR strings to match exactly.

## Tests

Fast local and deterministic packaging gate:

```bash
bash scripts/release-check.sh
```

This runs the Python 3.12 tests—including assembler hardening and deployment-contract checks—renders the publish-safe Compose example when Docker Compose is available, builds the wheel twice with the fixed epoch, compares the artifacts byte-for-byte, and verifies the size/SHA-256 in `SOURCE_PINS.json`.

Build the final image separately:

```bash
export DSPARK_VLLM_BASE_IMAGE="$(python3 scripts/resolve_dgx_spark_base.py)"
docker build \
  --build-arg DSPARK_VLLM_BASE_IMAGE="$DSPARK_VLLM_BASE_IMAGE" \
  -t anemll-dsv4-vision:0.1.1-dspark-headfix1 .
```

Live qualification additionally requires:

1. both TP ranks running with `OOMKilled=false`;
2. `/health` and `/v1/models` responding;
3. logs confirming `nvfp4_ds_mla`, native B12X, and DSpark/EAGLE3;
4. measured cache admission above the intended aggregate token requirement;
5. text and genuine image requests passing;
6. `tests/measure_vision_spec_rails.py` issuing up to `MAX_DSPARK_QUALIFICATION_REQUESTS` requests (default `8`) until it records an uncontaminated sample at or above `MIN_DSPARK_ACCEPTANCE_RATE` (default `0.20`) with at least `MIN_DSPARK_DRAFTS` drafts (default `8`), stable per-label counter identities, consistent per-position counters, and matching completion/generation token counts. Treat this only as a catastrophic-regression smoke floor; final qualification must compare representative text, genuine-image, and coding workloads at the same configured draft length against the historical acceptance and throughput range.

## Model-name changes

Clients must use the ID returned by `/v1/models`. This vision profile serves:

```text
DeepSeek-V4-Flash-0731-Vision
```

The preserved text-only profile uses `DeepSeek-V4-Flash-0731`. Switching profiles therefore requires updating clients such as OpenClaw; changing only the endpoint is insufficient.

## Limitations

- This overlay is pinned to the Anemll `0.1.1` runtime ABI and its native DeepSeek V4 implementation.
- It is not a generic multimodal vLLM plugin for unrelated DeepSeek releases.
- One image per prompt was qualified.
- OCR smoke tests are not a comprehensive visual benchmark.
- The abliterated language model has substantially reduced refusal behavior.
- Vision assets retain their upstream terms; see [`model-release/THIRD_PARTY_NOTICES.md`](model-release/THIRD_PARTY_NOTICES.md).

## Licenses

Repository-local overlay, deployment, test, and documentation changes are MIT-licensed under [`LICENSE`](LICENSE), retaining upstream notices. vLLM-derived runtime code remains under Apache-2.0 in the Anemll source repository. Model weights and vision assets retain their own terms and are not relicensed here.

See [`CREDITS.md`](CREDITS.md) and [`model-release/THIRD_PARTY_NOTICES.md`](model-release/THIRD_PARTY_NOTICES.md).

## Responsibility

This software is provided without warranty. The maintainers are not responsible for model outputs or for applications built with this abliterated checkpoint. Users must operate it responsibly, apply appropriate safeguards, and comply with applicable laws and third-party rights.
