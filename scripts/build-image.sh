#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${DSPARK_VLLM_IMAGE:-anemll-dsv4-vision:0.1.1-dspark-headfix1}"
BASE_IMAGE="${DSPARK_VLLM_BASE_IMAGE:?set DSPARK_VLLM_BASE_IMAGE to the digest-pinned runtime reference}"

cd "$ROOT"
echo "Building $IMAGE from $ROOT/Dockerfile (context = repo root)"
docker build --build-arg DSPARK_VLLM_BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" -f Dockerfile .
docker image inspect "$IMAGE" --format 'OK {{.Id}} {{.RepoTags}}'
