#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
env_file="${1:?usage: start-node.sh <env-file>}"
[[ "$env_file" = /* ]] || env_file="$root/$env_file"
[[ -f "$env_file" ]] || { echo "Missing environment file: $env_file" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

for name in NODE_RANK MASTER_ADDR MASTER_PORT VLLM_HOST_IP NCCL_IB_HCA NCCL_SOCKET_IFNAME DSPARK_MODEL_HOST DSPARK_VLLM_IMAGE; do
  [[ -n "${!name:-}" ]] || { echo "$name is required in $env_file" >&2; exit 1; }
done
[[ -d "$DSPARK_MODEL_HOST" ]] || { echo "Model directory is missing: $DSPARK_MODEL_HOST" >&2; exit 1; }
if [[ -n "${DSPARK_VLLM_IMAGE_ID:-}" ]]; then
  actual_image_id="$(docker image inspect "$DSPARK_VLLM_IMAGE" --format '{{.Id}}')"
  [[ "$actual_image_id" = "$DSPARK_VLLM_IMAGE_ID" ]] || {
    echo "Unexpected vision image identity for $DSPARK_VLLM_IMAGE: $actual_image_id" >&2
    exit 1
  }
else
  docker image inspect "$DSPARK_VLLM_IMAGE" >/dev/null
fi

ids="$(docker ps -aq --filter label=com.docker.compose.service=vllm-dspark)"
if [[ -n "$ids" ]]; then
  # shellcheck disable=SC2086
  docker rm -f $ids
fi

COMPOSE_DISABLE_ENV_FILE=1 docker compose \
  -p anemll-dsv4-vision \
  --env-file "$env_file" \
  -f "$root/docker-compose.yml" \
  up -d

docker ps --filter label=com.docker.compose.service=vllm-dspark \
  --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
