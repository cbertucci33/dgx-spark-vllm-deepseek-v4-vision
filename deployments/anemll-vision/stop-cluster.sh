#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
head_env="${1:-config/head.smoke.env}"
[[ "$head_env" = /* ]] || head_env="$root/$head_env"
set -a
# shellcheck disable=SC1090
source "$head_env"
set +a
: "${WORKER_SSH:?WORKER_SSH is required}"

stop_local='ids=$(docker ps -aq --filter label=com.docker.compose.service=vllm-dspark); if [ -n "$ids" ]; then docker rm -f $ids; fi'
ssh -o BatchMode=yes "$WORKER_SSH" "$stop_local"
bash -lc "$stop_local"
echo "Both TP ranks stopped."
