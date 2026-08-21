#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
head_env="${1:-config/head.smoke.env}"
worker_env="${2:-config/worker.smoke.env}"
[[ "$head_env" = /* ]] || head_env="$root/$head_env"
[[ -f "$head_env" ]] || { echo "Missing $head_env" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$head_env"
set +a
: "${WORKER_SSH:?WORKER_SSH is required}"
: "${WORKER_REPO_DIR:?WORKER_REPO_DIR is required}"

printf 'Starting vision TP rank 1 on %s...\n' "$WORKER_SSH"
ssh -o BatchMode=yes "$WORKER_SSH" \
  "cd '$WORKER_REPO_DIR' && ./start-node.sh '$worker_env'"

printf 'Waiting 12 seconds for rank 1 rendezvous...\n'
sleep 12
printf 'Starting vision TP rank 0...\n'
"$root/start-node.sh" "$head_env"

port="${VLLM_PORT:-8000}"
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "Vision vLLM is ready: http://127.0.0.1:${port}"
    curl -fsS "http://127.0.0.1:${port}/version"
    echo
    exit 0
  fi
  sleep 5
done

echo "Timed out waiting for vision vLLM readiness." >&2
exit 1