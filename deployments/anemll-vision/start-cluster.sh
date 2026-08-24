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

verify_cluster_image_consistency() {
  local head_image_id worker_image_id worker_image worker_expected_image_id
  local worker_env_cmd worker_env_values inspect_cmd
  local -a worker_env_lines

  if ! head_image_id=$(docker image inspect --format '{{.Id}}' "$DSPARK_VLLM_IMAGE" 2>/dev/null) \
      || [[ -z "$head_image_id" ]]; then
    echo "Could not inspect vision image '$DSPARK_VLLM_IMAGE' on the head node." >&2
    return 1
  fi
  head_image_id=$(printf '%s' "$head_image_id" | tr -d '\015')

  printf -v worker_env_cmd \
    "cd %q && set -a && source %q && : \"\${DSPARK_VLLM_IMAGE:?DSPARK_VLLM_IMAGE is required}\" && printf '%%s\\n%%s\\n' \"\$DSPARK_VLLM_IMAGE\" \"\${DSPARK_VLLM_IMAGE_ID:-}\"" \
    "$WORKER_REPO_DIR" "$worker_env"
  if ! worker_env_values=$(ssh -o BatchMode=yes "$WORKER_SSH" "$worker_env_cmd" 2>/dev/null); then
    echo "Could not resolve DSPARK_VLLM_IMAGE from worker environment '$worker_env'." >&2
    return 1
  fi
  mapfile -t worker_env_lines <<< "$worker_env_values"
  worker_image=${worker_env_lines[0]:-}
  worker_expected_image_id=${worker_env_lines[1]:-}
  if [[ -z "$worker_image" ]]; then
    echo "Worker environment '$worker_env' did not define DSPARK_VLLM_IMAGE." >&2
    return 1
  fi

  printf -v inspect_cmd "docker image inspect --format '{{.Id}}' %q" "$worker_image"
  if ! worker_image_id=$(ssh -o BatchMode=yes "$WORKER_SSH" "$inspect_cmd" 2>/dev/null) \
      || [[ -z "$worker_image_id" ]]; then
    echo "Could not inspect vision image '$worker_image' on worker '$WORKER_SSH'." >&2
    return 1
  fi
  worker_image_id=$(printf '%s' "$worker_image_id" | tr -d '\015')

  if [[ -n "${DSPARK_VLLM_IMAGE_ID:-}" ]]; then
    if [[ "$head_image_id" != "$DSPARK_VLLM_IMAGE_ID" ]]; then
      echo "Head image ID '$head_image_id' does not match DSPARK_VLLM_IMAGE_ID '$DSPARK_VLLM_IMAGE_ID'." >&2
      return 1
    fi
  fi
  if [[ -n "$worker_expected_image_id" ]] \
      && [[ "$worker_image_id" != "$worker_expected_image_id" ]]; then
    echo "Worker image ID '$worker_image_id' does not match worker DSPARK_VLLM_IMAGE_ID '$worker_expected_image_id'." >&2
    return 1
  fi

  if [[ "$worker_image_id" != "$head_image_id" ]]; then
    echo "Docker image mismatch between head '$DSPARK_VLLM_IMAGE' and worker '$worker_image'." >&2
    echo "Head:   $head_image_id ($DSPARK_VLLM_IMAGE)" >&2
    echo "Worker: $worker_image_id ($worker_image)" >&2
    echo "Cluster launch aborted before either rank started." >&2
    return 1
  fi

  echo "Docker image consistency check passed: $head_image_id"
}

verify_cluster_image_consistency

printf 'Starting vision TP rank 1 on %s...\n' "$WORKER_SSH"
printf -v worker_launch_cmd "cd %q && ./start-node.sh %q" \
  "$WORKER_REPO_DIR" "$worker_env"
ssh -o BatchMode=yes "$WORKER_SSH" "$worker_launch_cmd"

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
