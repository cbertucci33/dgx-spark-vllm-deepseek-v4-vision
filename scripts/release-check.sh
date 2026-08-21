#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python3 -m json.tool SOURCE_PINS.json >/dev/null
python3 scripts/release_audit.py
bash -n scripts/*.sh deployments/anemll-vision/*.sh

uv run --python 3.12 --with pytest pytest -q \
  tests/test_assemble_hf_model.py \
  tests/test_deepencoderv2_mask.py \
  tests/test_wrapper_interfaces.py \
  tests/test_release_contract.py

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is required for the release contract" >&2
  exit 1
fi
for compose_file in docker-compose.yml docker-compose.nospec.yml; do
  COMPOSE_DISABLE_ENV_FILE=1 docker compose \
    -f "deployments/anemll-vision/$compose_file" \
    --env-file deployments/anemll-vision/config/head.example.env \
    config >/dev/null
done

build_a=$(mktemp -d)
build_b=$(mktemp -d)
trap 'rm -rf "$build_a" "$build_b"' EXIT
SOURCE_DATE_EPOCH=1787270400 uv build --wheel --python 3.12 --out-dir "$build_a" ./plugin >/dev/null
SOURCE_DATE_EPOCH=1787270400 uv build --wheel --python 3.12 --out-dir "$build_b" ./plugin >/dev/null
cmp "$build_a"/*.whl "$build_b"/*.whl

expected=$(python3 -c 'import json; print(json.load(open("SOURCE_PINS.json"))["plugin_artifact"]["sha256"])')
expected_size=$(python3 -c 'import json; print(json.load(open("SOURCE_PINS.json"))["plugin_artifact"]["size"])')
canonical_platform=$(python3 -c 'import json; print(json.load(open("SOURCE_PINS.json"))["plugin_artifact"]["canonical_build_platform"])')
actual=$(python3 - "$build_a"/*.whl <<'PY'
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())
PY
)
actual_size=$(python3 - "$build_a"/*.whl <<'PY'
import os,sys
print(os.path.getsize(sys.argv[1]))
PY
)

pin_enforced=no
if [[ "$(uname -s)" == "Linux" && "$canonical_platform" == "linux" ]]; then
  printf 'Canonical wheel: actual_sha256=%s expected_sha256=%s actual_size=%s expected_size=%s\n' \
    "$actual" "$expected" "$actual_size" "$expected_size"
  test "$actual" = "$expected"
  test "$actual_size" = "$expected_size"
  pin_enforced=yes
fi

echo "RELEASE_CHECK_PASS wheel_sha256=$actual wheel_size=$actual_size canonical_linux_pin_enforced=$pin_enforced"
