#!/usr/bin/env bash
set -euo pipefail

root=${VISION_ASSETS_DIR:-"${HOME}/models/staging/deepseek-v4-vision/assets"}
stage="${root}.download"
final="$root"
revision=d8efc7dfaceee965164d95952e2498b60fee323c
repo=FlyCockpit/DeepSeek-V4-Flash-0731-vision

mkdir -p "$stage/tower" "$stage/adapter" "$final/tower" "$final/adapter"

fetch() {
  local rel="$1" expected_size="$2" expected_sha="$3"
  local partial="$stage/$rel" destination="$final/$rel"
  local url="https://huggingface.co/$repo/resolve/$revision/$rel"

  echo "FETCH_START $rel expected_bytes=$expected_size"
  curl -fL --retry 8 --retry-all-errors --retry-delay 3 \
    -C - -o "$partial" "$url"

  local actual_size
  actual_size=$(stat -c %s "$partial")
  test "$actual_size" = "$expected_size"
  echo "$expected_sha  $partial" | sha256sum -c -
  mv "$partial" "$destination"
  echo "FETCH_VERIFIED $rel bytes=$actual_size sha256=$expected_sha"
}

fetch \
  tower/deepencoder_v2_tower.safetensors \
  906533408 \
  9dcf6803d4c6b63acc4008bc2409e599a2ab6e3886e241f1727f61550c300df5
fetch \
  adapter/merged-004800-5af0c5.pt \
  40921678 \
  6d0235333941210666bf347abb95e334943ef3f230dac65b83551186925468ec

rmdir "$stage/tower" "$stage/adapter" "$stage" 2>/dev/null || true
echo ASSET_DOWNLOAD_COMPLETE
