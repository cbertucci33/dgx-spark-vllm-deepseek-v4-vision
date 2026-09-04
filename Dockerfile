# Build the qualified vision plugin over the locally selected immutable
# Anemll DGX Spark runtime. Supply the complete digest-pinned reference with
# --build-arg; no upstream image identifier is embedded in this public tree.
ARG DSPARK_VLLM_BASE_IMAGE
FROM ${DSPARK_VLLM_BASE_IMAGE}

COPY plugin /tmp/dsv4-vision-vllm
COPY runtime-patches /tmp/dsv4-runtime-patches
RUN PURELIB="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && DSPARK_LOADER="${PURELIB}/vllm/v1/worker/gpu/spec_decode/dspark/utils.py" \
    && KV_CACHE_MANAGER="${PURELIB}/vllm/v1/core/kv_cache_manager.py" \
    && SCHEDULER="${PURELIB}/vllm/v1/core/sched/scheduler.py" \
    && test -f "${DSPARK_LOADER}" -a -f "${KV_CACHE_MANAGER}" -a -f "${SCHEDULER}" \
    && python3 /tmp/dsv4-runtime-patches/patch_dspark_loader.py "${DSPARK_LOADER}" \
    && python3 /tmp/dsv4-runtime-patches/patch_dspark_prefix_cache.py "${KV_CACHE_MANAGER}" "${SCHEDULER}" \
    && rm -f \
        "$(dirname "${DSPARK_LOADER}")/__pycache__/utils.cpython-312.pyc" \
        "$(dirname "${KV_CACHE_MANAGER}")/__pycache__/kv_cache_manager.cpython-312.pyc" \
        "$(dirname "${SCHEDULER}")/__pycache__/scheduler.cpython-312.pyc" \
    && python3 -m pip install --no-deps /tmp/dsv4-vision-vllm \
    && rm -rf /tmp/dsv4-vision-vllm /tmp/dsv4-runtime-patches
