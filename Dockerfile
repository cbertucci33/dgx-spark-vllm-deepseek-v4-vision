# Build the qualified vision plugin over the locally selected immutable
# Anemll DGX Spark runtime. Supply the complete digest-pinned reference with
# --build-arg; no upstream image identifier is embedded in this public tree.
ARG DSPARK_VLLM_BASE_IMAGE
FROM ${DSPARK_VLLM_BASE_IMAGE}

COPY plugin /tmp/dsv4-vision-vllm
COPY runtime-patches /tmp/dsv4-runtime-patches
RUN DSPARK_LOADER="$(python3 -c 'import inspect; import vllm.v1.worker.gpu.spec_decode.dspark.utils as module; print(inspect.getsourcefile(module))' | tail -n 1)" \
    && KV_CACHE_MANAGER="$(python3 -c 'import inspect; import vllm.v1.core.kv_cache_manager as module; print(inspect.getsourcefile(module))' | tail -n 1)" \
    && SCHEDULER="$(python3 -c 'import inspect; import vllm.v1.core.sched.scheduler as module; print(inspect.getsourcefile(module))' | tail -n 1)" \
    && python3 /tmp/dsv4-runtime-patches/patch_dspark_loader.py "${DSPARK_LOADER}" \
    && python3 /tmp/dsv4-runtime-patches/patch_dspark_prefix_cache.py "${KV_CACHE_MANAGER}" "${SCHEDULER}" \
    && python3 -m pip install --no-deps /tmp/dsv4-vision-vllm \
    && rm -rf /tmp/dsv4-vision-vllm /tmp/dsv4-runtime-patches
