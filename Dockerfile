# Build the qualified vision plugin over the locally selected immutable
# Anemll DGX Spark runtime. Supply the complete digest-pinned reference with
# --build-arg; no upstream image identifier is embedded in this public tree.
ARG DSPARK_VLLM_BASE_IMAGE
FROM ${DSPARK_VLLM_BASE_IMAGE}

COPY plugin /tmp/dsv4-vision-vllm
RUN python3 -m pip install --no-deps /tmp/dsv4-vision-vllm \
    && rm -rf /tmp/dsv4-vision-vllm
