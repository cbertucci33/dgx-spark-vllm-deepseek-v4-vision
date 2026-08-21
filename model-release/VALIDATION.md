# Validation record

Date: 2026-08-20

This is a redacted record of bounded qualification performed on a two-node NVIDIA DGX Spark / GB10 deployment. It records the observed contract without publishing private hostnames, addresses, credentials, or local filesystem paths. It is not a benchmark or comprehensive model evaluation.

## Artifact identity

- Language checkpoint: `cebeuq/DeepSeek-V4-Flash-0731-abliterated` at `21bd923c2574d9edcd7b914885024ce72fd5c076`
- Language weight-index SHA-256: `a93ace48e04f89e17222353191d30a5ba5744cbb8176a7a4832221656ce2d545`
- Vision assets: `FlyCockpit/DeepSeek-V4-Flash-0731-vision` at `d8efc7dfaceee965164d95952e2498b60fee323c`
- Tower SHA-256: `9dcf6803d4c6b63acc4008bc2409e599a2ab6e3886e241f1727f61550c300df5`
- Projector SHA-256: `6d0235333941210666bf347abb95e334943ef3f230dac65b83551186925468ec`
- Qualified Anemll base image digest: `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`

## Qualified serving profile

```text
served_model_name=DeepSeek-V4-Flash-0731-Vision
tensor_parallel_size=2
max_model_len=800000
max_num_seqs=2
gpu_memory_utilization=0.88
cache_dtype=nvfp4_ds_mla
speculative_runtime=DSpark/EAGLE3 enabled
```

Observed runtime state:

```text
rank0=running, OOM=false
rank1=running, OOM=false
health_endpoint=PASS
reported_cache_capacity=1,791,777 tokens
required_capacity_for_800K_x2=1,600,000 tokens
measured_headroom=191,777 tokens
```

The capacity figure is the runtime's measured/admission token capacity. It is not calculated as physical cache blocks multiplied by a nominal block size; that shortcut is invalid for this hybrid DeepSeek cache layout.

## Text and image contract

The live OpenAI-compatible chat endpoint was exercised with `chat_template_kwargs={"thinking": false}` and deterministic decoding.

Expected and observed text output:

```text
VISION TEXT PATH OK
```

Two independently generated images containing distinct text were submitted through the actual image-conditioned request path. Expected and observed OCR outputs:

```text
ORBIT-7391
NOVA-2846
```

This verifies that the request handler decoded image pixels, constructed multimodal inputs, and produced image-dependent answers for these bounded cases. It does not establish broad OCR accuracy or general visual reasoning quality.

## Release artifact checks

The publication-prep runtime wheel was built twice with Python 3.12, `uv`, and `SOURCE_DATE_EPOCH=1787270400`; the outputs compared byte-for-byte. The wheel includes the applicable MIT and Apache-2.0 license texts.

The final source-built release image was checked for package version `0.1.1`, vLLM plugin registration, and bundled license files. It was not swapped into the running qualified service because preserving the known-good live deployment was the rollback boundary; executable changes after live qualification were limited to attribution/comments and package/release metadata.

Both physical assembled model copies were subjected to `sha256sum -c MANIFEST.sha256`, covering every listed language shard, overlay, tower, projector, tokenizer/config file, and runtime artifact.

## Limits

- No comprehensive safety evaluation was performed.
- No broad vision benchmark was run.
- Long-context admission was verified; an 800,000-token semantic-quality benchmark was not performed.
- This package uses an abliterated/uncensored language checkpoint.
- Public redistribution of the model weights remains subject to the upstream terms for the DeepEncoderV2 tower; this does not restrict publication of the companion runtime source repository.
