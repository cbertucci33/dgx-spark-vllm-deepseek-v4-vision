---
license: apache-2.0
base_model:
  - cebeuq/DeepSeek-V4-Flash-0731-abliterated
  - FlyCockpit/DeepSeek-V4-Flash-0731-vision
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - deepseek-v4
  - multimodal
  - vision-language
  - screenshots
  - abliterated
  - uncensored
  - vllm
  - dgx-spark
---

# DeepSeek-V4-Flash-0731 Abliterated Vision

We were happy with Cebeuq's **DeepSeek-V4-Flash-0731 abliterated** checkpoint, but we wanted vision too. We did not want to spend roughly another 8 GB on the Kimi-derived vision towers used by the other DeepSeek vision grafts we considered, so we built this smaller composition ourselves from the Cebeuq language model and FlyCockpit's DeepEncoderV2 tower and projector.

This model requires our companion [`dgx-spark-vllm-deepseek-v4-vision`](https://github.com/cbertucci33/dgx-spark-vllm-deepseek-v4-vision/tree/v0.1.1) runtime repository to use its vision features. Stock Transformers and stock vLLM will not load the custom multimodal architecture correctly; use the pinned Anemll-based runtime and deployment instructions from that repository.

> **Uncensored model:** the language checkpoint has undergone abliteration to reduce refusal behavior. Treat outputs as untrusted, apply application-level safeguards, and do not assume the model will decline harmful requests.

> **User responsibility:** this model is provided without warranty. The maintainers are not responsible for what others generate, publish, deploy, or otherwise do with this abliterated model. Users must operate it responsibly, apply appropriate safeguards, comply with applicable law, and respect third-party rights.

## What “merged” means here

This is a **composition, not weight averaging or parameter blending**:

1. The complete language-model repository is copied from `cebeuq/DeepSeek-V4-Flash-0731-abliterated` at revision `21bd923c2574d9edcd7b914885024ce72fd5c076`.[2]
2. The DeepEncoderV2 tower and trained projector are copied from `FlyCockpit/DeepSeek-V4-Flash-0731-vision` at revision `d8efc7dfaceee965164d95952e2498b60fee323c`.[3]
3. `config.json` is updated to declare `DeepseekV4VisionForCausalLM` and record the image-token and preprocessing contract.
4. The original `model.safetensors.index.json` is required to remain byte-for-byte unchanged during assembly. No language tensor is rewritten by the vision composition step.
5. At inference, each image is encoded by DeepEncoderV2, projected from 896 dimensions into the language model’s 4096-dimensional embedding space, and spliced into the prompt at image token ID `129279`. A learned separator follows the view embeddings.

The underlying abliterated checkpoint itself is derived from DeepSeek’s official `DeepSeek-V4-Flash-0731` release.[1][2]

## Pinned components

| Component | Source | Revision / digest |
|---|---|---|
| Language checkpoint | `cebeuq/DeepSeek-V4-Flash-0731-abliterated` | revision `21bd923c2574d9edcd7b914885024ce72fd5c076`; weight-index SHA-256 `a93ace48e04f89e17222353191d30a5ba5744cbb8176a7a4832221656ce2d545`; 50-entry weight-manifest SHA-256 `6ddeea70678d7f11a6ea72514cbb799b7070617e9fa3d62f990722e27ba371f4` |
| Declared official base | `deepseek-ai/DeepSeek-V4-Flash-0731` | the Cebeuq card does not declare the exact official-base revision used |
| Vision tower + projector | `FlyCockpit/DeepSeek-V4-Flash-0731-vision` | `d8efc7dfaceee965164d95952e2498b60fee323c` |
| Tower file | `vision/tower/deepencoder_v2_tower.safetensors` | SHA-256 `9dcf6803d4c6b63acc4008bc2409e599a2ab6e3886e241f1727f61550c300df5` |
| Projector file | `vision/adapter/merged-004800-5af0c5.pt` | SHA-256 `6d0235333941210666bf347abb95e334943ef3f230dac65b83551186925468ec` |
| Runtime base | Locally supplied digest-pinned Anemll DGX Spark image | registry digest `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8` |
| Runtime source | Anemll GitHub repository ID `1301198905` | `47503f8e38dadd4dededca798150db2619594fce`[4] |
| Vision plugin lineage | `FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks` | `7cb20472e0f007a0626bd22ed9f5e22a8825c7e1`[5] |

The machine-readable copy of these pins is in `SOURCE_PINS.json`. `MANIFEST.sha256` covers every other file in the assembled repository; the manifest does not list itself.

## Vision contract

- Architecture: DeepEncoderV2 tower + MLP projector + learned view separator
- Input image size: `1024 × 1024`
- Image token: `<｜image｜>` (`129279`)
- Tokens per view: `256`
- Tiling: up to four local crops from a 2×2 grid, with one global view first and row-major crop order
- Image token counts: `257`, `769`, or `1281`, depending on aspect-aware tiling
- Main language KV cache format in the qualified runtime: `nvfp4_ds_mla`

## Qualified runtime

The release was exercised with:

- two NVIDIA GB10 systems in tensor parallel (`TP=2`);
- Anemll DSpark vLLM image `0.1.1` plus the included vision plugin;
- served model name `DeepSeek-V4-Flash-0731-Vision`;
- `max_model_len=800000`, `max_num_seqs=2`, and `gpu_memory_utilization=0.88`;
- measured startup cache admission of **1,791,777 tokens**, sufficient for two full 800K requests with **191,777 aggregate tokens** of admission headroom.

The measured number is specific to this exact hardware/runtime/profile. Re-profile after changing the GPU, runtime, graph mode, batching limits, vision implementation, or cache format.

## Serving

Use the required [`dgx-spark-vllm-deepseek-v4-vision`](https://github.com/cbertucci33/dgx-spark-vllm-deepseek-v4-vision/tree/v0.1.1) runtime repository and point it at this full model directory. The essential launch settings are:

```text
--served-model-name DeepSeek-V4-Flash-0731-Vision
--tensor-parallel-size 2
--kv-cache-dtype nvfp4_ds_mla
--block-size 256
--max-model-len 800000
--max-num-seqs 2
--gpu-memory-utilization 0.88
```

Example OpenAI-compatible request:

```bash
curl http://HEAD:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash-0731-Vision",
    "chat_template_kwargs": {"thinking": false},
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Read the text in this image."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}],
    "max_tokens": 128
  }'
```

## Validation performed

The qualified two-node deployment passed:

- API health and model-list checks;
- a deterministic text response;
- two independent image/OCR prompts (`ORBIT-7391` and `NOVA-2846`);
- rank health checks with no OOM;
- verification that the native DSpark/EAGLE3 path and `nvfp4_ds_mla` cache remained active.

This is a bounded deployment smoke test, not a comprehensive vision benchmark or safety evaluation. The redacted artifact identities, profile, observed outputs, and limitations are recorded in [`VALIDATION.md`](VALIDATION.md).

## Limitations

- The vision adapter is primarily described and tested for screenshots and UI understanding; it is not established as a production computer-use or coordinate-grounding model.[3][5]
- Only one image per prompt was qualified.
- Stock Transformers and stock vLLM are unsupported for this assembled layout.
- The language component is abliterated/uncensored and may generate unsafe, misleading, or policy-violating content.[2]
- OCR smoke tests do not establish broad visual reasoning quality.
- The 800K × 2 result is an admission/cache measurement, not proof that every possible pair of 800K prompts will complete under every workload.

## Licensing and redistribution

This assembled repository uses **Apache-2.0** as its top-level license. The incorporated MIT notices are retained under `LICENSES/`, and third-party components retain their applicable upstream terms:

- DeepSeek’s base model and the abliterated language checkpoint identify MIT terms.[1][2]
- The FlyCockpit model repository is tagged Apache-2.0 and its card identifies the adapter packaging as Apache-2.0.[3]
- The same card says the DeepEncoderV2 tower remains subject to upstream terms.[3]
- The runtime overlay is derived from MIT-licensed FlyCockpit packaging and runs over Anemll’s runtime; vLLM-derived code in Anemll remains Apache-2.0.[4][5]

This top-level license choice does not remove or replace retained third-party notices. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Reproducibility

See [`ASSEMBLY.md`](ASSEMBLY.md) for the exact composition algorithm and verification gates. The release assembler:

- rejects symlinked or overlapping input trees;
- validates all 48 language shards, the overlay shard, and the language-model weight index against the pinned 50-entry `LANGUAGE_WEIGHTS.sha256` manifest;
- requires every weight-index reference to be present in that manifest;
- validates pinned sizes and SHA-256 hashes for the tower, projector, and runtime wheel;
- copies into a staging directory and atomically renames it;
- rejects any change to the language-model weight index;
- emits a complete SHA-256 manifest.

## Citation

If you use this package, cite the original DeepSeek V4 release and credit both component repositories:

```bibtex
@misc{deepseekai2026deepseekv4,
  title  = {DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author = {DeepSeek-AI},
  year   = {2026}
}
```

Also cite the exact pinned source URLs below so the composition can be reconstructed.

## Sources

[1] [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
[2] [DeepSeek V4 Flash 0731 Abliterated](https://huggingface.co/cebeuq/DeepSeek-V4-Flash-0731-abliterated/tree/21bd923c2574d9edcd7b914885024ce72fd5c076)
[3] [DeepSeek V4 Flash 0731 Vision Assets](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision/tree/d8efc7dfaceee965164d95952e2498b60fee323c)
[4] [Anemll DGX Spark vLLM source](https://api.github.com/repositories/1301198905)
[5] [FlyCockpit DeepSeek V4 Vision Runtime](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks/tree/7cb20472e0f007a0626bd22ed9f5e22a8825c7e1)