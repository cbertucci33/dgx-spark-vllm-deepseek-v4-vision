# Credits and upstream lineage

Exact machine-readable pins are recorded in [`SOURCE_PINS.json`](SOURCE_PINS.json).

## Language model

- Official base: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- Abliterated derivative used by this release: [`cebeuq/DeepSeek-V4-Flash-0731-abliterated`](https://huggingface.co/cebeuq/DeepSeek-V4-Flash-0731-abliterated/tree/21bd923c2574d9edcd7b914885024ce72fd5c076)

The vision composition does not rewrite the language-model tensor index. The abliteration method and its own upstream credits are documented in the source model card.

## Vision tower and projector

- [`FlyCockpit/DeepSeek-V4-Flash-0731-vision`](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision/tree/d8efc7dfaceee965164d95952e2498b60fee323c)
- Runtime/plugin lineage: [`FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks`](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks/tree/7cb20472e0f007a0626bd22ed9f5e22a8825c7e1)

## DGX Spark vLLM runtime

- [Anemll upstream repository ID `1301198905`](https://api.github.com/repositories/1301198905), pinned revision `47503f8e38dadd4dededca798150db2619594fce`
- Qualified base-image reference: supplied locally through `DSPARK_VLLM_BASE_IMAGE`
- Registry digest: `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`

The Anemll runtime incorporates and credits vLLM, FlashInfer, B12X, NVIDIA CUDA/NCCL, and other dependencies in its own `CREDITS.md` and bundled license notices.

## Two-node packaging lineage

MiaAI-Lab’s public DSpark two-node recipes informed the worker-first launch pattern:

- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

## License boundary

Repository-local scripts, plugin modifications, tests, deployment files, and documentation are under [`LICENSE`](LICENSE), retaining upstream notices. Detectron2/ViTDet-derived portions in `deepencoderv2.py` retain Apache-2.0 terms, bundled at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt); the plugin wheel bundles both applicable license texts. Other vLLM-derived runtime files remain Apache-2.0 in the Anemll source repository. Model weights and vision assets retain their own terms and are not relicensed here. See [`model-release/THIRD_PARTY_NOTICES.md`](model-release/THIRD_PARTY_NOTICES.md).
