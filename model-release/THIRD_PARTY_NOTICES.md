# Third-party notices

This assembled model uses Apache-2.0 as its top-level license. Independently sourced components retain their applicable upstream notices and terms.

## Language model

- Component: `cebeuq/DeepSeek-V4-Flash-0731-abliterated`
- Pinned revision: `21bd923c2574d9edcd7b914885024ce72fd5c076`
- Declared license: MIT
- Base model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Base model declared license: MIT

The language checkpoint is an abliterated derivative. Its source model card describes its modification method, safety implications, and validation.

## Vision assets

- Component: `FlyCockpit/DeepSeek-V4-Flash-0731-vision`
- Pinned revision: `d8efc7dfaceee965164d95952e2498b60fee323c`
- Projector/adapter packaging declared license: Apache-2.0
- DeepEncoderV2 tower: source card states that upstream terms apply

The source repository is tagged Apache-2.0, while its card says the tower remains subject to upstream terms. This notice preserves that qualification.

## Runtime

- Anemll DGX Spark runtime: GitHub repository ID `1301198905` at `47503f8e38dadd4dededca798150db2619594fce`
- Vision plugin lineage: `FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks` at `7cb20472e0f007a0626bd22ed9f5e22a8825c7e1`
- Repository-local plugin changes and FlyCockpit-derived portions retain MIT terms; the complete text is bundled as `LICENSES/MIT-FlyCockpit.txt` and inside the runtime wheel.
- Detectron2/ViTDet-derived portions of `deepencoderv2.py` retain Apache-2.0 terms; the complete text is bundled as `LICENSES/Apache-2.0.txt` and inside the runtime wheel.
- The projector/adapter packaging is declared Apache-2.0 by its source card; `LICENSES/Apache-2.0.txt` is bundled for that declared packaging license.
- Other vLLM-derived runtime files remain under their upstream Apache-2.0 terms.

Bundling a license text records the applicable terms and retained notices; it does not relicense third-party material.

## No implied endorsement

The names DeepSeek, Hugging Face, vLLM, NVIDIA, Anemll, FlyCockpit, and other project names are used only to identify upstream components. Their inclusion does not imply endorsement of this assembled package.
