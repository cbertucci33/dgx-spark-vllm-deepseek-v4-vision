#!/usr/bin/env python3
"""Patch the pinned DSpark loader for multimodal target head sharing."""

from __future__ import annotations

import hashlib
import pathlib
import sys


REPLACEMENTS = (
    (
        "from vllm.model_executor.model_loader import get_model\n",
        "from vllm.model_executor.model_loader import get_model\n"
        "from vllm.model_executor.models.utils import get_draft_quant_config\n",
    ),
    (
        "\n\ndef load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:\n",
        "\n\ndef _resolve_dspark_attention_backend(\n"
        "    draft_model_config, draft_backend, target_backend\n"
        "):\n"
        "    if draft_backend is not None:\n"
        "        return draft_backend\n"
        "    if draft_model_config.hf_config.model_type == \"deepseek_v4\":\n"
        "        return target_backend\n"
        "    return None\n\n\n"
        "def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:\n",
    ),
    (
        "    # DSpark uses non-causal attention.\n"
        "    causal = False\n"
        "    draft_vllm_config = replace(\n",
        "    # DeepSeek V4 DSpark shares the target KV-cache layout.\n"
        "    draft_attention_backend = _resolve_dspark_attention_backend(\n"
        "        draft_model_config,\n"
        "        speculative_config.attention_backend,\n"
        "        vllm_config.attention_config.backend,\n"
        "    )\n"
        "    causal = False\n"
        "    draft_vllm_config = replace(\n",
    ),
    (
        "            backend=speculative_config.attention_backend,\n",
        "            backend=draft_attention_backend,\n",
    ),
    (
        "        ),\n"
        "    )\n\n"
        "    with set_model_tag(\"dspark_head\"):\n",
        "        ),\n"
        "    )\n"
        "    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)\n\n"
        "    with set_model_tag(\"dspark_head\"):\n",
    ),
    (
        'target_lm_head = getattr(target_model, "lm_head", None)',
        'target_lm_head = getattr(target_language_model, "lm_head", None) or '
        'getattr(target_model, "lm_head", None)',
    ),
    (
        "    target_inner = target_language_model.model\n"
        "    draft_inner = draft_model.model\n",
        "    target_inner = target_language_model.model\n"
        "    draft_inner = draft_model.model\n"
        "    target_vocab_size = vllm_config.model_config.get_vocab_size()\n",
    ),
    (
        "    if target_embed is not None and _should_share(\n"
        "        draft_model, \"has_own_embed_tokens\", draft_embed, target_embed\n"
        "    ):\n",
        "    draft_requires_shared_embed = not getattr(\n"
        "        draft_model, \"has_own_embed_tokens\", True\n"
        "    )\n"
        "    if draft_requires_shared_embed and target_embed is None:\n"
        "        raise ValueError(\"DeepSeek DSpark requires a target embedding table\")\n"
        "    if (\n"
        "        draft_requires_shared_embed\n"
        "        and draft_model_config.get_vocab_size() > target_vocab_size\n"
        "    ):\n"
        "        raise ValueError(\"DeepSeek DSpark embedding vocabulary is incompatible\")\n"
        "    if (\n"
        "        target_embed is not None\n"
        "        and draft_model_config.get_vocab_size() <= target_vocab_size\n"
        "        and _should_share(\n"
        "            draft_model, \"has_own_embed_tokens\", draft_embed, target_embed\n"
        "        )\n"
        "    ):\n",
    ),
    (
        "        draft_inner.embed_tokens = target_embed\n\n"
        "    target_lm_head =",
        "        draft_inner.embed_tokens = target_embed\n"
        "    if (\n"
        "        draft_requires_shared_embed\n"
        "        and getattr(draft_inner, \"embed_tokens\", None) is not target_embed\n"
        "    ):\n"
        "        raise ValueError(\"DeepSeek DSpark target embedding sharing failed\")\n\n"
        "    target_lm_head =",
    ),
    (
        "    draft_lm_head = getattr(draft_model, \"lm_head\", None)\n"
        "    if target_lm_head is not None and _should_share(\n"
        "        draft_model, \"has_own_lm_head\", draft_lm_head, target_lm_head\n"
        "    ):\n",
        "    draft_lm_head = getattr(draft_model, \"lm_head\", None)\n"
        "    draft_output_vocab_size = (\n"
        "        getattr(draft_model_config.hf_config, \"draft_vocab_size\", None)\n"
        "        or draft_model_config.get_vocab_size()\n"
        "    )\n"
        "    draft_requires_shared_head = not getattr(\n"
        "        draft_model, \"has_own_lm_head\", True\n"
        "    )\n"
        "    if draft_requires_shared_head and target_lm_head is None:\n"
        "        raise ValueError(\"DeepSeek DSpark requires a target lm_head\")\n"
        "    if (\n"
        "        draft_requires_shared_head\n"
        "        and draft_output_vocab_size != target_vocab_size\n"
        "    ):\n"
        "        raise ValueError(\"DeepSeek DSpark output vocabulary is incompatible\")\n"
        "    if (\n"
        "        target_lm_head is not None\n"
        "        and draft_output_vocab_size == target_vocab_size\n"
        "        and _should_share(\n"
        "            draft_model, \"has_own_lm_head\", draft_lm_head, target_lm_head\n"
        "        )\n"
        "    ):\n",
    ),
    (
        "        draft_model.lm_head = target_lm_head\n\n"
        "    return draft_model\n",
        "        draft_model.lm_head = target_lm_head\n"
        "    if (\n"
        "        draft_requires_shared_head\n"
        "        and getattr(draft_model, \"lm_head\", None) is not target_lm_head\n"
        "    ):\n"
        "        raise ValueError(\"DeepSeek DSpark target lm_head sharing failed\")\n\n"
        "    return draft_model\n",
    ),
)
EXPECTED_SHA256 = (
    "457c44aec45fe00780748c34288a091bd212d18b40b0d901a8f628640c0a2d24"
)
PATCHED_SHA256 = (
    "c50985fb2737417636c9e55a8decbcff28083909a9aa4d97c0129e70b7c390b3"
)


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()

    if digest == EXPECTED_SHA256:
        patched = source
        for broken, fixed in REPLACEMENTS:
            if patched.count(broken.encode()) != 1:
                print("expected one DSpark patch anchor", file=sys.stderr)
                return 1
            patched = patched.replace(broken.encode(), fixed.encode(), 1)
        if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
            print(
                "patched DSpark loader hash did not match the pinned result",
                file=sys.stderr,
            )
            return 1
        path.write_bytes(patched)
        print(f"patched DSpark DeepSeek compatibility in {path}")
        return 0
    if digest == PATCHED_SHA256:
        print(f"DSpark DeepSeek compatibility already patched in {path}")
        return 0

    print(
        "pinned DSpark loader hash mismatch "
        f"(expected={EXPECTED_SHA256}, actual={digest}) in {path}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
