#!/usr/bin/env python3
"""Patch pinned vLLM prefix caching for DSpark sliding-window replay."""

from __future__ import annotations

import hashlib
import pathlib
import sys


KV_REPLACEMENTS = (
    (
        "        use_eagle: bool = False,\n"
        "        log_stats: bool = False,\n",
        "        use_eagle: bool = False,\n"
        "        dspark_window_size: int | None = None,\n"
        "        log_stats: bool = False,\n",
    ),
    (
        "        self.enable_caching = enable_caching\n"
        "        self.use_eagle = use_eagle\n"
        "        self.log_stats = log_stats\n",
        "        self.enable_caching = enable_caching\n"
        "        self.use_eagle = use_eagle\n"
        "        self.dspark_window_size = dspark_window_size\n"
        "        self.log_stats = log_stats\n",
    ),
    (
        "        max_cache_hit_length = request.num_tokens - 1\n"
        "        computed_blocks, num_new_computed_tokens = (\n",
        "        max_cache_hit_length = request.num_tokens - 1\n"
        "        # Recompute the DSpark sliding window so prefix-cache hits also\n"
        "        # repopulate the draft model's context KV from target states.\n"
        "        if self.dspark_window_size is not None and self.dspark_window_size > 0:\n"
        "            max_cache_hit_length = max(\n"
        "                request.num_tokens - 1 - self.dspark_window_size, 0\n"
        "            )\n"
        "        computed_blocks, num_new_computed_tokens = (\n",
    ),
)

SCHED_REPLACEMENTS = (
    (
        "\n        # Create the KV cache manager.\n"
        "        if hash_block_size is None:\n",
        "\n        dspark_window_size: int | None = None\n"
        "        if speculative_config is not None and speculative_config.use_dspark():\n"
        "            draft_cfg = getattr(speculative_config, \"draft_model_config\", None)\n"
        "            hf_cfg = getattr(draft_cfg, \"hf_config\", None)\n"
        "            window = getattr(hf_cfg, \"sliding_window\", None)\n"
        "            if window is not None:\n"
        "                if (\n"
        "                    not isinstance(window, int)\n"
        "                    or isinstance(window, bool)\n"
        "                    or window <= 0\n"
        "                ):\n"
        "                    raise ValueError(\n"
        "                        \"DSpark sliding_window must be a positive integer\"\n"
        "                    )\n"
        "                dspark_window_size = window\n\n"
        "        # Create the KV cache manager.\n"
        "        if hash_block_size is None:\n",
    ),
    (
        "            use_eagle=self.use_eagle,\n"
        "            log_stats=self.log_stats,\n",
        "            use_eagle=self.use_eagle,\n"
        "            dspark_window_size=dspark_window_size,\n"
        "            log_stats=self.log_stats,\n",
    ),
)

KV_EXPECTED_SHA256 = "be9c50918c8cd01736102de4020e2a9a8650675bcd4eb4227d40d2bede6bb853"
KV_PATCHED_SHA256 = "6c599def2dcdae6222dbb68a897541276b92b209a7ddb4026822b2cf489f1144"
SCHED_EXPECTED_SHA256 = "e25d4c9a95abdbe8e516714ed02574d929ca0d5e8c11c4cc73b84d3a3b905443"
SCHED_PATCHED_SHA256 = "5af11de059edfa7f1a62adbf3cae85b37c5801527b02802885654d54f41655d8"


def capped_cache_hit_length(num_tokens: int, dspark_window_size: object | None) -> int:
    maximum = num_tokens - 1
    if dspark_window_size is not None:
        if (
            not isinstance(dspark_window_size, int)
            or isinstance(dspark_window_size, bool)
            or dspark_window_size <= 0
        ):
            raise ValueError("DSpark sliding_window must be a positive integer")
        maximum = max(num_tokens - 1 - dspark_window_size, 0)
    return maximum


def apply_replacements(source: bytes, replacements: tuple[tuple[str, str], ...]) -> bytes:
    patched = source
    for broken, fixed in replacements:
        old = broken.encode()
        if patched.count(old) != 1:
            raise ValueError("expected exactly one runtime patch anchor")
        patched = patched.replace(old, fixed.encode(), 1)
    return patched


def patch_file(
    path: pathlib.Path,
    *,
    expected_sha256: str,
    patched_sha256: str,
    replacements: tuple[tuple[str, str], ...],
) -> None:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == patched_sha256:
        print(f"DSpark prefix-cache replay already patched in {path}")
        return
    if digest != expected_sha256:
        raise ValueError(
            f"pinned runtime hash mismatch (expected={expected_sha256}, actual={digest}) "
            f"in {path}"
        )
    patched = apply_replacements(source, replacements)
    actual_patched = hashlib.sha256(patched).hexdigest()
    if actual_patched != patched_sha256:
        raise ValueError(
            f"patched runtime hash mismatch (expected={patched_sha256}, "
            f"actual={actual_patched}) in {path}"
        )
    path.write_bytes(patched)
    print(f"patched DSpark prefix-cache replay in {path}")


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} KV_CACHE_MANAGER SCHEDULER", file=sys.stderr)
        return 2
    try:
        patch_file(
            pathlib.Path(sys.argv[1]),
            expected_sha256=KV_EXPECTED_SHA256,
            patched_sha256=KV_PATCHED_SHA256,
            replacements=KV_REPLACEMENTS,
        )
        patch_file(
            pathlib.Path(sys.argv[2]),
            expected_sha256=SCHED_EXPECTED_SHA256,
            patched_sha256=SCHED_PATCHED_SHA256,
            replacements=SCHED_REPLACEMENTS,
        )
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
