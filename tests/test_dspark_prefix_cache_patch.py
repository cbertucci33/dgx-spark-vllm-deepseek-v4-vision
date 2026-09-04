from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCHER = ROOT / "runtime-patches" / "patch_dspark_prefix_cache.py"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("dspark_prefix_patch", PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DSparkPrefixCachePatchTests(unittest.TestCase):
    def test_prefix_hit_cap_preserves_the_draft_sliding_window(self) -> None:
        patcher = _load_patcher()

        self.assertEqual(patcher.capped_cache_hit_length(1000, None), 999)
        self.assertEqual(patcher.capped_cache_hit_length(1000, 128), 871)
        self.assertEqual(patcher.capped_cache_hit_length(64, 128), 0)

    def test_invalid_dspark_windows_fail_explicitly(self) -> None:
        patcher = _load_patcher()

        for value in (True, False, 0, -1, 1.5, "128", [128], object()):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    patcher.capped_cache_hit_length(1000, value)

    def test_kv_manager_patch_wires_and_applies_the_window_cap(self) -> None:
        patcher = _load_patcher()
        source = "\n".join(broken for broken, _ in patcher.KV_REPLACEMENTS)
        patched = patcher.apply_replacements(source.encode(), patcher.KV_REPLACEMENTS)
        text = patched.decode()

        self.assertIn("dspark_window_size: int | None = None", text)
        self.assertIn("self.dspark_window_size = dspark_window_size", text)
        self.assertIn("request.num_tokens - 1 - self.dspark_window_size", text)

    def test_scheduler_patch_reads_window_only_for_dspark(self) -> None:
        patcher = _load_patcher()
        source = "\n".join(broken for broken, _ in patcher.SCHED_REPLACEMENTS)
        patched = patcher.apply_replacements(source.encode(), patcher.SCHED_REPLACEMENTS)
        text = patched.decode()

        self.assertIn("speculative_config.use_dspark()", text)
        self.assertIn('getattr(hf_cfg, "sliding_window", None)', text)
        self.assertIn("not isinstance(window, int)", text)
        self.assertIn("isinstance(window, bool)", text)
        self.assertIn("raise ValueError", text)
        self.assertIn("dspark_window_size=dspark_window_size", text)

    def test_file_patching_fails_closed_on_unknown_sources(self) -> None:
        patcher = _load_patcher()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "unknown.py"
            path.write_text("unknown source\n")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                patcher.patch_file(
                    path,
                    expected_sha256="0" * 64,
                    patched_sha256="1" * 64,
                    replacements=patcher.KV_REPLACEMENTS,
                )

    def test_dockerfile_applies_both_prefix_cache_runtime_patches(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()

        self.assertIn("patch_dspark_prefix_cache.py", dockerfile)
        self.assertIn("vllm/v1/core/kv_cache_manager.py", dockerfile)
        self.assertIn("vllm/v1/core/sched/scheduler.py", dockerfile)
        self.assertNotIn("import vllm", dockerfile)


if __name__ == "__main__":
    unittest.main()
