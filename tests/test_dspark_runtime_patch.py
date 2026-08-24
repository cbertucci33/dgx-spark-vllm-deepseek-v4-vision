import ast
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCHER = ROOT / "runtime-patches" / "patch_dspark_loader.py"
PINNED_LOADER = ROOT / "tests" / "fixtures" / "dspark_utils_pinned.py"


def pinned_loader_bytes() -> bytes:
    """Return the pinned Linux runtime source on every checkout platform."""
    return b"\n".join(PINNED_LOADER.read_bytes().splitlines()) + b"\n"


class DSparkRuntimePatchTests(unittest.TestCase):
    def test_patcher_uses_unwrapped_language_model_for_lm_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            loader.write_bytes(pinned_loader_bytes())

            result = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            patched = loader.read_text()
            self.assertIn(
                'target_lm_head = getattr(target_language_model, "lm_head", None) or '
                'getattr(target_model, "lm_head", None)',
                patched,
            )
            self.assertNotIn(
                'target_lm_head = getattr(target_model, "lm_head", None)\n', patched
            )

    def test_patcher_inherits_target_backend_for_deepseek_v4_dspark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            loader.write_bytes(pinned_loader_bytes())

            result = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            patched = loader.read_text()
            self.assertIn(
                "draft_attention_backend = _resolve_dspark_attention_backend(\n"
                "        draft_model_config,\n"
                "        speculative_config.attention_backend,\n"
                "        vllm_config.attention_config.backend,\n"
                "    )",
                patched,
            )
            self.assertIn("backend=draft_attention_backend", patched)
            self.assertNotIn(
                "backend=speculative_config.attention_backend", patched
            )

    def test_patcher_does_not_inherit_target_backend_for_other_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            loader.write_bytes(pinned_loader_bytes())
            result = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            module = ast.parse(loader.read_text())
            resolver = next(
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_resolve_dspark_attention_backend"
            )
            namespace: dict[str, object] = {}
            exec(compile(ast.Module([resolver], []), str(loader), "exec"), namespace)
            resolve = namespace["_resolve_dspark_attention_backend"]
            deepseek = types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="deepseek_v4")
            )
            qwen = types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="qwen3")
            )

            self.assertEqual(resolve(deepseek, None, "target"), "target")
            self.assertIsNone(resolve(qwen, None, "target"))
            self.assertEqual(resolve(qwen, "explicit", "target"), "explicit")

    def test_patcher_uses_the_draft_models_quantization_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            loader.write_bytes(pinned_loader_bytes())

            result = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            patched = loader.read_text()
            self.assertIn(
                "from vllm.model_executor.models.utils import "
                "get_draft_quant_config",
                patched,
            )
            self.assertIn(
                "draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)",
                patched,
            )

    def test_patcher_guards_shared_weights_by_vocabulary_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            loader.write_bytes(pinned_loader_bytes())

            result = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            patched = loader.read_text()
            self.assertIn(
                "target_vocab_size = vllm_config.model_config.get_vocab_size()",
                patched,
            )
            self.assertIn(
                "draft_model_config.get_vocab_size() <= target_vocab_size",
                patched,
            )
            self.assertIn(
                "draft_output_vocab_size == target_vocab_size",
                patched,
            )
            self.assertIn("draft_requires_shared_embed", patched)
            self.assertIn("draft_requires_shared_head", patched)
            self.assertIn("requires a target embedding table", patched)
            self.assertIn("requires a target lm_head", patched)
            self.assertIn("target embedding sharing failed", patched)
            self.assertIn("target lm_head sharing failed", patched)

    def test_dockerfile_applies_dspark_patch_to_installed_runtime(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()

        self.assertIn("COPY runtime-patches /tmp/dsv4-runtime-patches", dockerfile)
        self.assertIn("patch_dspark_loader.py", dockerfile)
        self.assertIn(
            "vllm.v1.worker.gpu.spec_decode.dspark.utils", dockerfile
        )

    def test_patcher_fails_closed_when_loader_shape_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            original = "def unrelated_loader():\n    return None\n"
            loader.write_text(original)

            result = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pinned DSpark loader hash mismatch", result.stderr)
            self.assertEqual(loader.read_text(), original)

    def test_patcher_fails_closed_when_surrounding_source_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            original = PINNED_LOADER.read_text().replace(
                "# DSpark uses non-causal attention.",
                "# Drift outside the patched assignment.",
            )
            loader.write_text(original)

            result = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pinned DSpark loader hash", result.stderr)
            self.assertEqual(loader.read_text(), original)

    def test_patcher_accepts_only_the_exact_patched_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loader = pathlib.Path(directory) / "utils.py"
            loader.write_bytes(pinned_loader_bytes())
            first = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            second = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)

            drifted = loader.read_text() + "# post-patch drift\n"
            loader.write_text(drifted)
            third = subprocess.run(
                [sys.executable, str(PATCHER), str(loader)],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(third.returncode, 0)
            self.assertEqual(loader.read_text(), drifted)


if __name__ == "__main__":
    unittest.main()
