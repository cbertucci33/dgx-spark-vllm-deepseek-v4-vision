import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSEMBLER = REPO_ROOT / "scripts" / "assemble_hf_model.py"


class AssembleHfModelTests(unittest.TestCase):
    def _minimal_fixture(self, root: pathlib.Path):
        source = root / "source"
        assets = root / "assets"
        runtime = root / "runtime"
        destination = root / "output"
        source.mkdir()
        (assets / "tower").mkdir(parents=True)
        (assets / "adapter").mkdir(parents=True)
        runtime.mkdir()
        (source / "config.json").write_text(
            json.dumps({"architectures": ["DeepseekV4ForCausalLM"]})
        )
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.a": "model.safetensors"}})
        )
        (source / "model.safetensors").write_bytes(b"weights")
        tower = assets / "tower" / "deepencoder_v2_tower.safetensors"
        adapter = assets / "adapter" / "merged-004800-5af0c5.pt"
        tower.write_bytes(b"tower")
        adapter.write_bytes(b"adapter")
        wheel = runtime / "plugin.whl"
        wheel.write_bytes(b"wheel")
        source_pins = root / "SOURCE_PINS.json"
        weight_manifest = root / "LANGUAGE_WEIGHTS.sha256"
        weight_manifest.write_text(
            "".join(
                f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n"
                for name in ("model.safetensors", "model.safetensors.index.json")
            )
        )
        source_pins.write_text(
            json.dumps(
                {
                    "language_model": {
                        "repository": "example/source",
                        "revision": "0" * 40,
                        "weight_index_sha256": hashlib.sha256(
                            (source / "model.safetensors.index.json").read_bytes()
                        ).hexdigest(),
                        "weight_manifest_path": weight_manifest.name,
                        "weight_manifest_sha256": hashlib.sha256(
                            weight_manifest.read_bytes()
                        ).hexdigest(),
                    },
                    "plugin_artifact": {
                        "filename": wheel.name,
                        "size": wheel.stat().st_size,
                        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    },
                    "vision_assets": {
                        "files": {
                            "vision/tower/deepencoder_v2_tower.safetensors": {
                                "size": tower.stat().st_size,
                                "sha256": hashlib.sha256(tower.read_bytes()).hexdigest(),
                            },
                            "vision/adapter/merged-004800-5af0c5.pt": {
                                "size": adapter.stat().st_size,
                                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                            },
                        }
                    }
                }
            )
        )
        return source, assets, runtime, destination, source_pins

    def test_rejects_symlink_in_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
            secret = root / "secret.txt"
            secret.write_text("must-not-be-published")
            try:
                (source / "leak.txt").symlink_to(secret)
            except OSError as exc:
                if getattr(exc, "winerror", None) == 1314:
                    self.skipTest("Windows symlink privilege is unavailable")
                raise

            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model",
                    str(source),
                    "--vision-assets",
                    str(assets),
                    "--runtime-artifacts",
                    str(runtime),
                    "--destination",
                    str(destination),
                    "--source-pins",
                    str(source_pins),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("symlink", result.stderr.lower())
            self.assertFalse(destination.exists())

    def test_rejects_traversal_in_vision_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            pins = json.loads(source_pins.read_text())
            pins["vision_assets"]["files"] = {
                "vision/../outside.bin": {
                    "size": outside.stat().st_size,
                    "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                }
            }
            source_pins.write_text(json.dumps(pins))

            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model",
                    str(source),
                    "--vision-assets",
                    str(assets),
                    "--runtime-artifacts",
                    str(runtime),
                    "--destination",
                    str(destination),
                    "--source-pins",
                    str(source_pins),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe path", result.stderr.lower())
            self.assertFalse(destination.exists())

    def test_rejects_traversal_in_weight_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
            outside = root / "outside.safetensors"
            outside.write_bytes(b"outside")
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.a": "../outside.safetensors"}})
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model",
                    str(source),
                    "--vision-assets",
                    str(assets),
                    "--runtime-artifacts",
                    str(runtime),
                    "--destination",
                    str(destination),
                    "--source-pins",
                    str(source_pins),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe path", result.stderr.lower())
            self.assertFalse(destination.exists())

    def test_rejects_destination_inside_input_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, _, source_pins = self._minimal_fixture(root)
            destination = source / "nested-output"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model",
                    str(source),
                    "--vision-assets",
                    str(assets),
                    "--runtime-artifacts",
                    str(runtime),
                    "--destination",
                    str(destination),
                    "--source-pins",
                    str(source_pins),
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("overlaps input tree", result.stderr.lower())
            self.assertFalse(destination.exists())

    def test_rejects_runtime_wheel_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
            pins = json.loads(source_pins.read_text())
            pins["plugin_artifact"]["sha256"] = "0" * 64
            source_pins.write_text(json.dumps(pins))

            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model",
                    str(source),
                    "--vision-assets",
                    str(assets),
                    "--runtime-artifacts",
                    str(runtime),
                    "--destination",
                    str(destination),
                    "--source-pins",
                    str(source_pins),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("runtime wheel sha256 mismatch", result.stderr.lower())
            self.assertFalse(destination.exists())

    def test_rejects_language_weight_index_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
            pins = json.loads(source_pins.read_text())
            pins["language_model"]["weight_index_sha256"] = "0" * 64
            source_pins.write_text(json.dumps(pins))

            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model", str(source),
                    "--vision-assets", str(assets),
                    "--runtime-artifacts", str(runtime),
                    "--destination", str(destination),
                    "--source-pins", str(source_pins),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("weight-index sha256 mismatch", result.stderr.lower())
            self.assertFalse(destination.exists())

    def test_rejects_language_weight_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
            (source / "model.safetensors").write_bytes(b"substituted weights")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model", str(source),
                    "--vision-assets", str(assets),
                    "--runtime-artifacts", str(runtime),
                    "--destination", str(destination),
                    "--source-pins", str(source_pins),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("language weight sha256 mismatch", result.stderr.lower())
            self.assertFalse(destination.exists())

    def test_rejects_unpinned_extra_input_files(self) -> None:
        for tree_name, relative in (("assets", "extra.bin"), ("runtime", "extra.txt")):
            with self.subTest(tree=tree_name), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
                target = assets if tree_name == "assets" else runtime
                (target / relative).write_bytes(b"unexpected")

                result = subprocess.run(
                    [
                        sys.executable,
                        str(ASSEMBLER),
                        "--source-model", str(source),
                        "--vision-assets", str(assets),
                        "--runtime-artifacts", str(runtime),
                        "--destination", str(destination),
                        "--source-pins", str(source_pins),
                    ],
                    capture_output=True,
                    text=True,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("file set does not match pins", result.stderr.lower())
                self.assertFalse(destination.exists())

    def test_builds_independent_uploadable_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            assets = root / "assets"
            runtime = root / "runtime"
            destination = root / "output"
            source.mkdir()
            (assets / "tower").mkdir(parents=True)
            (assets / "adapter").mkdir(parents=True)
            runtime.mkdir()

            config = {
                "architectures": ["DeepseekV4ForCausalLM"],
                "hidden_size": 4096,
                "vocab_size": 129280,
            }
            index = {
                "metadata": {"total_size": 9},
                "weight_map": {
                    "model.a": "model-00001-of-00001.safetensors",
                    "model.edited": "model-overlay-00001-of-00001.safetensors",
                },
            }
            (source / "config.json").write_text(json.dumps(config))
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            (source / "model-00001-of-00001.safetensors").write_bytes(b"base")
            (source / "model-overlay-00001-of-00001.safetensors").write_bytes(
                b"overlay"
            )
            (source / "tokenizer.json").write_text("{}")
            tower = assets / "tower" / "deepencoder_v2_tower.safetensors"
            adapter = assets / "adapter" / "merged-004800-5af0c5.pt"
            tower.write_bytes(b"tower")
            adapter.write_bytes(b"adapter")
            (runtime / "wheels").mkdir()
            (runtime / "wheels" / "plugin.whl").write_bytes(b"wheel")
            source_pins = root / "SOURCE_PINS.json"
            weight_manifest = root / "LANGUAGE_WEIGHTS.sha256"
            weight_manifest.write_text(
                "".join(
                    f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n"
                    for name in (
                        "model-00001-of-00001.safetensors",
                        "model-overlay-00001-of-00001.safetensors",
                        "model.safetensors.index.json",
                    )
                )
            )
            source_pins.write_text(
                json.dumps(
                    {
                        "language_model": {
                            "repository": "example/source",
                            "revision": "0" * 40,
                            "weight_index_sha256": hashlib.sha256(
                                (source / "model.safetensors.index.json").read_bytes()
                            ).hexdigest(),
                            "weight_manifest_path": weight_manifest.name,
                            "weight_manifest_sha256": hashlib.sha256(
                                weight_manifest.read_bytes()
                            ).hexdigest(),
                        },
                        "plugin_artifact": {
                            "path": "wheels/plugin.whl",
                            "size": len(b"wheel"),
                            "sha256": hashlib.sha256(b"wheel").hexdigest(),
                        },
                        "vision_assets": {
                            "files": {
                                "vision/tower/deepencoder_v2_tower.safetensors": {
                                    "size": tower.stat().st_size,
                                    "sha256": hashlib.sha256(tower.read_bytes()).hexdigest(),
                                },
                                "vision/adapter/merged-004800-5af0c5.pt": {
                                    "size": adapter.stat().st_size,
                                    "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                                },
                            }
                        }
                    }
                )
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model",
                    str(source),
                    "--vision-assets",
                    str(assets),
                    "--runtime-artifacts",
                    str(runtime),
                    "--destination",
                    str(destination),
                    "--source-pins",
                    str(source_pins),
                ],
                check=True,
            )

            self.assertTrue(destination.is_dir())
            self.assertFalse(any(path.is_symlink() for path in destination.rglob("*")))
            output_config = json.loads((destination / "config.json").read_text())
            self.assertEqual(
                output_config["architectures"], ["DeepseekV4VisionForCausalLM"]
            )
            self.assertEqual(output_config["hidden_size"], 4096)
            self.assertEqual(
                output_config["vision_config"]["image_token_id"], 129279
            )
            self.assertEqual(
                json.loads((destination / "model.safetensors.index.json").read_text()),
                index,
            )
            self.assertEqual(
                (destination / "vision/tower/deepencoder_v2_tower.safetensors").read_bytes(),
                b"tower",
            )
            self.assertEqual(
                (destination / "vision/adapter/merged-004800-5af0c5.pt").read_bytes(),
                b"adapter",
            )
            self.assertEqual(
                (destination / "runtime/wheels/plugin.whl").read_bytes(), b"wheel"
            )

            manifest_lines = (destination / "MANIFEST.sha256").read_text().splitlines()
            manifest = {
                relative: digest
                for digest, relative in (line.split("  ", 1) for line in manifest_lines)
            }
            for relative in (
                "config.json",
                "model-00001-of-00001.safetensors",
                "model-overlay-00001-of-00001.safetensors",
                "LANGUAGE_WEIGHTS.sha256",
                "vision/tower/deepencoder_v2_tower.safetensors",
                "vision/adapter/merged-004800-5af0c5.pt",
                "runtime/wheels/plugin.whl",
            ):
                payload = (destination / relative).read_bytes()
                self.assertEqual(manifest[relative], hashlib.sha256(payload).hexdigest())

    def test_release_metadata_replaces_inherited_model_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, assets, runtime, destination, source_pins = self._minimal_fixture(root)
            (source / "README.md").write_text("stale language-only card")
            metadata = root / "release-metadata"
            metadata.mkdir()
            payloads = {
                "LICENSE": "apache terms\n",
                "README.md": "vision model card\n",
                "ASSEMBLY.md": "composition method\n",
                "THIRD_PARTY_NOTICES.md": "upstream terms\n",
                "VALIDATION.md": "bounded validation\n",
                "LICENSES/Apache-2.0.txt": "apache terms\n",
                "LICENSES/MIT-DeepSeek.txt": "deepseek mit terms\n",
                "LICENSES/MIT-FlyCockpit.txt": "mit terms\n",
            }
            for name, payload in payloads.items():
                path = metadata / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload)

            subprocess.run(
                [
                    sys.executable,
                    str(ASSEMBLER),
                    "--source-model", str(source),
                    "--vision-assets", str(assets),
                    "--runtime-artifacts", str(runtime),
                    "--release-metadata", str(metadata),
                    "--destination", str(destination),
                    "--source-pins", str(source_pins),
                ],
                check=True,
            )

            for name, payload in payloads.items():
                self.assertEqual((destination / name).read_text(), payload)
            manifest = (destination / "MANIFEST.sha256").read_text()
            for name in payloads:
                self.assertIn(f"  {name}\n", manifest)


if __name__ == "__main__":
    unittest.main()
