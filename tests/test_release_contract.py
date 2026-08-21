import importlib.util
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_env(path: pathlib.Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_release_image_contract_is_consistent() -> None:
    pins = json.loads((ROOT / "SOURCE_PINS.json").read_text())
    tag = pins["derived_image"]["local_tag"]
    base = pins["dgx_spark_runtime"]
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert base["source_repository_id"] == 1301198905
    assert base["base_image_reference_env"] == "DSPARK_VLLM_BASE_IMAGE"
    assert base["registry_tag"] == "0.1.1"
    assert "ARG DSPARK_VLLM_BASE_IMAGE" in dockerfile
    assert "FROM ${DSPARK_VLLM_BASE_IMAGE}" in dockerfile
    for name in ("head.example.env", "worker.example.env"):
        env = parse_env(ROOT / "deployments/anemll-vision/config" / name)
        assert env["DSPARK_VLLM_IMAGE"] == tag
        assert env["DSPARK_VLLM_IMAGE_ID"] == ""

    launcher = (ROOT / "deployments/anemll-vision/start-node.sh").read_text()
    assert "DSPARK_VLLM_IMAGE_ID" in launcher
    assert not re.search(r"sha256:[0-9a-f]{64}", launcher)


def test_base_image_resolver_builds_the_pinned_reference() -> None:
    module_path = ROOT / "scripts" / "resolve_dgx_spark_base.py"
    spec = importlib.util.spec_from_file_location("resolve_dgx_spark_base", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pins = json.loads((ROOT / "SOURCE_PINS.json").read_text())
    repository = {
        "id": pins["dgx_spark_runtime"]["source_repository_id"],
        "name": "runtime-package",
        "owner": {"login": "RuntimeOwner"},
    }
    reference = module.build_reference(pins, repository)
    assert reference.startswith("ghcr.io/runtimeowner/runtime-package:0.1.1@sha256:")
    assert reference.endswith(pins["dgx_spark_runtime"]["registry_digest"])
