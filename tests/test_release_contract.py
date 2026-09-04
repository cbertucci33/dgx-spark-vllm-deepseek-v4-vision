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
    qualified = pins["derived_image"]
    base = pins["dgx_spark_runtime"]
    plugin_artifact = pins["plugin_artifact"]
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert base["source_repository_id"] == 1301198905
    assert base["base_image_reference_env"] == "DSPARK_VLLM_BASE_IMAGE"
    assert base["registry_tag"] == "0.1.1"
    assert set(qualified) == {"local_tag"}
    assert plugin_artifact["canonical_build_platform"] == "linux"
    assert re.fullmatch(r"[0-9a-f]{64}", plugin_artifact["sha256"])
    assert plugin_artifact["size"] > 0
    patcher = (ROOT / "runtime-patches/patch_dspark_loader.py").read_text()
    assert base["dspark_loader_sha256"] in patcher
    assert base["dspark_loader_patched_sha256"] in patcher
    prefix_patcher = (
        ROOT / "runtime-patches/patch_dspark_prefix_cache.py"
    ).read_text()
    for key in (
        "kv_cache_manager_sha256",
        "kv_cache_manager_patched_sha256",
        "scheduler_sha256",
        "scheduler_patched_sha256",
    ):
        assert base[key] in prefix_patcher
    assert "ARG DSPARK_VLLM_BASE_IMAGE" in dockerfile
    assert "FROM ${DSPARK_VLLM_BASE_IMAGE}" in dockerfile
    assert "import vllm" not in dockerfile
    assert 'import sysconfig; print(sysconfig.get_paths()["purelib"])' in dockerfile
    assert "__pycache__/utils.cpython-312.pyc" in dockerfile
    readme = (ROOT / "README.md").read_text()
    assert tag in readme
    build_helper = (ROOT / "scripts/build-image.sh").read_text()
    assert tag in build_helper
    assert "--build-arg DSPARK_VLLM_BASE_IMAGE=" in build_helper
    for name in ("head.example.env", "worker.example.env"):
        env = parse_env(ROOT / "deployments/anemll-vision/config" / name)
        assert env["DSPARK_VLLM_IMAGE"] == tag
        assert env["DSPARK_VLLM_IMAGE_ID"] == ""

    launcher = (ROOT / "deployments/anemll-vision/start-node.sh").read_text()
    assert "DSPARK_VLLM_IMAGE_ID" in launcher
    assert not re.search(r"sha256:[0-9a-f]{64}", launcher)

    cluster_launcher = (
        ROOT / "deployments/anemll-vision/start-cluster.sh"
    ).read_text()
    verification = cluster_launcher.index("verify_cluster_image_consistency")
    worker_start = cluster_launcher.index("Starting vision TP rank 1")
    assert verification < worker_start
    assert "docker image inspect --format '{{.Id}}'" in cluster_launcher
    assert "Docker image mismatch" in cluster_launcher


def test_only_supported_v2_launcher_is_present() -> None:
    legacy_paths = (
        ".env.vision.example",
        "docker-compose.vision.yml",
        "scripts/download-assets.sh",
        "scripts/preflight-adapter.sh",
        "scripts/smoke-vision.sh",
        "scripts/start-vision.sh",
        "scripts/status-vision.sh",
        "scripts/stop-vision.sh",
    )
    for path in legacy_paths:
        assert not (ROOT / path).exists(), f"obsolete 1.0 launcher restored: {path}"

    deployment = ROOT / "deployments/anemll-vision"
    for path in ("docker-compose.yml", "start-cluster.sh", "start-node.sh", "stop-cluster.sh"):
        assert (deployment / path).is_file(), f"missing supported 2.0 path: {path}"


def test_compose_does_not_advertise_unimplemented_dspark_controls() -> None:
    unsupported = (
        "VLLM_DSPARK_CONFIDENCE_THRESHOLD",
        "VLLM_DSPARK_CONFIDENCE_SCHEDULER",
        "VLLM_DSPARK_LOCAL_ARGMAX",
        "VLLM_DSPARK_REPLICATE_MARKOV_W1",
        "VLLM_DSPARK_FUSED_MARKOV_ARGMAX",
        "VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK",
        "VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT",
        "VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP",
        "VLLM_DSV4_B12X_COMPRESSED_MLA",
        "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE",
        "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT",
    )
    for name in ("docker-compose.yml", "docker-compose.nospec.yml"):
        compose = (ROOT / "deployments/anemll-vision" / name).read_text()
        for variable in unsupported:
            assert variable not in compose, f"{name} exposes unused {variable}"


def test_speculative_compose_uses_explicit_safe_capture_limit() -> None:
    compose = (
        ROOT / "deployments/anemll-vision/docker-compose.yml"
    ).read_text()

    assert "CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-12};" in compose
    assert "MAX_NUM_SEQS:-6} *" not in compose
    assert (
        'COMPILATION_CONFIG="{\\"cudagraph_capture_sizes\\":'
        '[1,2,4,8,$${CAPTURE_SIZE}]}";'
    ) in compose
    assert "--max-cudagraph-capture-size $${CAPTURE_SIZE}" in compose
    assert '--compilation-config "$${COMPILATION_CONFIG}"' in compose
    assert 'KV_CACHE_MEMORY_BYTES: "${KV_CACHE_MEMORY_BYTES:-}"' in compose
    assert (
        'CACHE_ARGS="--kv-cache-memory-bytes $${KV_CACHE_MEMORY_BYTES} '
        '--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.80}"' in compose
    )
    assert 'CACHE_ARGS="--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.80}"' in compose


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
