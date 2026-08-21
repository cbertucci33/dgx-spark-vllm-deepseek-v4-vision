#!/usr/bin/env python3
"""Assemble an independent Hugging Face model repository atomically."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
from typing import Dict, Iterable


VISION_CONFIG = {
    "architecture": "DeepEncoderV2",
    "image_token": "<｜image｜>",
    "image_token_id": 129279,
    "image_size": 1024,
    "encoder_dim": 896,
    "hidden_size": 4096,
    "tokens_per_view": 256,
    "tiles": 2,
    "tile_threshold_px": 1536,
    "global_view_position": "head",
    "view_order": "global_then_row_major_crops_then_separator",
    "tower_path": "vision/tower/deepencoder_v2_tower.safetensors",
    "adapter_path": "vision/adapter/merged-004800-5af0c5.pt",
}

PREPROCESSOR_CONFIG = {
    "do_convert_rgb": True,
    "do_resize": True,
    "size": {"height": 1024, "width": 1024},
    "resample": "bicubic",
    "do_rescale": True,
    "rescale_factor": 1.0 / 255.0,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "custom_runtime_required": "dsv4-vision-vllm",
}

RELEASE_METADATA_FILES = (
    "LICENSE",
    "README.md",
    "ASSEMBLY.md",
    "THIRD_PARTY_NOTICES.md",
    "VALIDATION.md",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/MIT-DeepSeek.txt",
    "LICENSES/MIT-FlyCockpit.txt",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def reject_symlinks(root: pathlib.Path, label: str) -> None:
    if root.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {root}")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = pathlib.Path(directory)
        for name in [*dirnames, *filenames]:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"{label} contains symlink: {path}")


def safe_relative_path(value: object, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"unsafe path for {label}: {value!r}")
    if "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"unsafe path for {label}: {value!r}")
    raw_parts = value.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"unsafe path for {label}: {value!r}")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute():
        raise ValueError(f"unsafe path for {label}: {value!r}")
    return relative


def paths_overlap(first: pathlib.Path, second: pathlib.Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_copy_topology(
    source: pathlib.Path,
    assets: pathlib.Path,
    runtime: pathlib.Path,
    release_metadata: pathlib.Path | None,
    destination: pathlib.Path,
    building: pathlib.Path,
) -> None:
    inputs = [
        ("source model", source),
        ("vision assets", assets),
        ("runtime artifacts", runtime),
    ]
    if release_metadata is not None:
        inputs.append(("release metadata", release_metadata))
    for index, (left_label, left) in enumerate(inputs):
        for right_label, right in inputs[index + 1 :]:
            if paths_overlap(left, right):
                raise ValueError(
                    f"input trees overlap: {left_label}={left} {right_label}={right}"
                )
        for output_label, output in (("destination", destination), ("staging destination", building)):
            if paths_overlap(left, output):
                raise ValueError(
                    f"{output_label} overlaps input tree: {output} {left_label}={left}"
                )


def validate_weight_index(source: pathlib.Path) -> Dict[str, object]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"missing weight index: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("weight index has no non-empty weight_map")
    referenced = sorted(
        {
            safe_relative_path(name, "weight index entry").as_posix()
            for name in weight_map.values()
        }
    )
    missing = [name for name in referenced if not (source / name).is_file()]
    if missing:
        raise ValueError(f"weight index references missing files: {missing}")
    return index


def parse_sha256_manifest(path: pathlib.Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        digest, separator, relative_text = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"invalid sha256 manifest line {line_number}: {line!r}")
        relative = safe_relative_path(relative_text, "sha256 manifest entry").as_posix()
        if relative in entries:
            raise ValueError(f"duplicate sha256 manifest entry: {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError("sha256 manifest is empty")
    return entries


def validate_language_model_source(
    source: pathlib.Path,
    source_pins: Dict[str, object],
    weight_manifest_path: pathlib.Path,
) -> Dict[str, object]:
    language_model = source_pins.get("language_model")
    if not isinstance(language_model, dict):
        raise ValueError("SOURCE_PINS.json has no language_model object")
    for field in ("repository", "revision"):
        value = language_model.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"language_model has invalid {field}")
    expected_index_sha = language_model.get("weight_index_sha256")
    if not isinstance(expected_index_sha, str) or len(expected_index_sha) != 64:
        raise ValueError("language_model has invalid weight_index_sha256")
    index_path = source / "model.safetensors.index.json"
    index = validate_weight_index(source)
    actual_index_sha = sha256_file(index_path)
    if actual_index_sha != expected_index_sha:
        raise ValueError(
            "language-model weight-index sha256 mismatch: "
            f"{actual_index_sha} != {expected_index_sha}"
        )

    expected_manifest_sha = language_model.get("weight_manifest_sha256")
    if not isinstance(expected_manifest_sha, str) or len(expected_manifest_sha) != 64:
        raise ValueError("language_model has invalid weight_manifest_sha256")
    if not weight_manifest_path.is_file():
        raise ValueError(f"missing language weight manifest: {weight_manifest_path}")
    actual_manifest_sha = sha256_file(weight_manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            "language weight-manifest sha256 mismatch: "
            f"{actual_manifest_sha} != {expected_manifest_sha}"
        )
    entries = parse_sha256_manifest(weight_manifest_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("weight index has invalid weight_map")
    expected_files = {
        safe_relative_path(name, "weight index entry").as_posix()
        for name in weight_map.values()
    }
    expected_files.add("model.safetensors.index.json")
    if set(entries) != expected_files:
        raise ValueError(
            "language weight manifest does not match weight index: "
            f"unexpected={sorted(set(entries) - expected_files)} "
            f"missing={sorted(expected_files - set(entries))}"
        )
    for relative, expected_sha in entries.items():
        actual_sha = sha256_file(source / relative)
        if actual_sha != expected_sha:
            raise ValueError(
                f"language weight sha256 mismatch for {relative}: "
                f"{actual_sha} != {expected_sha}"
            )
    return index


def validate_vision_assets(
    assets: pathlib.Path, source_pins: Dict[str, object]
) -> None:
    vision_assets = source_pins.get("vision_assets")
    if not isinstance(vision_assets, dict):
        raise ValueError("SOURCE_PINS.json has no vision_assets object")
    files = vision_assets.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SOURCE_PINS.json has no vision asset file pins")
    expected_files = {
        safe_relative_path(logical_path, "vision pin").relative_to("vision").as_posix()
        for logical_path in files
    }
    actual_files = {
        path.relative_to(assets).as_posix() for path in iter_files(assets)
    }
    if actual_files != expected_files:
        raise ValueError(
            "vision asset file set does not match pins: "
            f"unexpected={sorted(actual_files - expected_files)} "
            f"missing={sorted(expected_files - actual_files)}"
        )
    for logical_path, expected in files.items():
        relative = safe_relative_path(logical_path, "vision pin")
        if len(relative.parts) < 2 or relative.parts[0] != "vision":
            raise ValueError(f"vision pin must start with vision/: {logical_path}")
        path = assets.joinpath(*relative.parts[1:])
        if not path.is_file():
            raise ValueError(f"missing pinned vision asset: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected["size"]:
            raise ValueError(
                f"size mismatch for {logical_path}: {actual_size} != {expected['size']}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != expected["sha256"]:
            raise ValueError(
                f"sha256 mismatch for {logical_path}: {actual_sha} != "
                f"{expected['sha256']}"
            )


def validate_runtime_artifact(
    runtime: pathlib.Path, source_pins: Dict[str, object]
) -> None:
    artifact = source_pins.get("plugin_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("SOURCE_PINS.json has no plugin_artifact object")
    relative = safe_relative_path(
        artifact.get("path", artifact.get("filename")), "runtime wheel"
    )
    if relative.suffix != ".whl":
        raise ValueError(f"runtime artifact must be a wheel path: {relative}")
    expected_sha = artifact.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("runtime wheel has invalid pinned sha256")
    expected_size = artifact.get("size")
    if not isinstance(expected_size, int) or expected_size < 1:
        raise ValueError("runtime wheel has invalid pinned size")
    actual_files = {
        path.relative_to(runtime).as_posix() for path in iter_files(runtime)
    }
    expected_files = {relative.as_posix()}
    if actual_files != expected_files:
        raise ValueError(
            "runtime artifact file set does not match pins: "
            f"unexpected={sorted(actual_files - expected_files)} "
            f"missing={sorted(expected_files - actual_files)}"
        )
    wheel = runtime / relative.as_posix()
    if not wheel.is_file():
        raise ValueError(f"missing pinned runtime wheel: {wheel}")
    if wheel.stat().st_size != expected_size:
        raise ValueError(
            f"runtime wheel size mismatch: {wheel.stat().st_size} != {expected_size}"
        )
    actual_sha = sha256_file(wheel)
    if actual_sha != expected_sha:
        raise ValueError(
            f"runtime wheel sha256 mismatch: {actual_sha} != {expected_sha}"
        )


def write_manifest(root: pathlib.Path) -> None:
    lines = []
    for path in iter_files(root):
        if path.name == "MANIFEST.sha256":
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}\n")
    (root / "MANIFEST.sha256").write_text("".join(lines))


def assemble(
    source: pathlib.Path,
    assets: pathlib.Path,
    runtime: pathlib.Path,
    release_metadata: pathlib.Path | None,
    destination: pathlib.Path,
    source_pins_path: pathlib.Path,
) -> None:
    inputs = [
        ("source model", source),
        ("vision assets", assets),
        ("runtime artifacts", runtime),
    ]
    if release_metadata is not None:
        inputs.append(("release metadata", release_metadata))
    for label, path in inputs:
        if not path.is_dir():
            raise ValueError(f"{label} directory does not exist: {path}")
    if not source_pins_path.is_file():
        raise ValueError(f"source pins do not exist: {source_pins_path}")
    for label, path in inputs:
        reject_symlinks(path, label)
    if release_metadata is not None:
        missing_metadata = [
            name
            for name in RELEASE_METADATA_FILES
            if not (release_metadata / name).is_file()
        ]
        if missing_metadata:
            raise ValueError(f"release metadata is missing files: {missing_metadata}")
    reject_symlinks(source_pins_path, "source pins")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")

    building = destination.with_name(destination.name + ".building")
    if building.exists():
        raise ValueError(f"staging destination already exists: {building}")
    validate_copy_topology(
        source, assets, runtime, release_metadata, destination, building
    )

    source_pins = json.loads(source_pins_path.read_text())
    language_model = source_pins.get("language_model")
    if not isinstance(language_model, dict):
        raise ValueError("SOURCE_PINS.json has no language_model object")
    weight_manifest_relative = safe_relative_path(
        language_model.get("weight_manifest_path"), "language weight manifest"
    )
    weight_manifest_path = source_pins_path.parent.joinpath(*weight_manifest_relative.parts)
    reject_symlinks(weight_manifest_path, "language weight manifest")
    index_before = validate_language_model_source(
        source, source_pins, weight_manifest_path
    )
    index_before_sha = sha256_file(source / "model.safetensors.index.json")
    validate_vision_assets(assets, source_pins)
    validate_runtime_artifact(runtime, source_pins)

    try:
        shutil.copytree(
            source,
            building,
            symlinks=False,
            ignore=shutil.ignore_patterns(".git", ".cache", "*.lock"),
        )

        config_path = building / "config.json"
        config = json.loads(config_path.read_text())
        config["architectures"] = ["DeepseekV4VisionForCausalLM"]
        config["vision_config"] = VISION_CONFIG
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

        vision_root = building / "vision"
        shutil.copytree(assets, vision_root)
        (vision_root / "vision_config.json").write_text(
            json.dumps(VISION_CONFIG, indent=2, sort_keys=True) + "\n"
        )
        (vision_root / "preprocessor_config.json").write_text(
            json.dumps(PREPROCESSOR_CONFIG, indent=2, sort_keys=True) + "\n"
        )

        shutil.copytree(runtime, building / "runtime")
        shutil.copy2(source_pins_path, building / "SOURCE_PINS.json")
        destination_weight_manifest = building.joinpath(*weight_manifest_relative.parts)
        destination_weight_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weight_manifest_path, destination_weight_manifest)
        if release_metadata is not None:
            for name in RELEASE_METADATA_FILES:
                destination_path = building / name
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(release_metadata / name, destination_path)

        index_after_path = building / "model.safetensors.index.json"
        index_after = json.loads(index_after_path.read_text())
        if index_after != index_before or sha256_file(index_after_path) != index_before_sha:
            raise ValueError("model.safetensors.index.json changed during assembly")

        symlinks = [path for path in building.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError(f"assembled repository contains symlinks: {symlinks[:5]}")

        validate_weight_index(building)
        write_manifest(building)
        os.replace(str(building), str(destination))
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise

    file_count = sum(1 for _ in iter_files(destination))
    total_bytes = sum(path.stat().st_size for path in iter_files(destination))
    print(f"ASSEMBLY_COMPLETE destination={destination}")
    print(f"files={file_count} bytes={total_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, type=pathlib.Path)
    parser.add_argument("--vision-assets", required=True, type=pathlib.Path)
    parser.add_argument("--runtime-artifacts", required=True, type=pathlib.Path)
    parser.add_argument("--release-metadata", type=pathlib.Path)
    parser.add_argument("--destination", required=True, type=pathlib.Path)
    parser.add_argument("--source-pins", required=True, type=pathlib.Path)
    args = parser.parse_args()
    assemble(
        args.source_model.resolve(),
        args.vision_assets.resolve(),
        args.runtime_artifacts.resolve(),
        args.release_metadata.resolve() if args.release_metadata else None,
        args.destination.resolve(),
        args.source_pins.resolve(),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ASSEMBLY_FAILED: {exc}", file=sys.stderr)
        raise
