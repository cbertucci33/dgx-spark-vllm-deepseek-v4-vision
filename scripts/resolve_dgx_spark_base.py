#!/usr/bin/env python3
"""Resolve the pinned Anemll DGX Spark base image without embedding its legacy name."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PINS_PATH = ROOT / "SOURCE_PINS.json"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def build_reference(pins: dict, repository: dict) -> str:
    runtime = pins["dgx_spark_runtime"]
    expected_id = runtime["source_repository_id"]
    if repository.get("id") != expected_id:
        raise ValueError("GitHub repository ID does not match SOURCE_PINS.json")

    owner = repository.get("owner", {}).get("login", "").lower()
    name = repository.get("name", "").lower()
    tag = runtime["registry_tag"]
    digest = runtime["registry_digest"]
    if not SAFE_COMPONENT.fullmatch(owner):
        raise ValueError("unsafe registry owner returned by GitHub")
    if not SAFE_COMPONENT.fullmatch(name):
        raise ValueError("unsafe registry package name returned by GitHub")
    if not SAFE_COMPONENT.fullmatch(tag):
        raise ValueError("unsafe registry tag in SOURCE_PINS.json")
    if not DIGEST.fullmatch(digest):
        raise ValueError("invalid registry digest in SOURCE_PINS.json")
    return f"ghcr.io/{owner}/{name}:{tag}@{digest}"


def resolve() -> str:
    pins = json.loads(PINS_PATH.read_text())
    source_api = pins["dgx_spark_runtime"]["source_api"]
    request = urllib.request.Request(
        source_api,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "dgx-spark-deployment"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        repository = json.load(response)
    return build_reference(pins, repository)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the immutable base-image reference for the DGX Spark deployment."
    )
    parser.parse_args()
    print(resolve())


if __name__ == "__main__":
    main()
