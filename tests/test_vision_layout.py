import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "plugin" / "src" / "dsv4_vision_vllm" / "vision_layout.py"


def load_module():
    spec = importlib.util.spec_from_file_location("vision_layout", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_metadata_is_authoritative() -> None:
    module = load_module()
    assert module.resolve_checkpoint_tiles(metadata_tiles=2, override=None) == 2


def test_matching_override_is_accepted() -> None:
    module = load_module()
    assert module.resolve_checkpoint_tiles(metadata_tiles=2, override="2") == 2


def test_mismatched_override_is_rejected() -> None:
    module = load_module()
    try:
        module.resolve_checkpoint_tiles(metadata_tiles=2, override="0")
    except RuntimeError as exc:
        assert "does not match checkpoint" in str(exc)
    else:
        raise AssertionError("mismatched layout override was accepted")


def test_missing_metadata_requires_explicit_override() -> None:
    module = load_module()
    try:
        module.resolve_checkpoint_tiles(metadata_tiles=None, override=None)
    except RuntimeError as exc:
        assert "metadata is unavailable" in str(exc)
    else:
        raise AssertionError("missing metadata silently selected a layout")


def test_explicit_legacy_override_allows_missing_metadata() -> None:
    module = load_module()
    assert module.resolve_checkpoint_tiles(metadata_tiles=None, override="0") == 0


def test_invalid_tile_values_are_rejected() -> None:
    module = load_module()
    for metadata, override in (
        (-1, None),
        (2.5, None),
        (True, None),
        (None, "-1"),
        (None, "2.5"),
        (None, "not-an-int"),
    ):
        try:
            module.resolve_checkpoint_tiles(metadata_tiles=metadata, override=override)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError((metadata, override))


def test_missing_tiles_member_is_unavailable_metadata() -> None:
    module = load_module()

    class MetadataWithoutTiles:
        pass

    assert module.extract_metadata_tiles({}) is None
    assert module.extract_metadata_tiles(MetadataWithoutTiles()) is None


def test_model_routes_checkpoint_metadata_through_fail_closed_resolver() -> None:
    source = (
        ROOT / "plugin" / "src" / "dsv4_vision_vllm" / "model.py"
    ).read_text()
    assert "from dsv4_vision_vllm.vision_layout import (" in source
    assert "resolve_checkpoint_tiles," in source
    assert "tiles = resolve_checkpoint_tiles(" in source
    assert "never fail startup on metadata" not in source


def test_vision_asset_path_is_required(tmp_path: pathlib.Path) -> None:
    module = load_module()
    for value in (None, "", str(tmp_path / "missing.bin")):
        try:
            module.require_vision_asset("tower", value)
        except RuntimeError as exc:
            assert "tower" in str(exc)
        else:
            raise AssertionError(f"invalid tower path was accepted: {value!r}")


def test_existing_vision_asset_path_is_returned(tmp_path: pathlib.Path) -> None:
    module = load_module()
    asset = tmp_path / "tower.safetensors"
    asset.write_bytes(b"fixture")
    assert module.require_vision_asset("tower", str(asset)) == asset


def test_model_requires_both_vision_artifacts() -> None:
    source = (
        ROOT / "plugin" / "src" / "dsv4_vision_vllm" / "model.py"
    ).read_text()
    assert 'require_vision_asset("tower", TOWER_PATH)' in source
    assert 'require_vision_asset("adapter", ADAPTER_PATH)' in source


def test_tower_state_validation_rejects_missing_or_unexpected_tower_keys() -> None:
    module = load_module()
    for missing, unexpected in (
        (["sam_model.encoder.weight"], []),
        ([], ["qwen2_model.unexpected"]),
    ):
        try:
            module.validate_tower_state(missing, unexpected)
        except RuntimeError as exc:
            assert "tower" in str(exc)
        else:
            raise AssertionError((missing, unexpected))


def test_tower_state_validation_ignores_non_tower_missing_keys() -> None:
    module = load_module()
    module.validate_tower_state(["language_model.model.weight", "adapter.proj.0.weight"], [])
