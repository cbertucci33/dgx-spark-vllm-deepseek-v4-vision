"""Fail-closed vision adapter layout resolution."""

from __future__ import annotations

import operator
import pathlib


_TOWER_PREFIXES = ("sam_model.", "qwen2_model.")


def validate_tower_state(missing: list[str], unexpected: list[str]) -> None:
    """Reject incomplete or incompatible tower state while ignoring wrapper keys."""
    tower_missing = [name for name in missing if name.startswith(_TOWER_PREFIXES)]
    tower_unexpected = [
        name for name in unexpected if name.startswith(_TOWER_PREFIXES)
    ]
    if tower_missing or tower_unexpected:
        raise RuntimeError(
            "tower state mismatch "
            f"(missing={tower_missing[:3]}, unexpected={tower_unexpected[:3]})"
        )


def require_vision_asset(name: str, value: str | None) -> pathlib.Path:
    """Return an existing vision artifact path or fail before model startup."""
    if not value:
        raise RuntimeError(f"{name} vision asset path is required")
    path = pathlib.Path(value)
    if not path.is_file():
        raise RuntimeError(f"{name} vision asset does not exist: {path}")
    return path


def _parse_tiles(value: object, *, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} tiles must be an integer; got {value!r}")
    try:
        tiles = int(value) if isinstance(value, str) else operator.index(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} tiles must be an integer; got {value!r}") from exc
    if tiles < 0:
        raise ValueError(f"{source} tiles must be non-negative; got {tiles}")
    return tiles


def extract_metadata_tiles(metadata: object) -> object | None:
    """Return an explicitly present tile count without inventing a layout."""
    if isinstance(metadata, dict):
        return metadata.get("tiles")
    return getattr(metadata, "tiles", None)


def resolve_checkpoint_tiles(*, metadata_tiles: object | None, override: str | None) -> int:
    """Resolve the trained tile layout without silently guessing."""
    override_tiles = (
        _parse_tiles(override, source="DSV4_VISION_TILES")
        if override is not None
        else None
    )
    if metadata_tiles is None:
        if override_tiles is None:
            raise RuntimeError(
                "vision adapter tile metadata is unavailable; set "
                "DSV4_VISION_TILES explicitly only for a verified legacy adapter"
            )
        return override_tiles

    checkpoint_tiles = _parse_tiles(metadata_tiles, source="checkpoint")
    if override_tiles is not None and override_tiles != checkpoint_tiles:
        raise RuntimeError(
            f"DSV4_VISION_TILES={override_tiles} does not match checkpoint "
            f"config.tiles={checkpoint_tiles}"
        )
    return checkpoint_tiles
