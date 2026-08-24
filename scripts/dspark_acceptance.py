#!/usr/bin/env python3
"""DSpark speculative-acceptance validation helpers."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import NamedTuple


Labels = tuple[tuple[str, str], ...]


class SpecMetrics(NamedTuple):
    num_drafts: int
    draft_tokens: int
    accepted_tokens: int
    accepted_per_position: tuple[int, ...]
    generation_tokens: int


class AcceptanceStats(NamedTuple):
    acceptance_rate: float
    accepted_tokens_per_draft: float
    mean_acceptance_length: float
    per_position_rates: tuple[float, ...]


class SpecSnapshot:
    """Prometheus counters retaining canonical label-set identity."""

    def __init__(
        self,
        series: dict[str, dict[Labels, int]],
        positions: dict[Labels, dict[int, int]],
    ) -> None:
        self.series = series
        self.positions = positions

    @property
    def num_drafts(self) -> int:
        return sum(self.series["num_drafts"].values())

    @property
    def draft_tokens(self) -> int:
        return sum(self.series["draft_tokens"].values())

    @property
    def accepted_tokens(self) -> int:
        return sum(self.series["accepted_tokens"].values())

    @property
    def generation_tokens(self) -> int:
        return sum(self.series["generation_tokens"].values())

    @property
    def accepted_per_position(self) -> tuple[int, ...]:
        count = len(next(iter(self.positions.values())))
        return tuple(
            sum(values[position] for values in self.positions.values())
            for position in range(count)
        )

    def as_evidence(self) -> dict[str, object]:
        def label_dict(labels: Labels) -> dict[str, str]:
            return dict(labels)

        return {
            "aggregate": {
                "num_drafts": self.num_drafts,
                "draft_tokens": self.draft_tokens,
                "accepted_tokens": self.accepted_tokens,
                "accepted_per_position": self.accepted_per_position,
                "generation_tokens": self.generation_tokens,
            },
            "series": {
                field: [
                    {"labels": label_dict(labels), "value": value}
                    for labels, value in sorted(values.items())
                ]
                for field, values in self.series.items()
            },
            "accepted_per_position_series": [
                {
                    "labels": label_dict(labels),
                    "values": tuple(values[pos] for pos in range(len(values))),
                }
                for labels, values in sorted(self.positions.items())
            ],
        }


_SCALAR_METRICS = {
    "vllm:spec_decode_num_drafts_total": "num_drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
}
_POSITION_METRIC = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
_LABEL_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:]*)="((?:\\.|[^"\\])*)"(?:,|$)')


def _whole_counter(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite, nonnegative whole-number counter")
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite, nonnegative whole-number counter"
        ) from exc
    if not numeric.is_finite() or numeric < 0 or numeric != numeric.to_integral_value():
        raise ValueError(f"{name} must be a finite, nonnegative whole-number counter")
    return int(numeric)


def _decode_label(value: str) -> str:
    return value.replace(r"\n", "\n").replace(r'\"', '"').replace(r"\\", "\\")


def _parse_sample(sample: str) -> tuple[str, Labels]:
    if "{" not in sample:
        return sample, ()
    if not sample.endswith("}"):
        raise ValueError(f"malformed Prometheus sample: {sample}")
    name, body = sample[:-1].split("{", 1)
    labels: list[tuple[str, str]] = []
    offset = 0
    for match in _LABEL_RE.finditer(body):
        if match.start() != offset:
            raise ValueError(f"malformed Prometheus labels: {body}")
        labels.append((match.group(1), _decode_label(match.group(2))))
        offset = match.end()
    if offset != len(body):
        raise ValueError(f"malformed Prometheus labels: {body}")
    if len({name for name, _ in labels}) != len(labels):
        raise ValueError(f"duplicate Prometheus label in: {body}")
    return name, tuple(sorted(labels))


def validate_minimum_rate(minimum_rate: float) -> float:
    if not math.isfinite(minimum_rate) or not 0.0 <= minimum_rate <= 1.0:
        raise ValueError("DSpark acceptance threshold must be a finite fraction within [0, 1]")
    return minimum_rate


def validate_minimum_drafts(minimum_drafts: object) -> int:
    value = _whole_counter(minimum_drafts, "DSpark minimum drafts")
    if value == 0:
        raise ValueError("DSpark minimum drafts must be positive")
    return value


def validate_metric_counts(*, draft_tokens: float, accepted_tokens: float) -> float:
    try:
        draft = _whole_counter(draft_tokens, "DSpark draft tokens")
        accepted = _whole_counter(accepted_tokens, "DSpark accepted tokens")
    except ValueError as exc:
        raise ValueError(f"DSpark metric counts are invalid: {exc}") from exc
    if accepted > draft:
        raise ValueError(
            "DSpark metric counts must satisfy 0 <= accepted_tokens <= draft_tokens"
        )
    if draft == 0:
        raise RuntimeError("DSpark did not produce draft tokens during qualification")
    return accepted / draft


def validate_acceptance(
    *, draft_tokens: float, accepted_tokens: float, minimum_rate: float
) -> float:
    minimum_rate = validate_minimum_rate(minimum_rate)
    rate = validate_metric_counts(
        draft_tokens=draft_tokens, accepted_tokens=accepted_tokens
    )
    if rate < minimum_rate:
        raise RuntimeError(
            f"DSpark acceptance {rate:.2%} is below required minimum {minimum_rate:.2%}"
        )
    return rate


def parse_spec_metrics(text: str) -> SpecSnapshot:
    series: dict[str, dict[Labels, int]] = {
        field: {} for field in _SCALAR_METRICS.values()
    }
    positions: dict[Labels, dict[int, int]] = {}
    seen_names: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        sample, raw_value = parts
        metric_name, labels = _parse_sample(sample)
        if metric_name in _SCALAR_METRICS:
            field = _SCALAR_METRICS[metric_name]
            if labels in series[field]:
                raise ValueError(f"duplicate metric series for {metric_name}: {labels}")
            series[field][labels] = _whole_counter(raw_value, metric_name)
            seen_names.add(metric_name)
        elif metric_name == _POSITION_METRIC:
            label_map = dict(labels)
            raw_position = label_map.pop("position", None)
            if raw_position is None or not raw_position.isdigit():
                raise ValueError(f"{_POSITION_METRIC} is missing an integer position label")
            identity = tuple(sorted(label_map.items()))
            position = int(raw_position)
            values = positions.setdefault(identity, {})
            if position in values:
                raise ValueError(
                    f"duplicate {_POSITION_METRIC} series for {identity} position {position}"
                )
            values[position] = _whole_counter(
                raw_value, f"{_POSITION_METRIC}[{position}]"
            )
            seen_names.add(_POSITION_METRIC)

    required = set(_SCALAR_METRICS) | {_POSITION_METRIC}
    missing = sorted(required - seen_names)
    if missing:
        raise RuntimeError(f"missing metric series: {', '.join(missing)}")

    identity_sets = [set(values) for values in series.values()] + [set(positions)]
    if any(keys != identity_sets[0] for keys in identity_sets[1:]):
        raise ValueError("DSpark scalar and per-position metric label sets must match")
    expected_shape: tuple[int, ...] | None = None
    for identity, values in positions.items():
        shape = tuple(sorted(values))
        if shape != tuple(range(len(shape))):
            raise ValueError(
                f"DSpark per-position labels for {identity} must be contiguous from zero"
            )
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("DSpark per-position metric shapes must match across labels")
    return SpecSnapshot(series, positions)


def delta_spec_metrics(before: SpecSnapshot, after: SpecSnapshot) -> SpecMetrics:
    if set(before.series) != set(after.series):
        raise RuntimeError("DSpark metric field set changed during qualification")
    scalar_deltas: dict[str, int] = {}
    for field in before.series:
        if set(before.series[field]) != set(after.series[field]):
            raise RuntimeError(
                f"DSpark {field} label series changed during qualification"
            )
        total = 0
        for labels, start in before.series[field].items():
            delta = after.series[field][labels] - start
            if delta < 0:
                raise RuntimeError(
                    f"DSpark {field} counter reset for labels {dict(labels)}"
                )
            total += delta
        scalar_deltas[field] = total

    if set(before.positions) != set(after.positions):
        raise RuntimeError("DSpark per-position label series changed during qualification")
    position_totals: list[int] | None = None
    for labels, start_values in before.positions.items():
        end_values = after.positions[labels]
        if set(start_values) != set(end_values):
            raise RuntimeError(
                f"DSpark acceptance position shape changed for labels {dict(labels)}"
            )
        if position_totals is None:
            position_totals = [0] * len(start_values)
        for position, start in start_values.items():
            delta = end_values[position] - start
            if delta < 0:
                raise RuntimeError(
                    "DSpark per-position counter reset for "
                    f"labels {dict(labels)} position {position}"
                )
            position_totals[position] += delta

    assert position_totals is not None
    return SpecMetrics(
        num_drafts=scalar_deltas["num_drafts"],
        draft_tokens=scalar_deltas["draft_tokens"],
        accepted_tokens=scalar_deltas["accepted_tokens"],
        accepted_per_position=tuple(position_totals),
        generation_tokens=scalar_deltas["generation_tokens"],
    )


def calculate_acceptance_stats(metrics: SpecMetrics) -> AcceptanceStats:
    drafts = _whole_counter(metrics.num_drafts, "DSpark drafts")
    draft_tokens = _whole_counter(metrics.draft_tokens, "DSpark draft tokens")
    accepted = _whole_counter(metrics.accepted_tokens, "DSpark accepted tokens")
    _whole_counter(metrics.generation_tokens, "generation tokens")
    positions = tuple(
        _whole_counter(value, f"DSpark per-position counter {position}")
        for position, value in enumerate(metrics.accepted_per_position)
    )
    if drafts == 0:
        raise RuntimeError("DSpark did not produce drafts during qualification")
    if not positions:
        raise ValueError("DSpark per-position evidence is empty")
    if draft_tokens < drafts or draft_tokens > drafts * len(positions):
        raise ValueError("DSpark draft-token count is impossible for the observed drafts")
    if accepted > draft_tokens:
        raise ValueError("DSpark accepted tokens cannot exceed draft tokens")
    if any(value > drafts for value in positions):
        raise ValueError("DSpark per-position counters cannot exceed draft count")
    if any(left < right for left, right in zip(positions, positions[1:])):
        raise ValueError("DSpark per-position counters must be nonincreasing")
    if sum(positions) != accepted:
        raise ValueError("DSpark per-position counters must sum to accepted tokens")
    if draft_tokens < accepted + drafts - positions[0]:
        raise ValueError("DSpark draft-token and rejection counts are inconsistent")

    accepted_per_draft = accepted / drafts
    return AcceptanceStats(
        acceptance_rate=accepted / draft_tokens,
        accepted_tokens_per_draft=accepted_per_draft,
        mean_acceptance_length=1.0 + accepted_per_draft,
        per_position_rates=tuple(value / drafts for value in positions),
    )


def validate_qualification(
    metrics: SpecMetrics,
    *,
    minimum_rate: float,
    minimum_drafts: object,
    completion_tokens: object,
) -> AcceptanceStats:
    stats = calculate_acceptance_stats(metrics)
    required_drafts = validate_minimum_drafts(minimum_drafts)
    if metrics.num_drafts < required_drafts:
        raise RuntimeError(
            f"DSpark qualification observed only {metrics.num_drafts} drafts; "
            f"at least {required_drafts} are required"
        )
    completion_count = _whole_counter(completion_tokens, "completion tokens")
    if metrics.generation_tokens != completion_count:
        raise RuntimeError(
            "DSpark qualification metrics include concurrent traffic: "
            f"generation delta {metrics.generation_tokens} != completion usage "
            f"{completion_count}"
        )
    validate_acceptance(
        draft_tokens=metrics.draft_tokens,
        accepted_tokens=metrics.accepted_tokens,
        minimum_rate=minimum_rate,
    )
    return stats
