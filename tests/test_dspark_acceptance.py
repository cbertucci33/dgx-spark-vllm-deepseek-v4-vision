import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "dspark_acceptance.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dspark_acceptance", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DSparkAcceptanceTests(unittest.TestCase):
    PROMETHEUS_SAMPLE = """\
vllm:spec_decode_num_drafts_total{engine="0"} 3
vllm:spec_decode_num_drafts_total{engine="1"} 2
vllm:spec_decode_num_draft_tokens_total{engine="0"} 15
vllm:spec_decode_num_draft_tokens_total{engine="1"} 10
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 6
vllm:spec_decode_num_accepted_tokens_total{engine="1"} 3
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 3
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 2
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 2
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="1"} 1
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="2"} 1
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="2"} 0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="3"} 0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="3"} 0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="4"} 0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="4"} 0
vllm:generation_tokens_total{engine="0"} 9
vllm:generation_tokens_total{engine="1"} 5
"""

    def test_valid_acceptance_returns_fraction(self) -> None:
        module = load_module()

        rate = module.validate_acceptance(
            draft_tokens=100,
            accepted_tokens=50,
            minimum_rate=0.20,
        )

        self.assertEqual(rate, 0.50)

    def test_measurement_script_enforces_default_acceptance_floor(self) -> None:
        script = (ROOT / "tests" / "measure_vision_spec_rails.py").read_text()

        self.assertIn("from dspark_acceptance import (", script)
        self.assertIn('MIN_DSPARK_ACCEPTANCE_RATE", "0.20"', script)
        self.assertIn('MIN_DSPARK_DRAFTS", "8"', script)
        self.assertIn('MAX_DSPARK_QUALIFICATION_REQUESTS", "8"', script)
        self.assertIn("for _ in range(MAX_REQUESTS):", script)
        self.assertIn(".num_drafts >= MIN_DRAFTS", script)
        self.assertIn("parse_spec_metrics(text)", script)
        self.assertIn("delta_spec_metrics(before_spec, after_spec)", script)
        self.assertIn("validate_qualification(", script)
        self.assertIn("allow_nan=False", script)
        self.assertIn('"acceptance": stats._asdict()', script)

    def test_zero_draft_tokens_is_rejected_clearly(self) -> None:
        module = load_module()

        with self.assertRaisesRegex(RuntimeError, "did not produce draft tokens"):
            module.validate_acceptance(
                draft_tokens=0,
                accepted_tokens=0,
                minimum_rate=0.20,
            )

    def test_acceptance_below_minimum_is_rejected(self) -> None:
        module = load_module()

        with self.assertRaisesRegex(RuntimeError, "1.00%.*20.00%"):
            module.validate_acceptance(
                draft_tokens=100,
                accepted_tokens=1,
                minimum_rate=0.20,
            )

    def test_exact_threshold_is_accepted(self) -> None:
        module = load_module()
        self.assertEqual(
            module.validate_acceptance(
                draft_tokens=100, accepted_tokens=20, minimum_rate=0.20
            ),
            0.20,
        )

    def test_just_below_threshold_is_rejected(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(RuntimeError, "below required minimum"):
            module.validate_acceptance(
                draft_tokens=1000, accepted_tokens=199, minimum_rate=0.20
            )

    def test_invalid_thresholds_are_rejected(self) -> None:
        module = load_module()
        for threshold in (math.nan, math.inf, -math.inf, -0.01, 1.01):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "finite fraction"):
                    module.validate_acceptance(
                        draft_tokens=100,
                        accepted_tokens=50,
                        minimum_rate=threshold,
                    )

    def test_invalid_metric_counts_are_rejected(self) -> None:
        module = load_module()
        cases = (
            (math.nan, 1),
            (math.inf, 1),
            (100, math.nan),
            (-1, 0),
            (100, -1),
            (100, 101),
        )
        for draft_tokens, accepted_tokens in cases:
            with self.subTest(
                draft_tokens=draft_tokens, accepted_tokens=accepted_tokens
            ):
                with self.assertRaisesRegex(ValueError, "metric counts"):
                    module.validate_metric_counts(
                        draft_tokens=draft_tokens,
                        accepted_tokens=accepted_tokens,
                    )

    def test_metric_counts_return_acceptance_rate(self) -> None:
        module = load_module()
        self.assertEqual(
            module.validate_metric_counts(draft_tokens=100, accepted_tokens=25),
            0.25,
        )

    def test_prometheus_parser_aggregates_all_exact_labeled_series(self) -> None:
        module = load_module()
        metrics = module.parse_spec_metrics(self.PROMETHEUS_SAMPLE)
        self.assertEqual(metrics.num_drafts, 5)
        self.assertEqual(metrics.draft_tokens, 25)
        self.assertEqual(metrics.accepted_tokens, 9)
        self.assertEqual(metrics.accepted_per_position, (5, 3, 1, 0, 0))
        self.assertEqual(metrics.generation_tokens, 14)

    def test_prometheus_parser_rejects_cross_family_label_mismatch(self) -> None:
        module = load_module()
        text = self.PROMETHEUS_SAMPLE.replace(
            'vllm:generation_tokens_total{engine="0"} 9\n'
            'vllm:generation_tokens_total{engine="1"} 5',
            "vllm:generation_tokens_total 14",
        )
        with self.assertRaisesRegex(ValueError, "label sets must match"):
            module.parse_spec_metrics(text)

    def test_prometheus_parser_rejects_missing_or_fractional_counters(self) -> None:
        module = load_module()
        missing = "\n".join(
            line
            for line in self.PROMETHEUS_SAMPLE.splitlines()
            if not line.startswith("vllm:generation_tokens_total")
        )
        with self.assertRaisesRegex(RuntimeError, "missing metric"):
            module.parse_spec_metrics(missing)
        fractional = self.PROMETHEUS_SAMPLE.replace(
            'vllm:spec_decode_num_drafts_total{engine="0"} 3',
            'vllm:spec_decode_num_drafts_total{engine="0"} 3.5',
        )
        with self.assertRaisesRegex(ValueError, "whole-number"):
            module.parse_spec_metrics(fractional)

    def test_delta_rejects_counter_reset_and_position_shape_change(self) -> None:
        module = load_module()
        before = module.parse_spec_metrics(self.PROMETHEUS_SAMPLE)
        hidden_reset_text = self.PROMETHEUS_SAMPLE.replace(
            'num_drafts_total{engine="0"} 3',
            'num_drafts_total{engine="0"} 1',
        ).replace(
            'num_drafts_total{engine="1"} 2',
            'num_drafts_total{engine="1"} 10',
        )
        reset = module.parse_spec_metrics(hidden_reset_text)
        with self.assertRaisesRegex(RuntimeError, "counter reset"):
            module.delta_spec_metrics(before, reset)
        changed_shape_text = "\n".join(
            line
            for line in self.PROMETHEUS_SAMPLE.splitlines()
            if 'position="4"' not in line
        )
        changed_shape = module.parse_spec_metrics(changed_shape_text)
        with self.assertRaisesRegex(RuntimeError, "position shape changed"):
            module.delta_spec_metrics(before, changed_shape)

    def test_stats_validate_prefix_shape_and_report_useful_acceptance(self) -> None:
        module = load_module()
        metrics = module.SpecMetrics(10, 50, 25, (10, 7, 5, 2, 1), 35)
        stats = module.calculate_acceptance_stats(metrics)
        self.assertEqual(stats.acceptance_rate, 0.5)
        self.assertEqual(stats.accepted_tokens_per_draft, 2.5)
        self.assertEqual(stats.mean_acceptance_length, 3.5)
        self.assertEqual(stats.per_position_rates, (1.0, 0.7, 0.5, 0.2, 0.1))

    def test_stats_reject_inconsistent_per_position_evidence(self) -> None:
        module = load_module()
        cases = (
            (10, 25, (10, 7, 5, 2, 0)),
            (10, 25, (9, 10, 3, 2, 1)),
            (10, 25, (11, 7, 4, 2, 1)),
        )
        for drafts, accepted, positions in cases:
            with self.subTest(positions=positions):
                with self.assertRaisesRegex(ValueError, "per-position"):
                    module.calculate_acceptance_stats(
                        module.SpecMetrics(
                            drafts, 50, accepted, positions, 35
                        )
                    )

    def test_qualification_accepts_isolated_sufficient_sample(self) -> None:
        module = load_module()
        metrics = module.SpecMetrics(10, 50, 25, (10, 7, 5, 2, 1), 35)
        stats = module.validate_qualification(
            metrics,
            minimum_rate=0.20,
            minimum_drafts=8,
            completion_tokens=35,
        )
        self.assertEqual(stats.acceptance_rate, 0.5)
        self.assertEqual(stats.mean_acceptance_length, 3.5)

    def test_qualification_rejects_small_or_contaminated_sample(self) -> None:
        module = load_module()
        small = module.SpecMetrics(7, 35, 21, (7, 6, 4, 3, 1), 28)
        with self.assertRaisesRegex(RuntimeError, "only 7 drafts"):
            module.validate_qualification(
                small, minimum_rate=0.20, minimum_drafts=8, completion_tokens=28
            )
        contaminated = module.SpecMetrics(10, 50, 25, (10, 7, 5, 2, 1), 40)
        with self.assertRaisesRegex(RuntimeError, "concurrent traffic"):
            module.validate_qualification(
                contaminated,
                minimum_rate=0.20,
                minimum_drafts=8,
                completion_tokens=35,
            )


if __name__ == "__main__":
    unittest.main()
