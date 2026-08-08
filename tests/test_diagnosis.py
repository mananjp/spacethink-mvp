"""Diagnostic-correctness tests — the regression guard for the scorer fix.

These lock in the behaviour that the v0 engine lacked: the closed loop must
(a) diagnose each fault type correctly and (b) NOT invent a fault on nominal
data. They exercise the real detector/twin/scorer, only the LLM stays stubbed.
"""

from __future__ import annotations

from collections import Counter

from evaluate.harness import evaluate
from explore.detector import ZScoreDetector
from ingest.synthetic_generator import generate_reaction_wheel_telemetry
from plan.planner import CHANNELS, run_closed_loop

EXPECTED = {
    "none": "nominal_no_fault",
    "friction_increase": "bearing_friction_increase",
    "encoder_dropout": "encoder_dropout",
    "stiction": "stiction",
}


def _run_prediction(report: dict) -> str:
    if not report["events"]:
        return "nominal_no_fault"
    votes = Counter(
        e["top_hypothesis"] for e in report["events"] if e["top_hypothesis"]
    )
    return votes.most_common(1)[0][0] if votes else "nominal_no_fault"


def test_engine_discriminates_each_fault_type():
    for fault_type, expected in EXPECTED.items():
        df = generate_reaction_wheel_telemetry(
            fault_type=fault_type, n_points=4000, fault_start=2000, seed=3
        )
        report = run_closed_loop(df, n_sims_per_hypothesis=8)
        assert _run_prediction(report) == expected, f"{fault_type} misdiagnosed"


def test_nominal_telemetry_is_not_diagnosed_as_a_fault():
    # The v0 bug: confident 'encoder_dropout' on healthy data. Guard against it.
    for seed in (11, 22, 33):
        df = generate_reaction_wheel_telemetry(
            fault_type="none", n_points=4000, fault_start=2000, seed=seed
        )
        report = run_closed_loop(df, n_sims_per_hypothesis=8)
        assert _run_prediction(report) == "nominal_no_fault"


def test_harness_accuracy_and_zero_nominal_false_positives():
    result = evaluate(n_per_class=5, n_sims_per_hypothesis=8)
    assert result["accuracy"] >= 0.85, result
    # An alarm system that cries wolf on healthy data is worse than useless.
    assert result["nominal_false_positive_rate"] == 0.0, result


def test_detector_is_quiet_on_nominal_and_flags_every_fault():
    """Regression guard for detector over-triggering (v0 fired ~150 events/run).

    Nominal telemetry should produce only a handful of low-severity blips, while
    every fault type must still be flagged (or diagnosis would never run).
    """
    detector = ZScoreDetector()
    for seed in (3, 11, 22, 33, 44):
        nominal = generate_reaction_wheel_telemetry(
            fault_type="none", n_points=4000, fault_start=2000, seed=seed
        )
        assert len(detector.detect(nominal, CHANNELS)) <= 6

    for fault_type in ("friction_increase", "encoder_dropout", "stiction"):
        df = generate_reaction_wheel_telemetry(
            fault_type=fault_type, n_points=4000, fault_start=2000, seed=3
        )
        assert len(detector.detect(df, CHANNELS)) >= 1, fault_type
