"""Pluggable detector/scorer wiring — smoke tests.

Verifies the advanced components (Telemanom detector, SBIScorer) are reachable
from the default run path and degrade gracefully (no torch/sbi/trained model
required), without asserting accuracy — the accuracy comparison lives in the
diagnosis harness and docs/NEXT_STEPS.md.
"""

from __future__ import annotations

import pytest

from ingest.synthetic_generator import generate_reaction_wheel_telemetry
from plan.components import build_detector, build_scorer
from plan.planner import run_closed_loop


def _tele():
    return generate_reaction_wheel_telemetry(
        fault_type="friction_increase", n_points=3000, fault_start=1200, seed=7
    )


def test_registry_resolves_and_rejects_unknown():
    assert build_detector(None).__class__.__name__ == "ZScoreDetector"
    assert build_scorer(None).__class__.__name__ == "SignatureScorer"
    with pytest.raises(ValueError):
        build_detector("nope")
    with pytest.raises(ValueError):
        build_scorer("nope")


def test_default_path_unchanged():
    report = run_closed_loop(_tele(), n_sims_per_hypothesis=4)
    assert "run_id" in report and report["n_events"] >= 0


def test_telemanom_detector_is_pluggable():
    report = run_closed_loop(_tele(), n_sims_per_hypothesis=4, detector=build_detector("telemanom"))
    assert "run_id" in report


def test_sbi_scorer_is_pluggable_via_fallback():
    # SBIScorer uses its kernel fallback when sbi/trained posteriors are absent.
    report = run_closed_loop(_tele(), n_sims_per_hypothesis=4, scorer=build_scorer("sbi"))
    assert "run_id" in report
    if report["n_events"] > 0:
        assert report["events"][0]["top_hypothesis"] is not None
