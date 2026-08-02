"""Contract + smoke tests — run with: pytest"""
from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from domain import FaultParameter, SimMapping
from explore.detector import ThresholdDetector, ZScoreDetector
from evaluate.scorer import DistanceScorer, normalize_posteriors
from hypothesize.generator import StubLlm
from hypothesize.groq_explainer import generate_exhyte_timeline
from ingest.synthetic_generator import generate_reaction_wheel_telemetry
from plan.planner import run_closed_loop
from twin.simulator import ToySimulator


def test_threshold_detector_stub_returns_empty():
    detector = ThresholdDetector()
    df = generate_reaction_wheel_telemetry(fault_type="none", n_points=500)
    assert detector.detect(df, ["wheel_speed_rpm"]) == []


def test_zscore_detector_flags_injected_fault():
    df = generate_reaction_wheel_telemetry(fault_type="stiction", n_points=4000, fault_start=1500, seed=1)
    detector = ZScoreDetector()
    events = detector.detect(df, ["wheel_speed_rpm", "wheel_current_a"], run_id="test-run")
    assert len(events) > 0


def test_toy_simulator_runs_and_respects_fault_params():
    twin = ToySimulator().configure(SimMapping(subsystem="reaction_wheel", fault_params=(FaultParameter("friction", 0.8),)))
    sim = twin.run(duration_s=1000, seed=0)
    assert len(sim) == 1000
    assert sim["wheel_temp_c"].iloc[-1] > sim["wheel_temp_c"].iloc[0]


def test_stub_llm_generates_plausible_hypotheses():
    from domain import EventOfInterest, Severity, new_id

    event = EventOfInterest(
        id=new_id(), run_id="r", channel="wheel_speed_rpm",
        start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc),
        score=5.0, severity=Severity.HIGH, detector_name="test",
    )
    llm = StubLlm()
    hyps = llm.generate(event)
    # 3 fault mechanisms + 1 nominal ("no fault") candidate.
    assert len(hyps) == 4
    assert any(h.mechanism == "nominal_no_fault" for h in hyps)
    assert all(llm.critique(h).plausible for h in hyps)


def test_distance_scorer_normalizes_to_sum_one():
    from domain import Hypothesis, new_id

    real = pd.DataFrame({
        "wheel_speed_rpm": [4000] * 100,
        "wheel_current_a": [0.5] * 100,
        "wheel_temp_c": [25] * 100,
    })
    sims_a = [real.copy() for _ in range(5)]
    sims_b = [real.assign(wheel_speed_rpm=real["wheel_speed_rpm"] + 500) for _ in range(5)]

    hyp_a = Hypothesis(id=new_id(), event_id="e", text="", mechanism="a", fault_params=(), prior=0.5, generator="template")
    hyp_b = Hypothesis(id=new_id(), event_id="e", text="", mechanism="b", fault_params=(), prior=0.5, generator="template")

    scorer = DistanceScorer()
    r_a = scorer.score(hyp_a, real, sims_a)
    r_b = scorer.score(hyp_b, real, sims_b)
    ranked = normalize_posteriors([r_a, r_b])

    assert abs(sum(r.posterior for r in ranked) - 1.0) < 1e-6
    assert ranked[0].hypothesis_id == hyp_a.id  # closer match should win


def test_exhyte_timeline_generator_offline():
    timeline = generate_exhyte_timeline(
        event_id="test-1",
        channel="wheel_speed_rpm",
        severity="high",
        score=5.2,
        top_hypothesis="bearing_friction_increase",
        top_text="Test description",
        posterior=0.75,
        ranked_mechanisms=[("bearing_friction_increase", 0.75), ("stiction", 0.25)],
    )
    assert "EXHYTE Closed-Loop Timeline" in timeline
    assert "EXPLORE" in timeline
    assert "HYPOTHESIZE" in timeline
    assert "TEST" in timeline
    assert "REFINE" in timeline


def test_full_closed_loop_end_to_end():
    df = generate_reaction_wheel_telemetry(fault_type="friction_increase", n_points=3000, fault_start=1200, seed=7)
    report = run_closed_loop(df, n_sims_per_hypothesis=5)
    assert "run_id" in report
    assert report["n_events"] >= 0
    if report["n_events"] > 0:
        assert report["events"][0]["top_hypothesis"] is not None
        assert "ai_timeline" in report["events"][0]
