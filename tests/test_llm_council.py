"""Unit tests for LLM Council deliberation and Human Validation Gate triage."""
from __future__ import annotations

from datetime import datetime, timezone
from domain import (
    CouncilRole,
    CouncilVerdict,
    EventOfInterest,
    FaultParameter,
    Hypothesis,
    Severity,
    SimResult,
    ValidationStatus,
    new_id,
)
from evaluate.human_gate import evaluate_human_gate
from evaluate.llm_council import LLMCouncil


def test_llm_council_deliberation_offline():
    event = EventOfInterest(
        id=new_id(), run_id="r1", channel="wheel_speed_rpm",
        start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc),
        score=2.8, severity=Severity.MEDIUM, detector_name="test",
    )
    hyp = Hypothesis(
        id=new_id(), event_id=event.id, text="bearing friction",
        mechanism="bearing_friction_increase", fault_params=(FaultParameter("friction", 0.6),),
        prior=0.4, generator="template",
    )
    sim = SimResult(
        hypothesis_id=hyp.id, distance=1.2, posterior=0.72, n_sims=10,
    )

    council = LLMCouncil(offline=True)
    consensus = council.deliberate(event, hyp, sim)

    assert len(consensus.individual_votes) == 4
    assert consensus.consensus_score > 0.6
    assert consensus.verdict in (CouncilVerdict.UNANIMOUS_AGREEMENT, CouncilVerdict.STRONG_CONSENSUS)


def test_human_gate_escalates_high_severity():
    event = EventOfInterest(
        id=new_id(), run_id="r1", channel="wheel_speed_rpm",
        start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc),
        score=4.5, severity=Severity.HIGH, detector_name="test",
    )
    hyp = Hypothesis(
        id=new_id(), event_id=event.id, text="bearing friction",
        mechanism="bearing_friction_increase", fault_params=(),
        prior=0.4, generator="template",
    )
    sim = SimResult(hypothesis_id=hyp.id, distance=1.0, posterior=0.80, n_sims=10)

    council = LLMCouncil(offline=True)
    consensus = council.deliberate(event, hyp, sim)
    status = evaluate_human_gate(event, consensus, sim.posterior)

    # High severity must escalate for human validation regardless of council consensus
    assert status == ValidationStatus.ESCALATED_PENDING_HUMAN


def test_human_gate_auto_approves_medium_severity_with_strong_consensus():
    event = EventOfInterest(
        id=new_id(), run_id="r1", channel="wheel_speed_rpm",
        start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc),
        score=2.0, severity=Severity.MEDIUM, detector_name="test",
    )
    hyp = Hypothesis(
        id=new_id(), event_id=event.id, text="bearing friction",
        mechanism="bearing_friction_increase", fault_params=(FaultParameter("friction", 0.6),),
        prior=0.4, generator="template",
    )
    sim = SimResult(hypothesis_id=hyp.id, distance=1.1, posterior=0.75, n_sims=10)

    council = LLMCouncil(offline=True)
    consensus = council.deliberate(event, hyp, sim)
    status = evaluate_human_gate(event, consensus, sim.posterior)

    assert status == ValidationStatus.AUTO_APPROVED
