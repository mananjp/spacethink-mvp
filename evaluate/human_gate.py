"""Human Validation Gate — Triage logic for escalating major discoveries & anomalies to human operators.

Determines whether an anomaly/hypothesis pair requires human-in-the-loop validation
based on event severity, anomaly score, posterior probability, and LLM Council consensus.
"""
from __future__ import annotations

from domain import (
    CouncilConsensus,
    CouncilVerdict,
    EventOfInterest,
    Severity,
    ValidationStatus,
)


def evaluate_human_gate(
    event: EventOfInterest,
    consensus: CouncilConsensus,
    posterior: float | None = None,
) -> ValidationStatus:
    """Evaluate whether an event/diagnosis requires human sign-off or can be auto-approved."""
    # 1. High-severity events or extreme Z-scores ALWAYS escalate for human sign-off
    if event.severity == Severity.HIGH or event.score >= 3.0:
        return ValidationStatus.ESCALATED_PENDING_HUMAN

    # 2. Split or rejected LLM Council decisions ALWAYS escalate
    if consensus.verdict in (CouncilVerdict.SPLIT_COUNCIL, CouncilVerdict.REJECTED_BY_COUNCIL):
        return ValidationStatus.ESCALATED_PENDING_HUMAN

    # 3. Low posterior or low council consensus score escalates
    if (posterior is not None and posterior < 0.35) or consensus.consensus_score < 0.65:
        return ValidationStatus.ESCALATED_PENDING_HUMAN

    # 4. Routine low/medium severity events with strong Council agreement are auto-approved
    if consensus.verdict in (CouncilVerdict.UNANIMOUS_AGREEMENT, CouncilVerdict.STRONG_CONSENSUS):
        return ValidationStatus.AUTO_APPROVED

    return ValidationStatus.ESCALATED_PENDING_HUMAN
