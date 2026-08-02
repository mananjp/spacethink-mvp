"""Domain-agnostic AutonomyGate.

Switches between autonomous action (PASSIVE) and human approval (ACTIVE) based
strictly on a domain Verifier's CalibrationStatus and an OversightPolicy.

Architecturally prevented from reading raw domain data or domain-specific scores.
"""
from __future__ import annotations

from domain import CalibrationStatus, OversightMode, OversightPolicy


def decide_oversight(
    status: CalibrationStatus,
    policy: OversightPolicy | None = None,
) -> OversightMode:
    """Determine the oversight mode for downstream action based strictly on CalibrationStatus.

    Args:
        status: The domain-agnostic calibration status emitted by a Verifier.
        policy: The active oversight policy rules (defaults to standard policy if None).

    Returns:
        OversightMode.PASSIVE if calibration passed and confidence meets threshold.
        OversightMode.ACTIVE if calibration failed or confidence is below threshold.
    """
    if policy is None:
        policy = OversightPolicy()

    # Rule 1: Must pass calibration if required by policy
    if policy.require_calibration and not status.passed:
        return OversightMode.ACTIVE

    # Rule 2: Must meet or exceed minimum calibrated confidence
    if status.confidence < policy.min_confidence:
        return OversightMode.ACTIVE

    return OversightMode.PASSIVE
