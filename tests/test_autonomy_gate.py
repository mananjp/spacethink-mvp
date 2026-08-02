"""Unit tests for AutonomyGate oversight decision logic."""
import pytest

from domain import CalibrationStatus, OversightMode, OversightPolicy
from evaluate.autonomy_gate import decide_oversight
from evaluate.verifiers import AstroCatalogVerifier, ReactionWheelVerifier


def test_autonomy_gate_reaction_wheel_calibrated_pass():
    rw_verifier = ReactionWheelVerifier(calibrated=True, confidence=0.95)
    policy = OversightPolicy(min_confidence=0.80, require_calibration=True)

    mode = decide_oversight(rw_verifier.calibration_status(), policy)
    assert mode == OversightMode.PASSIVE


def test_autonomy_gate_reaction_wheel_uncalibrated_active():
    # If calibration failed, gate MUST require human active intervention
    rw_verifier = ReactionWheelVerifier(calibrated=False, confidence=0.99)
    policy = OversightPolicy(min_confidence=0.80, require_calibration=True)

    mode = decide_oversight(rw_verifier.calibration_status(), policy)
    assert mode == OversightMode.ACTIVE


def test_autonomy_gate_low_confidence_active():
    # If confidence is below policy threshold, gate MUST require human active intervention
    rw_verifier = ReactionWheelVerifier(calibrated=True, confidence=0.75)
    policy = OversightPolicy(min_confidence=0.85, require_calibration=True)

    mode = decide_oversight(rw_verifier.calibration_status(), policy)
    assert mode == OversightMode.ACTIVE


def test_autonomy_gate_domain_agnostic_astro_catalog():
    # Exact same gate logic evaluates AstroCatalogVerifier's CalibrationStatus
    astro_verifier = AstroCatalogVerifier(calibrated=True, confidence=0.90)
    policy = OversightPolicy(min_confidence=0.85, require_calibration=True)

    mode = decide_oversight(astro_verifier.calibration_status(), policy)
    assert mode == OversightMode.PASSIVE


def test_autonomy_gate_domain_agnostic_astro_uncalibrated():
    astro_verifier = AstroCatalogVerifier(calibrated=False, confidence=0.90)
    policy = OversightPolicy(min_confidence=0.85, require_calibration=True)

    mode = decide_oversight(astro_verifier.calibration_status(), policy)
    assert mode == OversightMode.ACTIVE
