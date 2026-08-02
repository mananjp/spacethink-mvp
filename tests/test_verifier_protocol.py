"""Contract compliance and functional tests for Verifier protocol implementations."""
from datetime import datetime, timezone
import pandas as pd
import pytest

from domain import (
    CalibrationStatus,
    Evidence,
    FaultParameter,
    Hypothesis,
    VerificationResult,
    Verifier,
)
from evaluate.verifiers import AstroCatalogVerifier, ReactionWheelVerifier


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="hyp_001",
        event_id="evt_100",
        text="Reaction wheel friction increase 8x",
        mechanism="rw_friction",
        fault_params=(FaultParameter("friction", 8.0),),
        prior=0.7,
        generator="template",
    )


@pytest.fixture
def astro_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="hyp_astro_001",
        event_id="evt_astro_100",
        text="Known variable star RR Lyrae",
        mechanism="known_variable_star",
        fault_params=(),
        prior=0.8,
        generator="template",
    )


def test_reaction_wheel_verifier_protocol_compliance():
    rw_verifier = ReactionWheelVerifier()
    assert isinstance(rw_verifier, Verifier)

    cal_status = rw_verifier.calibration_status()
    assert isinstance(cal_status, CalibrationStatus)
    assert cal_status.domain == "reaction_wheel"
    assert cal_status.passed is True
    assert cal_status.confidence > 0.0
    assert cal_status.method == "SBC+PPC"


def test_reaction_wheel_verifier_execution(sample_hypothesis):
    rw_verifier = ReactionWheelVerifier()
    real_df = pd.DataFrame({
        "t": list(range(10)),
        "wheel_speed_rpm": [1000.0 + i for i in range(10)],
        "wheel_current_a": [0.5 + 0.01 * i for i in range(10)],
        "wheel_temp_c": [20.0 + 0.1 * i for i in range(10)],
    })
    evidence = Evidence(domain="reaction_wheel", raw_data={"real": real_df})

    res = rw_verifier.verify(sample_hypothesis, evidence)
    assert isinstance(res, VerificationResult)
    assert res.hypothesis_id == sample_hypothesis.id
    assert isinstance(res.verified, bool)
    assert isinstance(res.fit_score, float)


def test_astro_catalog_verifier_protocol_compliance():
    astro_verifier = AstroCatalogVerifier()
    assert isinstance(astro_verifier, Verifier)

    cal_status = astro_verifier.calibration_status()
    assert isinstance(cal_status, CalibrationStatus)
    assert cal_status.domain == "astronomical_transient"
    assert cal_status.passed is True
    assert cal_status.confidence > 0.0
    assert "catalog cross-match" in cal_status.method


def test_astro_catalog_verifier_execution(astro_hypothesis):
    astro_verifier = AstroCatalogVerifier()
    evidence = Evidence(
        domain="astronomical_transient",
        raw_data={"ra": 180.0, "dec": 45.0, "peak_magnitude": 14.5},
    )

    res = astro_verifier.verify(astro_hypothesis, evidence)
    assert isinstance(res, VerificationResult)
    assert res.hypothesis_id == astro_hypothesis.id
    assert res.verified is True
    assert res.diagnostics["matched_catalog_object"] == "known_variable_star"
