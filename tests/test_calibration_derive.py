"""Tests for `derive_calibration_status` — the calibration -> CalibrationStatus map.

The whole point of the AutonomyGate is that its confidence is *derived from calibration
diagnostics*, never self-reported. These tests pin that derivation:
  - passed requires BOTH SBC and PPC to pass,
  - confidence = 0.5*ppc_coverage + 0.5*min(1, sbc_p/0.05), bounded [0,1],
  - failing SBC drags confidence down even at full PPC coverage,
  - empty coverage fails closed.
"""

from __future__ import annotations

from evaluate.calibration import (
    PPCResult,
    SBCResult,
    derive_calibration_status,
)


def _sbc(passed: bool, p: float) -> SBCResult:
    return SBCResult(
        family="f",
        n_prior_samples=10,
        n_sims_per_sample=5,
        rank_histogram=[1, 1, 1, 1, 1, 1],
        uniformity_p_value=p,
        passed=passed,
    )


def _ppc(passed: bool, coverages: list[float]) -> PPCResult:
    return PPCResult(
        family="f",
        summary_stat_names=["a", "b"],
        real_stats=[0.0, 0.0],
        predicted_stats_mean=[0.0, 0.0],
        predicted_stats_std=[1.0, 1.0],
        coverage_fractions=coverages,
        passed=passed,
    )


def test_confidence_is_derived_not_self_reported():
    # Full PPC coverage + well-uniform SBC -> confidence ~1.0, passed True.
    status = derive_calibration_status("dom", _sbc(True, 0.5), _ppc(True, [1.0, 1.0]))
    assert status.domain == "dom"
    assert status.passed is True
    assert status.confidence == 1.0
    assert status.diagnostics["ppc_coverage"] == 1.0


def test_failing_sbc_fails_status_and_lowers_confidence():
    # PPC covers perfectly but SBC fails (p=0): passed must be False and confidence
    # must drop to ~0.5 (the PPC half only) -- self-reported confidence cannot rescue it.
    status = derive_calibration_status("dom", _sbc(False, 0.0), _ppc(True, [1.0, 1.0]))
    assert status.passed is False
    assert status.confidence == 0.5


def test_empty_coverage_fails_closed():
    status = derive_calibration_status("dom", _sbc(False, 0.0), _ppc(False, []))
    assert status.passed is False
    assert status.confidence == 0.0


def test_confidence_is_bounded():
    # p far above 0.05 must clamp the SBC term at 1.0 (no >1 confidence).
    status = derive_calibration_status("dom", _sbc(True, 0.9), _ppc(True, [1.0, 1.0]))
    assert status.confidence <= 1.0
