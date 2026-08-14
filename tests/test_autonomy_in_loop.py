"""The AutonomyGate must actually run inside `run_closed_loop`, not just in isolation.

These tests pin the wiring:
  - every detected event carries the gate's decision (`autonomy_mode`, `requires_human`)
    and the run carries a `calibration` summary;
  - an injected PASSED, high-confidence status -> PASSIVE;
  - an injected FAILED status -> ACTIVE and requires_human, and it can override an
    otherwise-auto-approved human-gate result (calibration is a hard precondition);
  - the default (no injection) path gates on the real, derived calibration status.
"""

from __future__ import annotations

from domain import CalibrationStatus, OversightPolicy
from ingest.synthetic_generator import generate_reaction_wheel_telemetry
from plan.planner import run_closed_loop


def _tele():
    return generate_reaction_wheel_telemetry(
        fault_type="friction_increase", n_points=3000, fault_start=1200, seed=7
    )


def _passed_status() -> CalibrationStatus:
    return CalibrationStatus(
        domain="reaction_wheel", passed=True, confidence=0.99, method="SBC+PPC"
    )


def _failed_status() -> CalibrationStatus:
    return CalibrationStatus(
        domain="reaction_wheel", passed=False, confidence=0.10, method="SBC+PPC"
    )


def test_loop_reports_calibration_and_gate_fields():
    report = run_closed_loop(_tele(), n_sims_per_hypothesis=4, calibration_status=_passed_status())
    assert "calibration" in report
    assert report["calibration"]["passed"] is True
    for ev in report["events"]:
        assert ev["autonomy_mode"] in ("active", "passive")
        assert "requires_human" in ev
        assert ev["calibrated_confidence"] == 0.99


def test_passed_status_allows_passive():
    report = run_closed_loop(
        _tele(),
        n_sims_per_hypothesis=4,
        calibration_status=_passed_status(),
        oversight_policy=OversightPolicy(min_confidence=0.8, require_calibration=True),
    )
    assert report["events"], "expected at least one detected event"
    assert all(ev["autonomy_mode"] == "passive" for ev in report["events"])


def test_failed_calibration_forces_active_and_human():
    report = run_closed_loop(
        _tele(),
        n_sims_per_hypothesis=4,
        calibration_status=_failed_status(),
        oversight_policy=OversightPolicy(min_confidence=0.8, require_calibration=True),
    )
    assert report["events"], "expected at least one detected event"
    for ev in report["events"]:
        assert ev["autonomy_mode"] == "active"
        assert ev["requires_human"] is True  # failed calibration overrides auto-approval


def test_default_path_uses_derived_calibration():
    """No injection: the loop must gate on the *derived* status, whatever it says.

    This previously asserted ACTIVE, which pinned a transient fact (the scorer had
    not yet passed SBC) rather than a contract. Now that the synthetic-likelihood
    posterior is calibrated and sharp, the derived status passes and the honest
    answer is PASSIVE. What must hold in both worlds is that the gate *follows* the
    derived status instead of reporting its own — so that is what is asserted.
    """
    report = run_closed_loop(_tele(), n_sims_per_hypothesis=4)
    assert "calibration" in report
    assert report["calibration"]["method"] == "SBC+PPC"

    calibration = report["calibration"]
    # Derived, never self-reported: confidence must come from the diagnostics.
    assert calibration["diagnostics"]["sbc_method"] == "amortized_posterior_sbc"
    assert calibration["diagnostics"]["sharpness_passed"] is True

    expected_mode = "passive" if calibration["passed"] else "active"
    for ev in report["events"]:
        assert ev["autonomy_mode"] == expected_mode
        assert ev["calibrated_confidence"] == calibration["confidence"]
        if not calibration["passed"]:
            assert ev["requires_human"] is True


def test_calibration_is_earned_not_asserted():
    """The gate may only open on a posterior that is both calibrated and sharp."""
    report = run_closed_loop(_tele(), n_sims_per_hypothesis=4)
    diagnostics = report["calibration"]["diagnostics"]

    if report["calibration"]["passed"]:
        assert diagnostics["sbc_passed"] is True
        assert diagnostics["ppc_passed"] is True
        assert diagnostics["sharpness"] >= diagnostics["min_sharpness"], (
            "an uninformative posterior must never pass the gate"
        )
