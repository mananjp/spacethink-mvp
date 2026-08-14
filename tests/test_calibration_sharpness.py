"""Calibration must be earned *jointly* with sharpness.

SBC on its own is gameable. A posterior that simply returns the prior is perfectly
rank-calibrated and completely useless: it says nothing, so the ground truth is
uniformly ranked by construction. Measured on the reaction-wheel twin, the ABC
posterior passed SBC only once it had widened to ~85% of the prior — i.e. it bought
its calibration by giving up all of its information.

A gate that grants autonomy on that basis grants it to a scorer that knows nothing.
So the derived CalibrationStatus has to reflect the standard pairing (Gneiting et al.
2007): maximise sharpness *subject to* calibration. These tests pin that.
"""

from __future__ import annotations

from evaluate.calibration import (
    PPCResult,
    SBCResult,
    derive_calibration_status,
    posterior_sharpness,
)


def _sbc(passed: bool, p: float, sharpness: float | None = None) -> SBCResult:
    return SBCResult(
        family="f",
        n_prior_samples=10,
        n_sims_per_sample=5,
        rank_histogram=[1, 1, 1, 1, 1, 1],
        uniformity_p_value=p,
        passed=passed,
        sharpness=sharpness,
    )


def _ppc(passed: bool, coverages: list[float]) -> PPCResult:
    return PPCResult(
        family="f",
        summary_stat_names=["a"],
        real_stats=[0.0],
        predicted_stats_mean=[0.0],
        predicted_stats_std=[1.0],
        coverage_fractions=coverages,
        passed=passed,
    )


# --------------------------------------------------------------------------
# The sharpness measure itself
# --------------------------------------------------------------------------


def test_prior_width_posterior_is_not_sharp():
    """Draws spanning the prior carry no information -> sharpness ~0."""
    import numpy as np

    draws = np.random.default_rng(0).uniform(0.0, 1.0, size=4000)
    assert posterior_sharpness(draws, 0.0, 1.0) < 0.1


def test_concentrated_posterior_is_sharp():
    """A tight posterior -> sharpness near 1."""
    import numpy as np

    draws = np.random.default_rng(0).normal(0.5, 0.002, size=4000)
    assert posterior_sharpness(draws, 0.0, 1.0) > 0.95


def test_sharpness_is_bounded():
    import numpy as np

    wide = np.random.default_rng(0).uniform(-10.0, 10.0, size=1000)
    assert 0.0 <= posterior_sharpness(wide, 0.0, 1.0) <= 1.0


# --------------------------------------------------------------------------
# The gate must not reward a vacuous posterior
# --------------------------------------------------------------------------


def test_vacuous_but_uniform_posterior_does_not_pass():
    """The exact failure mode found on the twin: calibrated by being uninformative."""
    status = derive_calibration_status(
        "reaction_wheel", _sbc(True, 0.67, sharpness=0.15), _ppc(True, [1.0])
    )
    assert status.passed is False, "an uninformative posterior must not earn autonomy"
    assert status.diagnostics["sharpness_passed"] is False


def test_calibrated_and_sharp_passes():
    status = derive_calibration_status(
        "reaction_wheel", _sbc(True, 0.4, sharpness=0.8), _ppc(True, [1.0])
    )
    assert status.passed is True
    assert status.diagnostics["sharpness_passed"] is True
    assert status.confidence > 0.5


def test_sharp_but_miscalibrated_still_fails():
    """Sharpness cannot buy back a failed SBC — that would be self-reporting again."""
    status = derive_calibration_status(
        "reaction_wheel", _sbc(False, 0.0, sharpness=0.95), _ppc(True, [1.0])
    )
    assert status.passed is False


def test_sharpness_absent_preserves_legacy_derivation():
    """With no sharpness measured, behaviour is exactly the documented formula."""
    status = derive_calibration_status("dom", _sbc(True, 0.5), _ppc(True, [1.0, 1.0]))
    assert status.passed is True
    assert status.confidence == 1.0
