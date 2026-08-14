"""Contract tests for the Posterior Predictive Check.

These tests pin down what a PPC must actually *measure*. The defining property:
if posterior draws come from the true posterior, then data simulated from those
draws is distributed like data from the true parameter — so a held-out replicate
generated at the true theta falls at a uniformly-distributed quantile of the
posterior-predictive sample. A check without this property is not a PPC, and a
calibration gate leaning on it carries no information.

The regression these guard against: ``run_ppc`` accepted a ``scorer``, never used
it, and compared the twin at theta against *the same twin at the same theta*. That
is a tautology — it returned ``passed=True`` with coverage 1.0 for a scorer whose
distance function is a constant zero, at a physically absurd friction of 50.0 —
while contributing 0.4 of the confidence weight that opens the autonomy gate.

Why a replicate rather than the observation itself: scoring the observation inside
its own posterior-predictive distribution uses the data twice, which makes the
statistic conservative and non-uniform even for an exact posterior. Ranking an
*independent* draw from the same ground-truth theta removes the double-use and
restores exact uniformity, so the oracle has an unambiguous right answer.

The toy conjugate model is used deliberately: its posterior and posterior
predictive are both known in closed form, so correctness here is independent of
any spacecraft modelling assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate.calibration import (
    posterior_predictive_pit,
    run_ppc_with_posterior,
    run_sbc_with_posterior,
)

# ---------------------------------------------------------------------------
# Toy conjugate model.
#   theta ~ N(0, 1);  x | theta ~ N(theta, 1)  =>  theta | x ~ N(x/2, 1/sqrt(2))
# Posterior predictive for a replicate:  x_rep | x ~ N(x/2, sqrt(1 + 1/2))
# ---------------------------------------------------------------------------

_POSTERIOR_SD = 1.0 / np.sqrt(2.0)


class _ToyPosterior:
    """Posterior object in the shape ``run_ppc_with_posterior`` consumes.

    Mirrors the ``SyntheticLikelihoodPosterior`` surface: prior support, a forward
    ``simulate``, and a ``sample`` conditioned on an observation.
    """

    prior_low = -4.0
    prior_high = 4.0

    def __init__(self, sd_factor: float = 1.0, shift: float = 0.0):
        self._sd_factor = sd_factor
        self._shift = shift

    def prior_sample(self, rng: np.random.Generator) -> float:
        """theta ~ N(0, 1) — the prior the conjugate posterior below assumes.

        Without this hook the check would draw theta from the uniform support, for
        which N(x/2, 1/sqrt(2)) is *not* the posterior, and the oracle would look
        miscalibrated for a reason that has nothing to do with the check.
        """
        return float(rng.normal(0.0, 1.0))

    def simulate(self, theta: float, rng: np.random.Generator) -> np.ndarray:
        return np.asarray([float(theta) + rng.normal(0.0, 1.0)], dtype=float)

    def sample(self, stats, n: int, rng: np.random.Generator) -> np.ndarray:
        x = float(np.asarray(stats, dtype=float).ravel()[0])
        return rng.normal(
            x / 2.0 + self._shift, _POSTERIOR_SD * self._sd_factor, size=n
        )


class _ObservationIgnoringPosterior(_ToyPosterior):
    """Returns prior draws regardless of the observation.

    This is the exact failure the old ``run_ppc`` could not see. It is perfectly
    calibrated in the trivial sense (it knows nothing and admits it), so a PPC that
    accepts it is measuring nothing about inference quality.
    """

    def sample(self, stats, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=n)


# ---------------------------------------------------------------------------
# The defining property
# ---------------------------------------------------------------------------


def test_oracle_posterior_yields_uniform_pit():
    """An exactly-correct posterior must produce uniform PIT values."""
    pit, observations = posterior_predictive_pit(
        _ToyPosterior(),
        n_trials=300,
        n_predictive_draws=19,
        seed=7,
    )
    assert pit.shape == (300, 1)
    assert observations.shape == (300, 1)
    assert np.all((pit >= 0.0) & (pit <= 1.0)), "PIT values must lie in [0, 1]"

    # A correct posterior predictive puts the replicate at a uniform quantile.
    from evaluate.calibration import uniformity_p_value

    ranks = np.round(pit[:, 0] * 19).astype(int)
    p = uniformity_p_value(ranks, n_posterior_samples=19)
    assert p > 0.05, f"exact posterior predictive must look uniform, got p={p}"


def test_oracle_passes_the_check():
    result = run_ppc_with_posterior(
        _ToyPosterior(), n_trials=300, n_predictive_draws=19, seed=13
    )
    assert result.passed, (
        f"exact posterior rejected: coverage={result.coverage_fractions}, "
        f"diagnostics={result.diagnostics}"
    )


@pytest.mark.parametrize(
    "posterior,label",
    [
        (_ToyPosterior(sd_factor=10.0), "underconfident"),
        (_ToyPosterior(shift=3.0), "biased"),
        (_ObservationIgnoringPosterior(), "ignores-the-observation"),
    ],
)
def test_miscalibrated_posteriors_are_rejected(posterior, label):
    """The PPC must have teeth: each failure mode in its remit has to be caught."""
    result = run_ppc_with_posterior(
        posterior, n_trials=300, n_predictive_draws=19, seed=17
    )
    assert not result.passed, (
        f"{label} posterior slipped through the PPC: "
        f"coverage={result.coverage_fractions}, diagnostics={result.diagnostics}"
    )


def test_conditional_stratification_is_what_catches_an_unresponsive_posterior():
    """Pin the mechanism, not just the verdict.

    A posterior returning the prior is *marginally* calibrated — pooled PIT values are
    uniform by construction, so a marginal-only check accepts it. Only the
    observation-stratified test sees that the predictive never moves with the data.
    Guards against anyone "simplifying" the stratification away.
    """
    result = run_ppc_with_posterior(
        _ObservationIgnoringPosterior(), n_trials=300, n_predictive_draws=19, seed=17
    )
    assert not result.passed

    marginal_ok = min(result.diagnostics["marginal_pit_p_values"])
    conditional = result.diagnostics["min_conditional_pit_p"]
    assert marginal_ok > result.diagnostics["bonferroni_alpha"], (
        "expected the prior-returning posterior to look fine marginally; "
        f"got marginal p={marginal_ok}"
    )
    assert conditional < result.diagnostics["bonferroni_alpha"], (
        f"stratified check should have caught it, got conditional p={conditional}"
    )


def test_check_is_a_function_of_the_posterior():
    """The direct regression test.

    The old check returned ``passed=True`` with coverage 1.0 for every input — a
    constant-zero distance function at a physically absurd friction of 50.0 included.
    """
    good = run_ppc_with_posterior(
        _ToyPosterior(), n_trials=200, n_predictive_draws=19, seed=23
    )
    bad = run_ppc_with_posterior(
        _ToyPosterior(shift=3.0), n_trials=200, n_predictive_draws=19, seed=23
    )
    assert good.coverage_fractions != bad.coverage_fractions, (
        "PPC output is independent of the posterior under test — it is measuring nothing"
    )
    assert good.passed and not bad.passed


def test_coverage_is_reported_near_nominal_for_a_correct_posterior():
    """Coverage is the interpretable companion to the PIT uniformity verdict."""
    result = run_ppc_with_posterior(
        _ToyPosterior(), n_trials=300, n_predictive_draws=39, seed=29
    )
    coverage = float(np.mean(result.coverage_fractions))
    assert 0.90 <= coverage <= 1.0, (
        f"a correct posterior should cover its replicate near the nominal 95%, got {coverage}"
    )


# ---------------------------------------------------------------------------
# The documented blind spot, and the leg of the gate that covers it
# ---------------------------------------------------------------------------


def test_the_gate_does_not_use_the_tautological_check():
    """``run_ppc`` cannot fail; the gate must not be wired back to it.

    Pinned as a test rather than a comment because the failure is invisible: the gate
    keeps reporting a high confidence either way, and only the diagnostics reveal which
    check produced it.
    """
    from evaluate.calibration import run_ppc
    from evaluate.verifiers.reaction_wheel import default_reaction_wheel_calibration

    class _ConstantZeroScorer:
        name = "constant-zero"

        def _distance(self, real, sim):
            return 0.0

    tautological = run_ppc(_ConstantZeroScorer(), param_value=50.0, n_sims=10, seed=1)
    assert tautological.passed and tautological.diagnostics["overall_coverage"] == 1.0, (
        "run_ppc was expected to be a tautology; if it now has teeth, update this test "
        "and reconsider deprecating it"
    )

    status = default_reaction_wheel_calibration()
    assert status.diagnostics["ppc_coverage"] < 1.0, (
        "the gate's PPC leg reports perfect coverage — it is back on the tautological "
        "check and cannot fail"
    )


def test_ppc_is_blind_to_overconfidence_but_sbc_is_not():
    """Why the gate needs three legs, pinned as an executable fact.

    Simulator noise enters the true and the estimated predictive distribution alike,
    so it dilutes posterior error: a posterior ten times too narrow yields a predictive
    only ~18% too narrow, and the predictive check cannot reject it. Measured here at
    coverage 0.92 against the oracle's 0.97 — the right direction, too small to call.

    This is not a defect to fix by tightening the PPC threshold, which would simply
    start rejecting correct posteriors. Overconfidence is SBC's job, and SBC does it
    decisively. The test fails loudly if either half of that division of labour breaks.
    """
    overconfident = _ToyPosterior(sd_factor=0.1)

    ppc = run_ppc_with_posterior(
        overconfident, n_trials=300, n_predictive_draws=39, seed=31
    )
    assert ppc.passed, (
        "PPC unexpectedly rejected an overconfident posterior — if the check gained "
        "this much power, re-check that it has not started rejecting the oracle too"
    )

    oracle_coverage = float(
        np.mean(
            run_ppc_with_posterior(
                _ToyPosterior(), n_trials=300, n_predictive_draws=39, seed=31
            ).coverage_fractions
        )
    )
    coverage = float(np.mean(ppc.coverage_fractions))
    assert coverage < oracle_coverage, (
        f"overconfident predictive should undercover relative to the oracle: "
        f"{coverage} vs {oracle_coverage}"
    )

    sbc = run_sbc_with_posterior(overconfident, n_prior_samples=300, n_sims=19, seed=31)
    assert not sbc.passed, (
        "SBC must catch what the predictive check cannot — without this the gate has "
        f"no leg covering overconfidence (sbc p={sbc.uniformity_p_value})"
    )
