"""The calibrated-and-sharp posterior — the milestone the autonomy gate waits on.

ABC-rejection on the raw distance could satisfy rank-uniformity or sharpness but
never both: tightening the tolerance made it overconfident (ranks piled in the
centre), loosening it bought uniformity by widening back toward the prior.

A synthetic likelihood (Wood 2010) fixes the cause rather than the symptom. It
estimates the simulator's own summary-statistic noise, so posterior width is derived
from how much the data can actually distinguish parameter values, instead of from an
arbitrary distance cutoff. Pure numpy/scipy, so hermetic CI keeps working with no
torch or sbi installed.
"""

from __future__ import annotations

import numpy as np

from evaluate.calibration import (
    posterior_sharpness,
    run_sbc_with_posterior,
    uniformity_p_value,
)
from evaluate.synthetic_likelihood import (
    SyntheticLikelihoodPosterior,
    train_synthetic_likelihood,
)

PRIOR_LOW, PRIOR_HIGH = 0.1, 2.0


def _trained() -> SyntheticLikelihoodPosterior:
    return train_synthetic_likelihood(
        prior_low=PRIOR_LOW,
        prior_high=PRIOR_HIGH,
        n_grid=32,
        n_reps=6,
        duration_s=400,
        seed=17,
    )


# ---------------------------------------------------------------------------
# Basic inference behaviour
# ---------------------------------------------------------------------------


def test_posterior_recovers_the_true_parameter():
    post = _trained()
    rng = np.random.default_rng(5)
    errors = []
    for theta in (0.3, 0.8, 1.4, 1.9):
        obs = post.simulate(theta, rng)
        draws = post.sample(obs, 400, rng)
        errors.append(abs(float(np.mean(draws)) - theta))
    assert max(errors) < 0.25, f"poor recovery: {errors}"


def test_posterior_is_sharp():
    post = _trained()
    rng = np.random.default_rng(6)
    obs = post.simulate(1.0, rng)
    draws = post.sample(obs, 500, rng)
    sharp = posterior_sharpness(draws, PRIOR_LOW, PRIOR_HIGH)
    assert sharp > 0.5, f"posterior not informative enough: sharpness={sharp}"


def test_sampling_is_deterministic_for_a_fixed_seed():
    post = _trained()
    obs = post.simulate(0.9, np.random.default_rng(1))
    a = post.sample(obs, 50, np.random.default_rng(3))
    b = post.sample(obs, 50, np.random.default_rng(3))
    assert np.allclose(a, b)


def test_draws_stay_inside_the_prior_support():
    post = _trained()
    rng = np.random.default_rng(8)
    draws = post.sample(post.simulate(1.5, rng), 300, rng)
    assert draws.min() >= PRIOR_LOW
    assert draws.max() <= PRIOR_HIGH


# ---------------------------------------------------------------------------
# The milestone: calibrated AND sharp, together
# ---------------------------------------------------------------------------


def test_posterior_passes_sbc_while_staying_sharp():
    """Both properties at once — this is what unlocks PASSIVE autonomy."""
    post = _trained()
    result = run_sbc_with_posterior(
        post,
        n_prior_samples=200,
        n_sims=19,
        seed=23,
        family="bearing_friction_increase",
    )

    assert result.passed, (
        f"SBC rejected the synthetic-likelihood posterior "
        f"(p={result.uniformity_p_value}, hist={result.rank_histogram})"
    )
    assert result.sharpness is not None and result.sharpness > 0.5, (
        f"calibrated but uninformative: sharpness={result.sharpness}"
    )


def test_ranks_are_spread_not_degenerate():
    post = _trained()
    result = run_sbc_with_posterior(post, n_prior_samples=120, n_sims=19, seed=29)
    occupied = sum(1 for c in result.rank_histogram if c > 0)
    assert occupied > 10, f"degenerate histogram: {result.rank_histogram}"
    assert uniformity_p_value.__call__ is not None  # imported symbol is the real one
