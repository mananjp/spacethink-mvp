"""Contract tests for Simulation-Based Calibration.

These tests pin down what SBC must actually *measure*. The property under test is
the defining one (Talts et al. 2018): if posterior samples are drawn from the true
posterior, the rank of the ground-truth parameter among those samples is uniformly
distributed. A check that does not have this property is not SBC, and a calibration
gate built on it carries no information.

The regression these guard against: `run_sbc` previously accepted a `scorer` and
never used it, ranking theta* against *prior* draws by raw simulator distance. That
statistic is pinned at rank 0 by construction (histogram [12,0,0,0,0,0,0]), so it
reported "uncalibrated" for every scorer, trained or not.

The toy Gaussian model here is used deliberately: its posterior is known in closed
form, so a correct SBC implementation has an unambiguous right answer, independent
of any spacecraft modelling assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate.calibration import run_sbc, sbc_ranks, uniformity_p_value

# ---------------------------------------------------------------------------
# Toy conjugate model with an analytically known posterior.
#   theta ~ N(0, 1);  x | theta ~ N(theta, 1)  =>  theta | x ~ N(x/2, 1/sqrt(2))
# ---------------------------------------------------------------------------

_POSTERIOR_SD = 1.0 / np.sqrt(2.0)


def _prior(rng: np.random.Generator) -> float:
    return float(rng.normal(0.0, 1.0))


def _simulate(theta: float, rng: np.random.Generator) -> float:
    return float(theta + rng.normal(0.0, 1.0))


def _oracle_posterior(x: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact posterior — SBC must accept this."""
    return rng.normal(x / 2.0, _POSTERIOR_SD, size=n)


def _overconfident_posterior(x: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Correct centre, far too narrow — SBC must reject this."""
    return rng.normal(x / 2.0, _POSTERIOR_SD / 10.0, size=n)


def _underconfident_posterior(x: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Correct centre, far too wide — SBC must reject this."""
    return rng.normal(x / 2.0, _POSTERIOR_SD * 10.0, size=n)


def _biased_posterior(x: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Right width, shifted centre — SBC must reject this."""
    return rng.normal(x / 2.0 + 3.0, _POSTERIOR_SD, size=n)


# ---------------------------------------------------------------------------
# The defining property
# ---------------------------------------------------------------------------


def test_oracle_posterior_yields_uniform_ranks():
    """An exactly-correct posterior must produce uniform ranks and pass."""
    ranks = sbc_ranks(
        prior_sampler=_prior,
        simulator=_simulate,
        posterior_sampler=_oracle_posterior,
        n_trials=300,
        n_posterior_samples=19,
        seed=7,
    )
    assert len(ranks) == 300
    assert all(0 <= r <= 19 for r in ranks), "ranks must lie in [0, L]"

    p = uniformity_p_value(ranks, n_posterior_samples=19)
    assert p > 0.05, f"exact posterior must look uniform, got p={p}"


def test_oracle_ranks_span_the_whole_range():
    """Guards the specific regression: ranks collapsing onto a single bin."""
    ranks = sbc_ranks(
        prior_sampler=_prior,
        simulator=_simulate,
        posterior_sampler=_oracle_posterior,
        n_trials=300,
        n_posterior_samples=19,
        seed=11,
    )
    occupied = len(set(ranks))
    assert occupied > 10, f"expected ranks spread across bins, only {occupied} occupied"


@pytest.mark.parametrize(
    "sampler,label",
    [
        (_overconfident_posterior, "overconfident"),
        (_underconfident_posterior, "underconfident"),
        (_biased_posterior, "biased"),
    ],
)
def test_miscalibrated_posteriors_are_rejected(sampler, label):
    """SBC must have teeth: each classic failure mode has to be caught."""
    ranks = sbc_ranks(
        prior_sampler=_prior,
        simulator=_simulate,
        posterior_sampler=sampler,
        n_trials=300,
        n_posterior_samples=19,
        seed=7,
    )
    p = uniformity_p_value(ranks, n_posterior_samples=19)
    assert p < 0.01, f"{label} posterior must be rejected, got p={p}"


def test_uniformity_test_is_one_sided():
    """A perfectly flat histogram is evidence *for* calibration, not against it.

    The previous implementation used a two-sided normal approximation, which
    rejects a suspiciously-flat histogram. Goodness-of-fit is an upper-tail test.
    """
    flat = [i % 20 for i in range(400)]  # exactly 20 per bin
    p = uniformity_p_value(flat, n_posterior_samples=19)
    assert p > 0.5, f"a flat histogram must not be rejected, got p={p}"


# ---------------------------------------------------------------------------
# The scorer must actually be consulted
# ---------------------------------------------------------------------------


def test_run_sbc_consults_the_scorer():
    """The regression that started this: `scorer` was accepted and ignored."""
    calls: list[int] = []

    class _SpyScorer:
        name = "spy"

        def __init__(self, channels=None):
            self.channels = channels or ["wheel_speed_rpm"]

        def _distance(self, real, sim) -> float:
            calls.append(1)
            n = min(len(real), len(sim))
            r = real["wheel_temp_c"].to_numpy()[:n]
            s = sim["wheel_temp_c"].to_numpy()[:n]
            return float(np.sqrt(np.mean((r - s) ** 2)))

    run_sbc(_SpyScorer(), n_prior_samples=3, n_sims=4, seed=3)
    assert calls, "run_sbc must use the scorer it was handed"


def test_run_sbc_reports_a_non_degenerate_histogram():
    """Whatever the verdict, the statistic must not be pinned to one bin."""
    res = run_sbc(None, n_prior_samples=40, n_sims=9, seed=5)

    assert sum(res.rank_histogram) == 40
    assert len(res.rank_histogram) == 10, "histogram has L+1 bins"
    occupied = sum(1 for c in res.rank_histogram if c > 0)
    assert occupied > 1, f"degenerate rank histogram: {res.rank_histogram}"
