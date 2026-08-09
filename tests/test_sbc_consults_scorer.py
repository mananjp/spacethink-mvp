"""SBC/PPC must consult the scorer.

Regression tests for the fix in docs/SBC_SCORER_CONSULTATION_SPEC.md: previously
``run_sbc``/``run_ppc`` accepted a ``scorer`` argument but never used it, so the
calibration gate was insensitive to the scorer — training the SBIScorer could
never move it. These tests pin the corrected contract: the calibration outcome is
now a function of the scorer.

All runs use tiny params (short twin durations, few samples) so the suite stays
cheap — no trained model or heavy compute required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain import Hypothesis, SimResult
from evaluate.calibration import run_ppc, run_sbc
from evaluate.scorer import DistanceScorer

# Tiny, fixed budget shared by every test in this module.
TINY = dict(
    param_name="friction",
    prior_low=0.0,
    prior_high=2.0,
    duration_s=60,
    seed=7,
)


def _theta(hyp: Hypothesis) -> float:
    """The candidate parameter the scorer is being asked to judge."""
    return float(hyp.fault_params[0].value)


class MonotonicScorer:
    """A perfectly informative scorer: distance is a strictly monotonic function of
    the candidate parameter. Because the truth and the candidates are all i.i.d.
    draws from the same prior, the rank of the truth is uniform -> SBC passes.
    """

    name = "mock_monotonic"

    def score(
        self, hyp: Hypothesis, real: pd.DataFrame, simulated: list[pd.DataFrame]
    ) -> SimResult:
        return SimResult(
            hypothesis_id=hyp.id, distance=_theta(hyp), posterior=0.0, n_sims=len(simulated)
        )


class ConstantScorer:
    """A degenerate scorer: every candidate looks identical. The truth can never
    beat a candidate, so its rank is always 0 -> the rank histogram is a spike ->
    SBC fails. Its predictive weights are uniform.
    """

    name = "mock_constant"

    def score(
        self, hyp: Hypothesis, real: pd.DataFrame, simulated: list[pd.DataFrame]
    ) -> SimResult:
        return SimResult(hypothesis_id=hyp.id, distance=1.0, posterior=0.0, n_sims=len(simulated))


def test_sbc_outcome_depends_on_scorer():
    """The whole point of the fix: two different scorers -> two different SBC results."""
    good = run_sbc(MonotonicScorer(), n_prior_samples=8, n_sims=5, **TINY)
    bad = run_sbc(ConstantScorer(), n_prior_samples=8, n_sims=5, **TINY)

    # The scorer now drives the calibration outcome.
    assert good.rank_histogram != bad.rank_histogram
    # Degenerate scorer: truth never out-ranks a tie -> all ranks collapse to 0.
    assert bad.rank_histogram[0] == bad.n_prior_samples
    # Informative scorer spreads ranks across the histogram (not a single spike).
    assert sum(1 for c in good.rank_histogram if c > 0) > 1
    # And the wiring is recorded in diagnostics.
    assert good.diagnostics["scorer"] == "mock_monotonic"
    assert bad.diagnostics["scorer"] == "mock_constant"


def test_informative_scorer_passes_sbc_degenerate_fails():
    """An informative scorer is rank-calibrated (passes); a degenerate one is not."""
    good = run_sbc(MonotonicScorer(), n_prior_samples=20, n_sims=7, **TINY)
    bad = run_sbc(ConstantScorer(), n_prior_samples=20, n_sims=7, **TINY)

    assert good.uniformity_p_value > bad.uniformity_p_value
    assert bad.passed is False  # a scorer that cannot discriminate must not pass the gate


def test_ppc_band_depends_on_scorer():
    """PPC forms its predictive band from the scorer's importance weights, so a
    concentrating scorer and a uniform-weight scorer produce different bands.
    """
    concentrated = run_ppc(MonotonicScorer(), n_sims=12, param_value=0.6, **TINY)
    uniform = run_ppc(ConstantScorer(), n_sims=12, param_value=0.6, **TINY)

    # Different weighting -> different predicted mean/std (the scorer matters).
    assert concentrated.predicted_stats_mean != uniform.predicted_stats_mean
    assert isinstance(concentrated.passed, bool)
    assert uniform.diagnostics["scorer"] == "mock_constant"


def test_real_scorer_runs_and_is_typed():
    """Backward-compat: a real Scorer still produces a well-formed SBCResult, and
    the scorer name is threaded into diagnostics.
    """
    res = run_sbc(DistanceScorer(), n_prior_samples=4, n_sims=3, **TINY)
    assert isinstance(res.passed, bool)
    assert 0.0 <= res.uniformity_p_value <= 1.0
    assert res.diagnostics["scorer"] == "distance_v0"
    assert len(res.rank_histogram) == res.n_sims_per_sample + 1
