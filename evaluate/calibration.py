"""Calibration guardrails — Simulation-Based Calibration (SBC) and Posterior Predictive Checks (PPC).

A scorer that fails SBC or PPC is blocked, not deployed. Diagnostics land
in runstore next to every SimResult.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

from domain import CalibrationStatus, FaultParameter, SimMapping
from twin.simulator import ToySimulator


@dataclass
class SBCResult:
    """Result of Simulation-Based Calibration check.

    ``sharpness`` is the companion to ``passed``: rank-uniformity alone is satisfied
    by a posterior that just returns the prior, so it is only meaningful paired with
    a measure of how much the posterior actually narrows. ``None`` means sharpness
    was not measured (legacy callers), not that it was zero.
    """

    family: str
    n_prior_samples: int
    n_sims_per_sample: int
    rank_histogram: list[int]
    uniformity_p_value: float
    passed: bool
    sharpness: float | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PPCResult:
    """Result of Posterior Predictive Check."""

    family: str
    summary_stat_names: list[str]
    real_stats: list[float]
    predicted_stats_mean: list[float]
    predicted_stats_std: list[float]
    coverage_fractions: list[float]  # fraction of real stats within predicted 95% CI
    passed: bool
    diagnostics: dict = field(default_factory=dict)


def posterior_sharpness(
    draws: Sequence[float] | np.ndarray, prior_low: float, prior_high: float
) -> float:
    """How much narrower than the prior a posterior is, on a [0, 1] scale.

    ``1.0`` is a point mass; ``0.0`` is no narrower than the prior (i.e. the
    posterior carries no information about the parameter). Reported alongside
    rank-uniformity because uniformity on its own is trivially satisfied by
    returning the prior.
    """
    prior_sd = abs(float(prior_high) - float(prior_low)) / np.sqrt(12.0)
    if prior_sd <= 0:
        return 0.0
    posterior_sd = float(np.std(np.asarray(draws, dtype=float)))
    return float(np.clip(1.0 - posterior_sd / prior_sd, 0.0, 1.0))


def rank_histogram(ranks: Sequence[int], n_posterior_samples: int) -> list[int]:
    """Bin SBC ranks into the L+1 bins they can occupy."""
    n_bins = n_posterior_samples + 1
    counts = np.bincount(np.asarray(ranks, dtype=int), minlength=n_bins)
    return [int(c) for c in counts[:n_bins]]


def uniformity_p_value(ranks: Sequence[int], n_posterior_samples: int) -> float:
    """Chi-squared goodness-of-fit p-value against a uniform rank distribution.

    Goodness-of-fit is an **upper-tail** test: only histograms *further* from flat
    than chance count as evidence of miscalibration. (The previous implementation
    used a two-sided normal approximation, which also rejected suspiciously-flat
    histograms — i.e. it rejected the very evidence that calibration holds.)
    """
    counts = np.asarray(rank_histogram(ranks, n_posterior_samples), dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    expected = n / len(counts)
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    return float(chi2.sf(statistic, len(counts) - 1))


def sbc_ranks(
    prior_sampler: Callable[[np.random.Generator], Any],
    simulator: Callable[[Any, np.random.Generator], Any],
    posterior_sampler: Callable[[Any, int, np.random.Generator], Any],
    n_trials: int,
    n_posterior_samples: int,
    seed: int = 42,
) -> list[int]:
    """Compute SBC rank statistics (Talts et al. 2018).

    For each trial: draw θ* from the prior, simulate x* ~ p(x|θ*), draw L samples
    from the *posterior* q(θ|x*), and record the rank of θ* among them. If q is the
    true posterior, these ranks are uniform on {0, …, L}.

    The posterior is what is under test. Ranking θ* against draws from the *prior*
    — as this module previously did — tests nothing about the inference procedure
    and is pinned to rank 0 whenever the parameter has any effect on the data.
    """
    if n_posterior_samples < 1:
        raise ValueError("n_posterior_samples must be >= 1")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(seed)
    ranks: list[int] = []

    for _ in range(n_trials):
        theta_true = prior_sampler(rng)
        observed = simulator(theta_true, rng)
        draws = np.asarray(
            posterior_sampler(observed, n_posterior_samples, rng), dtype=float
        ).ravel()

        if draws.size != n_posterior_samples:
            raise ValueError(
                f"posterior_sampler returned {draws.size} draws, expected {n_posterior_samples}"
            )

        below = int(np.count_nonzero(draws < theta_true))
        tied = int(np.count_nonzero(draws == theta_true))
        # Randomised tie-breaking keeps ranks uniform under discrete draws.
        ranks.append(below + (int(rng.integers(0, tied + 1)) if tied else 0))

    return ranks


def _scorer_distance(
    scorer: object, real: pd.DataFrame, sim: pd.DataFrame, channels: Sequence[str]
) -> float:
    """Distance between two telemetry frames, delegated to the scorer under test.

    A scorer exposing ``_distance`` defines its own geometry, and that geometry is
    exactly what SBC must probe. The normalized-RMSE fallback applies only when no
    scorer is supplied.
    """
    scorer_distance = getattr(scorer, "_distance", None)
    if callable(scorer_distance):
        return float(scorer_distance(real, sim))

    n_pts = min(len(real), len(sim))
    dists = []
    for ch in channels:
        r = real[ch].to_numpy()[:n_pts]
        s = sim[ch].to_numpy()[:n_pts]
        scale = max(float(np.std(r)), 1e-6)
        dists.append(float(np.sqrt(np.mean(((r - s) / scale) ** 2))))
    return float(np.mean(dists))


def _run_twin(
    twin_cls: type, param_name: str, theta: float, duration_s: float, seed: int
) -> pd.DataFrame:
    """Simulate one telemetry window under a single fault parameter value."""
    twin = twin_cls()
    twin.configure(
        SimMapping(
            subsystem="reaction_wheel",
            fault_params=(FaultParameter(param_name, float(theta)),),
        )
    )
    return twin.run(duration_s=duration_s, seed=seed)


def run_sbc(
    scorer: object = None,
    twin_cls: type = ToySimulator,
    family: str = "bearing_friction_increase",
    param_name: str = "friction",
    prior_low: float = 0.1,
    prior_high: float = 2.0,
    n_prior_samples: int = 100,
    n_sims: int = 20,
    channels: list[str] | None = None,
    seed: int = 42,
    duration_s: float = 400,
    acceptance_rate: float = 0.125,
    alpha: float = 0.01,
) -> SBCResult:
    """Simulation-Based Calibration for a scorer/twin/family triple.

    The posterior under test is the **ABC-rejection posterior induced by the
    scorer's own distance function**: draw candidates from the prior, simulate each,
    keep the ``n_sims`` closest under ``scorer._distance``. ``acceptance_rate`` is the
    ABC tolerance — the knob that decides whether the resulting posterior is too
    tight (ranks pile in the middle), too loose (ranks pile at the edges), or
    calibrated (ranks uniform). Passing SBC is therefore something a scorer *earns*.
    """
    channels = list(channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"])
    if not 0.0 < acceptance_rate <= 1.0:
        raise ValueError("acceptance_rate must lie in (0, 1]")
    n_candidates = max(n_sims + 1, int(round(n_sims / acceptance_rate)))

    def prior_sampler(rng: np.random.Generator) -> float:
        return float(rng.uniform(prior_low, prior_high))

    def simulator(theta: float, rng: np.random.Generator) -> pd.DataFrame:
        return _run_twin(twin_cls, param_name, theta, duration_s, int(rng.integers(0, 100_000)))

    def posterior_sampler(observed: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
        candidates = rng.uniform(prior_low, prior_high, size=n_candidates)
        distances = np.array(
            [
                _scorer_distance(
                    scorer,
                    observed,
                    _run_twin(
                        twin_cls,
                        param_name,
                        theta,
                        duration_s,
                        int(rng.integers(0, 100_000)),
                    ),
                    channels,
                )
                for theta in candidates
            ]
        )
        return candidates[np.argsort(distances)[:n]]

    # Sharpness is recorded per trial as the posterior is drawn, so calibration and
    # informativeness are measured on exactly the same draws.
    per_trial_sharpness: list[float] = []

    def measured_posterior_sampler(
        observed: pd.DataFrame, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        draws = posterior_sampler(observed, n, rng)
        per_trial_sharpness.append(posterior_sharpness(draws, prior_low, prior_high))
        return draws

    ranks = sbc_ranks(
        prior_sampler=prior_sampler,
        simulator=simulator,
        posterior_sampler=measured_posterior_sampler,
        n_trials=n_prior_samples,
        n_posterior_samples=n_sims,
        seed=seed,
    )

    histogram = rank_histogram(ranks, n_sims)
    p_value = uniformity_p_value(ranks, n_sims)
    sharpness = float(np.mean(per_trial_sharpness)) if per_trial_sharpness else 0.0

    return SBCResult(
        family=family,
        n_prior_samples=n_prior_samples,
        n_sims_per_sample=n_sims,
        rank_histogram=histogram,
        uniformity_p_value=p_value,
        passed=bool(p_value > alpha),
        sharpness=round(sharpness, 4),
        diagnostics={
            "method": "abc_rejection_sbc",
            "scorer": getattr(scorer, "name", type(scorer).__name__),
            "n_candidates_per_trial": n_candidates,
            "acceptance_rate": acceptance_rate,
            "alpha": alpha,
            "duration_s": duration_s,
        },
    )


def run_ppc(
    scorer: object,
    twin_cls: type = ToySimulator,
    family: str = "bearing_friction_increase",
    param_name: str = "friction",
    param_value: float = 0.6,
    n_sims: int = 50,
    channels: list[str] | None = None,
    seed: int = 42,
) -> PPCResult:
    """Twin self-consistency check. **Not** a posterior predictive check.

    .. warning::
       Despite the name and the ``scorer`` parameter, this consults neither. It
       generates data from the twin at ``param_value`` and checks that summary
       statistics of *the same twin at the same parameter* fall inside their own 95%
       band — a tautology. Measured: it returns ``passed=True`` with coverage 1.0 for a
       scorer whose distance function is a constant zero, at every parameter value
       tried including a physically absurd friction of 50.0.

       It must never feed :func:`derive_calibration_status`. It did, contributing 0.4 of
       the confidence weight that opens the AutonomyGate, which meant one of the three
       legs of the calibration claim could not fail. Use
       :func:`run_ppc_with_posterior`, which is a genuine function of the posterior
       under test.

    What it does measure honestly is twin run-to-run stability: whether repeated runs
    at fixed parameters stay in a consistent envelope. That is a useful smoke test for
    the simulator, and the only thing this result may be read as.
    """
    rng = np.random.default_rng(seed)
    channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]

    # Generate "real" data
    mapping = SimMapping(
        subsystem="reaction_wheel",
        fault_params=(FaultParameter(param_name, param_value),),
    )
    twin = twin_cls()
    twin.configure(mapping)
    real = twin.run(duration_s=2000, seed=int(rng.integers(0, 100_000)))

    # Generate predicted data under the same parameters
    sim_stats_all: list[list[float]] = []
    stat_names: list[str] = []

    for i in range(n_sims):
        sim = twin.run(duration_s=2000, seed=int(rng.integers(0, 100_000)))
        stats = []
        names = []
        for ch in channels:
            s = sim[ch].to_numpy()
            stats.extend([np.mean(s), np.std(s)])
            if not stat_names:
                names.extend([f"{ch}_mean", f"{ch}_std"])
        sim_stats_all.append(stats)
        if not stat_names:
            stat_names = names

    sim_stats_arr = np.array(sim_stats_all)

    # Real data stats
    real_stats = []
    for ch in channels:
        r = real[ch].to_numpy()
        real_stats.extend([np.mean(r), np.std(r)])

    # Check coverage
    pred_mean = sim_stats_arr.mean(axis=0).tolist()
    pred_std = sim_stats_arr.std(axis=0).tolist()

    coverages = []
    for i, (r_val, p_mean, p_std) in enumerate(zip(real_stats, pred_mean, pred_std)):
        low = p_mean - 2 * p_std
        high = p_mean + 2 * p_std
        coverages.append(1.0 if low <= r_val <= high else 0.0)

    # Pass if >= 80% of statistics are covered
    coverage_frac = float(np.mean(coverages))
    passed = bool(coverage_frac >= 0.8)

    return PPCResult(
        family=family,
        summary_stat_names=stat_names,
        real_stats=real_stats,
        predicted_stats_mean=pred_mean,
        predicted_stats_std=pred_std,
        coverage_fractions=coverages,
        passed=passed,
        diagnostics={"overall_coverage": float(coverage_frac)},
    )


def _prior_draw(posterior: Any) -> Callable[[np.random.Generator], float]:
    """The posterior object's own prior sampler, falling back to uniform support.

    ``SyntheticLikelihoodPosterior`` and the NPE posteriors carry a uniform prior
    described by ``prior_low``/``prior_high``, which is the fallback. A posterior
    whose prior is not uniform supplies ``prior_sample`` instead — without this hook,
    calibration silently probes the wrong prior and even an exact posterior looks
    miscalibrated.
    """
    explicit = getattr(posterior, "prior_sample", None)
    if callable(explicit):
        return lambda rng: float(explicit(rng))

    low = float(posterior.prior_low)
    high = float(posterior.prior_high)
    return lambda rng: float(rng.uniform(low, high))


def _pit_uniformity_p(pit_values: np.ndarray, n_bins: int) -> float:
    """Chi-squared uniformity p-value for PIT values, on a coarse fixed grid.

    Reuses the SBC rank machinery by mapping each PIT onto one of ``n_bins`` bins.
    Coarse bins keep the chi-squared approximation honest inside a stratum, where the
    trial count is a fraction of the total.
    """
    ranks = np.clip((np.asarray(pit_values, dtype=float) * n_bins).astype(int), 0, n_bins - 1)
    return float(uniformity_p_value(ranks, n_posterior_samples=n_bins - 1))


def posterior_predictive_pit(
    posterior: Any,
    n_trials: int = 200,
    n_predictive_draws: int = 39,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Probability-integral-transform values for a posterior predictive distribution.

    For each trial: draw θ* from the prior, simulate an observation x_obs ~ p(x|θ*)
    **and an independent replicate** x_rep ~ p(x|θ*) at the same θ*, condition the
    posterior on x_obs, push its draws back through the simulator, and record where
    x_rep falls in that predictive sample.

    The replicate is what makes this a real check. Scoring x_obs inside a predictive
    distribution that was itself conditioned on x_obs uses the data twice, which is
    conservative and non-uniform even for an exact posterior — the standard objection
    to naive posterior-predictive p-values. An independent replicate drawn from the
    same ground truth removes the double-use, so an exact posterior yields uniform
    PIT values and the check has an unambiguous null.

    Returns:
        ``(pit, observations)`` — a ``(n_trials, n_stats)`` array of PIT values in
        [0, 1], one column per summary statistic returned by ``posterior.simulate``,
        and the matching ``(n_trials, n_stats)`` observations. The observations are
        returned because marginal PIT uniformity is not sufficient: they are what the
        conditional check in :func:`run_ppc_with_posterior` stratifies on.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_predictive_draws < 1:
        raise ValueError("n_predictive_draws must be >= 1")

    rng = np.random.default_rng(seed)
    draw_theta = _prior_draw(posterior)

    rows: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    for _ in range(n_trials):
        theta_true = draw_theta(rng)
        observed = np.asarray(posterior.simulate(theta_true, rng), dtype=float).ravel()
        replicate = np.asarray(posterior.simulate(theta_true, rng), dtype=float).ravel()
        observations.append(observed)

        thetas = np.asarray(
            posterior.sample(observed, n_predictive_draws, rng), dtype=float
        ).ravel()
        if thetas.size != n_predictive_draws:
            raise ValueError(
                f"posterior.sample returned {thetas.size} draws, "
                f"expected {n_predictive_draws}"
            )

        predictive = np.array(
            [
                np.asarray(posterior.simulate(float(t), rng), dtype=float).ravel()
                for t in thetas
            ]
        )

        below = np.count_nonzero(predictive < replicate[None, :], axis=0)
        tied = np.count_nonzero(predictive == replicate[None, :], axis=0)
        # Randomised tie-breaking, matching sbc_ranks, keeps PIT uniform under
        # discrete or duplicated draws.
        jitter = np.array([int(rng.integers(0, t + 1)) if t else 0 for t in tied])
        rows.append((below + jitter) / float(n_predictive_draws))

    return np.array(rows), np.array(observations)


def run_ppc_with_posterior(
    posterior: Any,
    n_trials: int = 200,
    n_predictive_draws: int = 39,
    seed: int = 42,
    family: str = "bearing_friction_increase",
    alpha: float = 0.01,
    interval: float = 0.95,
    n_strata: int = 4,
    pit_bins: int = 5,
    stat_names: Sequence[str] | None = None,
) -> PPCResult:
    """Native posterior predictive check — a function of the posterior under test.

    Unlike :func:`run_ppc`, every quantity here is generated *through* the posterior:
    its draws choose the parameters the simulator runs at, so a posterior that ignores
    the observation, sits in the wrong place, or carries far too much spread produces a
    predictive distribution that misplaces the replicate.

    ``passed`` requires PIT uniformity **conditional on the observation**, not merely
    marginal uniformity. That distinction is the whole check. A posterior that returns
    the prior regardless of the data is *marginally* perfect — ranking a prior-predictive
    replicate among prior-predictive draws is uniform by construction — so a pooled test
    accepts it, exactly as rank-uniformity alone accepts a prior-returning posterior in
    SBC. Stratifying by the observation exposes it: in the strata where the observation
    ran high, an unresponsive predictive sits too low, and vice versa, errors that cancel
    when pooled and do not cancel within a stratum.

    ``coverage_fractions`` carries the interpretable companion: how often the replicate
    landed inside the central ``interval`` of the predictive sample, which should sit
    near ``interval`` itself.

    **Known limit on power.** Simulator noise is common to the true and the estimated
    predictive distribution, so it dilutes posterior error: a posterior ten times too
    *narrow* still yields a predictive only slightly narrower than the truth, and this
    check will not reliably reject it. Overconfidence is SBC's and sharpness's job (see
    :func:`run_sbc_with_posterior` and :func:`posterior_sharpness`); a predictive check
    answers the complementary question of whether simulating at the inferred parameters
    reproduces the observed data at all. The gate's three legs are complementary rather
    than redundant, and none is sufficient alone.
    """
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must lie in (0, 1)")
    if n_strata < 1:
        raise ValueError("n_strata must be >= 1")

    pit, observations = posterior_predictive_pit(
        posterior,
        n_trials=n_trials,
        n_predictive_draws=n_predictive_draws,
        seed=seed,
    )
    n_stats = pit.shape[1]
    names = (
        list(stat_names) if stat_names is not None else [f"stat_{i}" for i in range(n_stats)]
    )

    tail = (1.0 - interval) / 2.0
    coverage_fractions = [
        float(np.mean((pit[:, i] >= tail) & (pit[:, i] <= 1.0 - tail)))
        for i in range(n_stats)
    ]

    # Marginal uniformity — reported, but never the pass criterion on its own.
    marginal_p = [_pit_uniformity_p(pit[:, i], n_bins=pit_bins) for i in range(n_stats)]

    # Conditional uniformity: within each stratum of the observation, the replicate
    # must still land at a uniform quantile. This is the criterion the prior cannot
    # satisfy — a predictive that does not move with the observation is misplaced in
    # every stratum, even though those errors cancel when pooled.
    conditional_p: list[float] = []
    for i in range(n_stats):
        order = np.argsort(observations[:, i], kind="stable")
        for chunk in np.array_split(order, n_strata):
            if chunk.size >= pit_bins:
                conditional_p.append(_pit_uniformity_p(pit[chunk, i], n_bins=pit_bins))

    all_p = marginal_p + conditional_p
    # Bonferroni across every test performed. Chosen for a stable null: with dozens of
    # correlated statistics an uncorrected threshold rejects a perfect posterior often
    # enough to make the gate flap. Strictness comes from requiring all three legs of
    # the gate plus sharpness, not from this one threshold.
    corrected_alpha = alpha / max(len(all_p), 1)
    passed = bool(min(all_p) > corrected_alpha)

    return PPCResult(
        family=family,
        summary_stat_names=names,
        # These three carry PIT summaries, not telemetry values: this check spans many
        # observations, so there is no single "real" summary statistic to report. Mean
        # PIT observed, the 0.5 a calibrated posterior should produce, and the observed
        # spread. Units are quantiles in [0, 1].
        real_stats=[float(np.mean(pit[:, i])) for i in range(n_stats)],
        predicted_stats_mean=[0.5] * n_stats,
        predicted_stats_std=[float(np.std(pit[:, i])) for i in range(n_stats)],
        coverage_fractions=coverage_fractions,
        passed=passed,
        diagnostics={
            "method": "replicate_pit_posterior_predictive",
            "posterior": type(posterior).__name__,
            "n_trials": n_trials,
            "n_predictive_draws": n_predictive_draws,
            "interval": interval,
            "alpha": alpha,
            "bonferroni_alpha": corrected_alpha,
            "n_strata": n_strata,
            "pit_bins": pit_bins,
            "n_tests": len(all_p),
            "marginal_pit_p_values": [round(p, 6) for p in marginal_p],
            "min_marginal_pit_p": round(float(min(marginal_p)), 6),
            "min_conditional_pit_p": (
                round(float(min(conditional_p)), 6) if conditional_p else None
            ),
            "overall_coverage": round(float(np.mean(coverage_fractions)), 4),
        },
    )


def run_sbc_with_posterior(
    posterior: Any,
    n_prior_samples: int = 200,
    n_sims: int = 19,
    seed: int = 42,
    family: str = "bearing_friction_increase",
    alpha: float = 0.01,
) -> SBCResult:
    """Run SBC against an explicit amortized posterior object.

    ``posterior`` supplies its own prior support, forward simulator and sampler
    (``prior_low``/``prior_high``, ``simulate``, ``sample``), so this works for any
    inference method — synthetic likelihood, NPE, or anything added later — without
    calibration.py knowing which one it is.

    Unlike :func:`run_sbc`, no simulation happens at inference time: an amortized
    posterior is trained once, so SBC costs one forward simulation per trial.
    """
    low = float(posterior.prior_low)
    high = float(posterior.prior_high)
    per_trial_sharpness: list[float] = []
    # Same hook as the predictive check: probe the posterior's own prior, not an
    # assumed-uniform one. Unchanged for uniform-prior posteriors.
    prior_sampler = _prior_draw(posterior)

    def simulator(theta: float, rng: np.random.Generator) -> Any:
        return posterior.simulate(theta, rng)

    def posterior_sampler(observed: Any, n: int, rng: np.random.Generator) -> np.ndarray:
        draws = np.asarray(posterior.sample(observed, n, rng), dtype=float)
        per_trial_sharpness.append(posterior_sharpness(draws, low, high))
        return draws

    ranks = sbc_ranks(
        prior_sampler=prior_sampler,
        simulator=simulator,
        posterior_sampler=posterior_sampler,
        n_trials=n_prior_samples,
        n_posterior_samples=n_sims,
        seed=seed,
    )

    p_value = uniformity_p_value(ranks, n_sims)
    sharpness = float(np.mean(per_trial_sharpness)) if per_trial_sharpness else 0.0

    return SBCResult(
        family=family,
        n_prior_samples=n_prior_samples,
        n_sims_per_sample=n_sims,
        rank_histogram=rank_histogram(ranks, n_sims),
        uniformity_p_value=p_value,
        passed=bool(p_value > alpha),
        sharpness=round(sharpness, 4),
        diagnostics={
            "method": "amortized_posterior_sbc",
            "posterior": type(posterior).__name__,
            "alpha": alpha,
            "prior_low": low,
            "prior_high": high,
        },
    )


MIN_SHARPNESS = 0.5
"""Posterior must be at least twice as narrow as the prior to count as informative."""


def derive_calibration_status(
    domain: str,
    sbc: SBCResult,
    ppc: PPCResult,
    method: str = "SBC+PPC",
    min_sharpness: float = MIN_SHARPNESS,
) -> CalibrationStatus:
    """Reduce real SBC + PPC results to a domain-agnostic ``CalibrationStatus``.

    The ``confidence`` field is *derived from the calibration diagnostics* — never
    self-reported by the scorer — which is the property the AutonomyGate relies on.

    ``passed`` requires SBC (rank-uniformity), PPC (predictive coverage), **and**
    sufficient sharpness. The sharpness term is not optional rigour: rank-uniformity
    is trivially satisfied by a posterior that returns the prior, so without it a
    scorer can buy autonomy by knowing nothing — measured on the reaction-wheel twin,
    the ABC posterior did exactly that, passing SBC only at ~85% of prior width.

        with sharpness:     confidence = 0.4*ppc + 0.4*sbc_score + 0.2*sharpness
        without (legacy):   confidence = 0.5*ppc + 0.5*sbc_score
        sbc_score = min(1.0, uniformity_p_value / 0.05)          # in [0, 1]

    ``sharpness=None`` means it was never measured, so the legacy two-term formula
    applies unchanged. Fails closed on empty coverage.
    """
    coverage = float(np.mean(ppc.coverage_fractions)) if ppc.coverage_fractions else 0.0
    sbc_uniformity_score = min(1.0, float(sbc.uniformity_p_value) / 0.05)

    diagnostics = {
        "sbc_passed": bool(sbc.passed),
        "sbc_uniformity_p_value": round(float(sbc.uniformity_p_value), 4),
        "ppc_passed": bool(ppc.passed),
        "ppc_coverage": round(coverage, 4),
        "sbc_rank_histogram": list(sbc.rank_histogram),
        "sbc_method": sbc.diagnostics.get("method"),
        "posterior": sbc.diagnostics.get("posterior"),
    }

    if sbc.sharpness is None:
        confidence = round(0.5 * coverage + 0.5 * sbc_uniformity_score, 4)
        passed = bool(sbc.passed and ppc.passed)
        diagnostics["sharpness"] = None
        diagnostics["sharpness_passed"] = None
    else:
        sharpness = float(sbc.sharpness)
        sharpness_passed = bool(sharpness >= min_sharpness)
        confidence = round(0.4 * coverage + 0.4 * sbc_uniformity_score + 0.2 * sharpness, 4)
        passed = bool(sbc.passed and ppc.passed and sharpness_passed)
        diagnostics["sharpness"] = round(sharpness, 4)
        diagnostics["sharpness_passed"] = sharpness_passed
        diagnostics["min_sharpness"] = min_sharpness
    return CalibrationStatus(
        domain=domain,
        passed=passed,
        confidence=confidence,
        method=method,
        diagnostics=diagnostics,
    )
