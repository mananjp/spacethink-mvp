"""Calibration guardrails — Simulation-Based Calibration (SBC) and Posterior Predictive Checks (PPC).

A scorer that fails SBC or PPC is blocked, not deployed. Diagnostics land
in runstore next to every SimResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from domain import CalibrationStatus, FaultParameter, Hypothesis, SimMapping
from twin.simulator import ToySimulator


@dataclass
class SBCResult:
    """Result of Simulation-Based Calibration check."""
    family: str
    n_prior_samples: int
    n_sims_per_sample: int
    rank_histogram: list[int]
    uniformity_p_value: float
    passed: bool
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


def run_sbc(
    scorer: object,
    twin_cls: type = ToySimulator,
    family: str = "bearing_friction_increase",
    param_name: str = "friction",
    prior_low: float = 0.1,
    prior_high: float = 2.0,
    n_prior_samples: int = 100,
    n_sims: int = 20,
    channels: list[str] | None = None,
    duration_s: int = 1000,
    seed: int = 42,
) -> SBCResult:
    """Run Simulation-Based Calibration for a scorer/family pair.

    SBC checks whether the *scorer's* implied posterior is well-calibrated by:
    1. Sampling theta* from the prior and simulating "real" data x* ~ p(x|theta*).
    2. Drawing candidate parameters theta_j from the prior and simulating x_j.
    3. Asking THE SCORER how well each candidate explains x* (its ``distance``).
    4. Ranking theta* among the candidates and checking the ranks are uniform.

    The scorer is what is under test: a well-calibrated scorer yields uniform ranks;
    a miscalibrated one does not. (The earlier implementation ranked by a hardcoded
    distance and ignored ``scorer`` entirely, so training the scorer could never move
    this result — see docs/SBC_SCORER_CONSULTATION_SPEC.md.)
    """
    rng = np.random.default_rng(seed)
    channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
    ranks: list[int] = []

    def _simulate(theta: float) -> pd.DataFrame:
        mapping = SimMapping(
            subsystem="reaction_wheel",
            fault_params=(FaultParameter(param_name, float(theta)),),
        )
        twin = twin_cls()
        twin.configure(mapping)
        return twin.run(duration_s=duration_s, seed=int(rng.integers(0, 100_000)))

    def _scorer_distance(theta: float, real: pd.DataFrame) -> float:
        """How poorly parameter ``theta`` explains ``real``, AS JUDGED BY THE SCORER."""
        sim = _simulate(theta)
        hyp = Hypothesis(
            id=f"sbc-{family}-{theta:.6g}",
            event_id="sbc",
            text=f"{family}: {param_name}={theta:.6g}",
            mechanism=family,
            fault_params=(FaultParameter(param_name, float(theta)),),
            prior=1.0,
            generator="sbc",
        )
        return float(scorer.score(hyp, real, [sim]).distance)

    for _ in range(n_prior_samples):
        theta_true = rng.uniform(prior_low, prior_high)
        real = _simulate(theta_true)

        # Lower scorer distance = better fit. Rank theta* by how many candidates the
        # scorer judges to fit x* better than the truth does.
        true_dist = _scorer_distance(theta_true, real)
        rank = sum(
            1
            for _ in range(n_sims)
            if _scorer_distance(rng.uniform(prior_low, prior_high), real) < true_dist
        )
        ranks.append(rank)

    # Build rank histogram and test uniformity
    histogram, _ = np.histogram(ranks, bins=n_sims + 1, range=(0, n_sims))
    histogram_list = histogram.tolist()

    # Chi-squared test for uniformity (simplified normal approximation)
    expected = n_prior_samples / (n_sims + 1)
    chi2 = sum((o - expected) ** 2 / expected for o in histogram_list)
    dof = n_sims
    z = (chi2 - dof) / np.sqrt(2 * dof)
    from scipy.stats import norm
    p_value = float(2 * (1 - norm.cdf(abs(z))))

    passed = bool(p_value > 0.01)  # Reject if p < 0.01

    return SBCResult(
        family=family,
        n_prior_samples=n_prior_samples,
        n_sims_per_sample=n_sims,
        rank_histogram=histogram_list,
        uniformity_p_value=p_value,
        passed=passed,
        diagnostics={
            "chi2": chi2,
            "dof": dof,
            "z": z,
            "scorer": getattr(scorer, "name", type(scorer).__name__),
        },
    )


def run_ppc(
    scorer: object,
    twin_cls: type = ToySimulator,
    family: str = "bearing_friction_increase",
    param_name: str = "friction",
    param_value: float = 0.6,
    prior_low: float = 0.1,
    prior_high: float = 2.0,
    n_sims: int = 50,
    channels: list[str] | None = None,
    duration_s: int = 2000,
    seed: int = 42,
) -> PPCResult:
    """Run a Posterior Predictive Check for a scorer/family pair.

    Draws candidate parameters from the prior, simulates each, and weights them by
    how well THE SCORER says they explain the real data (self-normalized importance
    sampling with weight exp(-distance), matching ``normalize_posteriors``). The
    weighted ensemble is the scorer's posterior-predictive distribution; PPC passes
    when the real summary statistics fall inside its 95% band.

    Like SBC, this now depends on the scorer: a scorer that concentrates weight on
    well-fitting parameters yields a tight, well-centered band. (The earlier
    implementation formed the band from the twin alone and ignored ``scorer``.) A
    scorer exposing a native posterior sampler could replace the prior proposal with
    true posterior draws; that refinement is deferred until a trained SBIScorer exists
    to validate it — see docs/SBC_SCORER_CONSULTATION_SPEC.md.
    """
    rng = np.random.default_rng(seed)
    channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]

    def _simulate(theta: float) -> pd.DataFrame:
        mapping = SimMapping(
            subsystem="reaction_wheel",
            fault_params=(FaultParameter(param_name, float(theta)),),
        )
        twin = twin_cls()
        twin.configure(mapping)
        return twin.run(duration_s=duration_s, seed=int(rng.integers(0, 100_000)))

    def _summary(df: pd.DataFrame) -> tuple[list[float], list[str]]:
        stats: list[float] = []
        names: list[str] = []
        for ch in channels:
            s = df[ch].to_numpy()
            stats.extend([float(np.mean(s)), float(np.std(s))])
            names.extend([f"{ch}_mean", f"{ch}_std"])
        return stats, names

    # "Real" data under the true parameter.
    real = _simulate(param_value)
    real_stats, stat_names = _summary(real)

    # Posterior-predictive ensemble: prior draws weighted by the scorer's fit to real.
    sim_stats_all: list[list[float]] = []
    log_weights: list[float] = []
    for _ in range(n_sims):
        theta_j = rng.uniform(prior_low, prior_high)
        sim_j = _simulate(theta_j)
        stats_j, _ = _summary(sim_j)
        sim_stats_all.append(stats_j)
        hyp = Hypothesis(
            id=f"ppc-{family}-{theta_j:.6g}",
            event_id="ppc",
            text=f"{family}: {param_name}={theta_j:.6g}",
            mechanism=family,
            fault_params=(FaultParameter(param_name, float(theta_j)),),
            prior=1.0,
            generator="ppc",
        )
        log_weights.append(-float(scorer.score(hyp, real, [sim_j]).distance))

    sim_stats_arr = np.array(sim_stats_all)
    # Self-normalized importance weights: w_j proportional to exp(-distance_j).
    lw = np.array(log_weights)
    lw -= lw.max()
    weights = np.exp(lw)
    weights /= weights.sum()

    pred_mean = (weights[:, None] * sim_stats_arr).sum(axis=0)
    pred_var = (weights[:, None] * (sim_stats_arr - pred_mean) ** 2).sum(axis=0)
    pred_std = np.sqrt(pred_var)

    coverages = []
    for r_val, p_mean, p_std in zip(real_stats, pred_mean.tolist(), pred_std.tolist()):
        low = p_mean - 2 * p_std
        high = p_mean + 2 * p_std
        coverages.append(1.0 if low <= r_val <= high else 0.0)

    coverage_frac = float(np.mean(coverages)) if coverages else 0.0
    passed = bool(coverage_frac >= 0.8)

    return PPCResult(
        family=family,
        summary_stat_names=stat_names,
        real_stats=real_stats,
        predicted_stats_mean=pred_mean.tolist(),
        predicted_stats_std=pred_std.tolist(),
        coverage_fractions=coverages,
        passed=passed,
        diagnostics={
            "overall_coverage": float(coverage_frac),
            "scorer": getattr(scorer, "name", type(scorer).__name__),
        },
    )


def derive_calibration_status(
    domain: str,
    sbc: SBCResult,
    ppc: PPCResult,
    method: str = "SBC+PPC",
) -> CalibrationStatus:
    """Reduce real SBC + PPC results to a domain-agnostic ``CalibrationStatus``.

    The ``confidence`` field is *derived from the calibration diagnostics* — never
    self-reported by the scorer — which is the property the AutonomyGate relies on:

        confidence = 0.5 * ppc_coverage + 0.5 * sbc_uniformity_score
        sbc_uniformity_score = min(1.0, uniformity_p_value / 0.05)   # in [0, 1]

    ``passed`` requires *both* SBC (posterior rank-uniformity) and PPC (predictive
    coverage) to pass; a scorer that is not rank-calibrated fails here regardless of
    how confident it claims to be. Fails closed on empty coverage.
    """
    coverage = float(np.mean(ppc.coverage_fractions)) if ppc.coverage_fractions else 0.0
    sbc_uniformity_score = min(1.0, float(sbc.uniformity_p_value) / 0.05)
    confidence = round(0.5 * coverage + 0.5 * sbc_uniformity_score, 4)
    passed = bool(sbc.passed and ppc.passed)
    diagnostics = {
        "sbc_passed": bool(sbc.passed),
        "sbc_uniformity_p_value": round(float(sbc.uniformity_p_value), 4),
        "ppc_passed": bool(ppc.passed),
        "ppc_coverage": round(coverage, 4),
        "sbc_rank_histogram": list(sbc.rank_histogram),
    }
    return CalibrationStatus(
        domain=domain,
        passed=passed,
        confidence=confidence,
        method=method,
        diagnostics=diagnostics,
    )
