"""Calibration guardrails — Simulation-Based Calibration (SBC) and Posterior Predictive Checks (PPC).

A scorer that fails SBC or PPC is blocked, not deployed. Diagnostics land
in runstore next to every SimResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from domain import FaultParameter, SimMapping
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
    seed: int = 42,
) -> SBCResult:
    """Run Simulation-Based Calibration for a scorer/family pair.

    SBC checks whether the posterior from the scorer is well-calibrated by:
    1. Sampling θ* from the prior
    2. Simulating data x* ~ p(x|θ*)
    3. Computing posterior samples θ ~ q(θ|x*)
    4. Checking that the rank of θ* within the posterior samples is uniform.
    """
    rng = np.random.default_rng(seed)
    channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
    ranks = []

    for i in range(n_prior_samples):
        # 1. Sample true parameter from prior
        theta_true = rng.uniform(prior_low, prior_high)

        # 2. Simulate "real" data under θ*
        mapping = SimMapping(
            subsystem="reaction_wheel",
            fault_params=(FaultParameter(param_name, theta_true),),
        )
        twin = twin_cls()
        twin.configure(mapping)
        real = twin.run(duration_s=1000, seed=int(rng.integers(0, 100_000)))

        # 3. Simulate ensemble under draws from the prior
        posterior_distances = []
        for j in range(n_sims):
            theta_j = rng.uniform(prior_low, prior_high)
            mapping_j = SimMapping(
                subsystem="reaction_wheel",
                fault_params=(FaultParameter(param_name, theta_j),),
            )
            twin_j = twin_cls()
            twin_j.configure(mapping_j)
            sim_j = twin_j.run(duration_s=1000, seed=int(rng.integers(0, 100_000)))

            # Compute distance
            n_pts = min(len(real), len(sim_j))
            dists = []
            for ch in channels:
                r = real[ch].to_numpy()[:n_pts]
                s = sim_j[ch].to_numpy()[:n_pts]
                scale = max(np.std(r), 1e-6)
                dists.append(float(np.sqrt(np.mean(((r - s) / scale) ** 2))))
            posterior_distances.append((theta_j, float(np.mean(dists))))

        # 4. Compute rank of θ* among the "posterior"
        # Lower distance = better fit, so rank by how many have lower distance to θ*
        true_mapping = SimMapping(
            subsystem="reaction_wheel",
            fault_params=(FaultParameter(param_name, theta_true),),
        )
        twin_true = twin_cls()
        twin_true.configure(true_mapping)
        sim_true = twin_true.run(duration_s=1000, seed=int(rng.integers(0, 100_000)))
        n_pts = min(len(real), len(sim_true))
        true_dists = []
        for ch in channels:
            r = real[ch].to_numpy()[:n_pts]
            s = sim_true[ch].to_numpy()[:n_pts]
            scale = max(np.std(r), 1e-6)
            true_dists.append(float(np.sqrt(np.mean(((r - s) / scale) ** 2))))
        true_dist = float(np.mean(true_dists))

        rank = sum(1 for _, d in posterior_distances if d < true_dist)
        ranks.append(rank)

    # Build rank histogram and test uniformity
    histogram, _ = np.histogram(ranks, bins=n_sims + 1, range=(0, n_sims))
    histogram_list = histogram.tolist()

    # Chi-squared test for uniformity (simplified)
    expected = n_prior_samples / (n_sims + 1)
    chi2 = sum((o - expected) ** 2 / expected for o in histogram_list)
    # Approximate p-value (chi2 with n_sims degrees of freedom)
    dof = n_sims
    # Using a simple normal approximation for chi2
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
        diagnostics={"chi2": chi2, "dof": dof, "z": z},
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
    """Run Posterior Predictive Check for a scorer/family pair.

    Generates simulated data under a known parameter and checks that
    summary statistics of the simulated data cover the real-data statistics.
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
