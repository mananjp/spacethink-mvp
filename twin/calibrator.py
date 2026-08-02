"""Twin Calibrator — auto-calibrate parametric twin from customer telemetry.

Fidelity tiers:
  Tier 1 — parametric, auto-calibrated, common subsystems (default)
  Tier 2 — Sedaro / TrueTwin import path for high-fidelity

Cold-start motion: "first 10 anomalies re-diagnosed" for design partners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from domain import CalibratedTwinParams, FaultParameter, SimMapping
from twin.simulator import ToySimulator


@dataclass
class CalibrationResult:
    """Result of twin calibration from customer telemetry."""
    customer_id: str
    params: CalibratedTwinParams
    fit_error: float
    n_iterations: int
    converged: bool


def calibrate_twin(
    customer_id: str,
    telemetry: pd.DataFrame,
    channels: list[str] | None = None,
    subsystem: str = "reaction_wheel",
    twin_cls: type = ToySimulator,
    n_iterations: int = 50,
    seed: int = 42,
) -> CalibrationResult:
    """Auto-calibrate a parametric twin from customer telemetry.

    Uses a simple grid search + gradient-free optimization to find the
    fault parameters that best reproduce the observed telemetry statistics.

    Parameters
    ----------
    customer_id : Unique customer identifier
    telemetry : Customer telemetry DataFrame with standard channel contract
    channels : Channels to calibrate against
    subsystem : Subsystem to calibrate
    twin_cls : Twin class to calibrate (default ToySimulator)
    n_iterations : Maximum optimization iterations
    seed : Random seed
    """
    channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
    rng = np.random.default_rng(seed)

    # Extract target statistics from real telemetry
    target_stats = {}
    for ch in channels:
        if ch in telemetry.columns:
            series = telemetry[ch].to_numpy(dtype=float)
            target_stats[ch] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "trend": float(np.polyfit(np.arange(len(series)), series, 1)[0]),
            }

    # Parameter search space
    param_ranges = {
        "friction": (0.0, 2.0),
        "dropout_rate": (0.0, 0.1),
        "stiction_rate": (0.0, 0.05),
    }

    best_params = {"friction": 0.0, "dropout_rate": 0.0, "stiction_rate": 0.0}
    best_error = float("inf")
    duration = min(len(telemetry), 2000)

    for iteration in range(n_iterations):
        # Perturb parameters
        candidate = {}
        for param, (low, high) in param_ranges.items():
            if iteration == 0:
                candidate[param] = (low + high) / 2
            else:
                # Gaussian perturbation around current best
                scale = (high - low) * max(0.1, 1.0 - iteration / n_iterations)
                candidate[param] = np.clip(
                    best_params[param] + rng.normal(0, scale),
                    low, high,
                )

        # Run twin with candidate parameters
        mapping = SimMapping(
            subsystem=subsystem,
            fault_params=tuple(
                FaultParameter(name, value) for name, value in candidate.items()
            ),
        )
        twin = twin_cls()
        twin.configure(mapping)
        sim = twin.run(duration_s=duration, seed=seed + iteration)

        # Compare statistics
        error = 0.0
        for ch in channels:
            if ch in sim.columns and ch in target_stats:
                sim_series = sim[ch].to_numpy()
                sim_stats = {
                    "mean": float(np.mean(sim_series)),
                    "std": float(np.std(sim_series)),
                    "trend": float(np.polyfit(np.arange(len(sim_series)), sim_series, 1)[0]),
                }
                for stat_name in ["mean", "std", "trend"]:
                    target_val = target_stats[ch][stat_name]
                    sim_val = sim_stats[stat_name]
                    scale = max(abs(target_val), 1e-6)
                    error += ((target_val - sim_val) / scale) ** 2

        if error < best_error:
            best_error = error
            best_params = candidate.copy()

    converged = best_error < 1.0

    calibrated = CalibratedTwinParams(
        customer_id=customer_id,
        subsystem=subsystem,
        fidelity_tier=1,
        parameters=best_params,
    )

    return CalibrationResult(
        customer_id=customer_id,
        params=calibrated,
        fit_error=best_error,
        n_iterations=n_iterations,
        converged=converged,
    )


def rediagnose_with_calibrated_twin(
    telemetry: pd.DataFrame,
    calibrated_params: CalibratedTwinParams,
    twin_cls: type = ToySimulator,
    n_sims: int = 10,
) -> list[pd.DataFrame]:
    """Re-run simulations using calibrated twin parameters.

    Used for the "first 10 anomalies re-diagnosed" cold-start motion.
    """
    mapping = SimMapping(
        subsystem=calibrated_params.subsystem,
        fault_params=tuple(
            FaultParameter(name, value)
            for name, value in calibrated_params.parameters.items()
        ),
    )
    twin = twin_cls()
    twin.configure(mapping)
    duration = min(len(telemetry), 5000)
    return twin.run_ensemble(n_sims=n_sims, duration_s=duration)
