"""Testing/Scoring layer — Scorer protocol + a distance-based SBI-lite scorer.

Compares real telemetry against twin-simulated ensembles per hypothesis and
turns distances into a normalized posterior across competing hypotheses.
This is a simplified stand-in for the amortized NPE approach in the research
(sbi package) — good enough to prove the closed loop end-to-end on synthetic
data before investing in full simulation-based inference.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from domain import Hypothesis, SimResult


class Scorer(Protocol):
    def score(self, hyp: Hypothesis, real: pd.DataFrame, simulated: list[pd.DataFrame]) -> SimResult:
        ...


class DistanceScorer:
    """Mean per-channel normalized RMSE between real telemetry and an ensemble
    of simulated runs under a hypothesis's fault parameters.
    """

    name = "distance_v0"

    def __init__(self, channels: list[str] | None = None):
        self.channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]

    def _distance(self, real: pd.DataFrame, sim: pd.DataFrame) -> float:
        n = min(len(real), len(sim))
        dists = []
        for ch in self.channels:
            r = real[ch].to_numpy()[:n]
            s = sim[ch].to_numpy()[:n]
            scale = max(np.std(r), 1e-6)
            dists.append(float(np.sqrt(np.mean(((r - s) / scale) ** 2))))
        return float(np.mean(dists))

    def score(self, hyp: Hypothesis, real: pd.DataFrame, simulated: list[pd.DataFrame]) -> SimResult:
        distances = [self._distance(real, sim) for sim in simulated]
        mean_dist = float(np.mean(distances))
        return SimResult(
            hypothesis_id=hyp.id,
            distance=mean_dist,
            posterior=0.0,  # filled in by normalize_posteriors across all hypotheses
            n_sims=len(simulated),
            diagnostics={"distances": distances},
        )


def normalize_posteriors(results: list[SimResult]) -> list[SimResult]:
    """Convert distances into a softmax-style posterior (lower distance -> higher belief)."""
    if not results:
        return results
    dists = np.array([r.distance for r in results])
    weights = np.exp(-dists)
    weights /= weights.sum()
    return [
        SimResult(
            hypothesis_id=r.hypothesis_id,
            distance=r.distance,
            posterior=float(w),
            n_sims=r.n_sims,
            diagnostics=r.diagnostics,
        )
        for r, w in zip(results, weights)
    ]
