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


class SignatureScorer:
    """Fault-signature scorer — the fix for the diagnosis loop.

    DistanceScorer compares whole raw telemetry series and is dominated by noise,
    so it cannot discriminate the fault mechanisms (and "diagnoses" faults on
    nominal data). SignatureScorer instead compares fault *signatures* (see
    evaluate/signature.py): the real event's signature vs. each hypothesis's
    twin-simulated signature. Kept ALONGSIDE DistanceScorer (which the
    counterfactual/verifier/CI paths still depend on), not a replacement.

    Different call shape: score(hyp, real_sig, sim_sigs) takes pre-computed
    signature vectors, not raw DataFrames — the planner extracts them.
    """

    name = "signature_v1"

    def score(self, hyp: Hypothesis, real_sig: np.ndarray, sim_sigs: list[np.ndarray]) -> SimResult:
        mean_sig = np.mean(np.stack(sim_sigs), axis=0) if sim_sigs else np.zeros_like(real_sig)
        distance = float(np.linalg.norm(real_sig - mean_sig))
        return SimResult(
            hypothesis_id=hyp.id,
            distance=distance,
            posterior=0.0,  # filled in by normalize_posteriors across all hypotheses
            n_sims=len(sim_sigs),
            diagnostics={
                "real_signature": [round(x, 4) for x in real_sig.tolist()],
                "sim_signature": [round(x, 4) for x in mean_sig.tolist()],
            },
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
