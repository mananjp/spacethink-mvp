"""SBI Scorer — Simulation-Based Inference scorer implementing the Scorer protocol.

Uses amortized Neural Posterior Estimation (NPE) per fault family when the ``sbi``
package is installed. Falls back to a lightweight kernel-density-based scorer
that still provides meaningful posterior estimates without neural networks.

Two components:
1. Amortized NPE per fault family (sbi v0.26.x)
2. Amortized Bayesian model comparison across hypotheses (neural classifier / BayesFlow)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from domain import Hypothesis, SimResult

try:
    import sbi  # type: ignore[import-untyped]
    from sbi.inference import SNPE  # type: ignore[import-untyped]

    _HAS_SBI = True
except ImportError:
    _HAS_SBI = False

# Path for pre-trained models
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "sbi"


def _extract_summary_stats(df: pd.DataFrame, channels: list[str] | None = None) -> np.ndarray:
    """Extract summary statistics from a telemetry DataFrame.

    Returns a 1-D feature vector: [mean, std, skew, kurtosis, trend_slope]
    per channel (5 × n_channels features).
    """
    channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
    features = []
    for ch in channels:
        series = df[ch].to_numpy(dtype=float)
        features.extend(
            [
                np.mean(series),
                np.std(series),
                float(_safe_skew(series)),
                float(_safe_kurtosis(series)),
                float(_trend_slope(series)),
            ]
        )
    return np.array(features)


#: Public alias — other inference modules build on the same summary statistics.
extract_summary_stats = _extract_summary_stats


def _safe_skew(x: np.ndarray) -> float:
    """Skewness without scipy dependency."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3) * n * n / ((n - 1) * (n - 2)))


def _safe_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis without scipy dependency."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _trend_slope(x: np.ndarray) -> float:
    """Simple linear trend (least-squares slope)."""
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    t -= t.mean()
    return float(np.dot(t, x - x.mean()) / (np.dot(t, t) + 1e-10))


class SBIScorer:
    """Simulation-Based Inference scorer.

    When ``sbi`` is installed and pre-trained posteriors exist, uses amortized
    NPE for instant posterior evaluation. Otherwise falls back to a
    summary-statistic kernel-density comparison.
    """

    name = "sbi_v1"

    def __init__(
        self,
        channels: list[str] | None = None,
        models_dir: Path = MODELS_DIR,
        bandwidth: float = 1.0,
    ):
        self.channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
        self.models_dir = models_dir
        self.bandwidth = bandwidth
        self._posterior_cache: dict[str, Any] = {}

        # Try to load pre-trained posteriors
        if _HAS_SBI and models_dir.exists():
            self._load_posteriors()

    def _load_posteriors(self) -> None:
        """Load pre-trained amortized posteriors from disk."""
        for family_dir in self.models_dir.iterdir():
            if family_dir.is_dir():
                posterior_path = family_dir / "posterior.pkl"
                if posterior_path.exists():
                    import pickle

                    with open(posterior_path, "rb") as f:
                        self._posterior_cache[family_dir.name] = pickle.load(f)

    def score(
        self, hyp: Hypothesis, real: pd.DataFrame, simulated: list[pd.DataFrame]
    ) -> SimResult:
        """Score a hypothesis against real telemetry using SBI or fallback."""
        if _HAS_SBI and hyp.mechanism in self._posterior_cache:
            return self._score_npe(hyp, real)
        return self._score_kernel(hyp, real, simulated)

    def _score_npe(self, hyp: Hypothesis, real: pd.DataFrame) -> SimResult:
        """Score using pre-trained amortized NPE posterior."""
        posterior = self._posterior_cache[hyp.mechanism]
        obs = _extract_summary_stats(real, self.channels)

        import torch  # type: ignore[import-untyped]

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        samples = posterior.sample((1000,), x=obs_tensor)

        # Use log-probability as a distance metric (higher = closer)
        log_prob = float(posterior.log_prob(samples.mean(0).unsqueeze(0), x=obs_tensor).item())
        distance = max(0.0, -log_prob)  # Convert to distance

        return SimResult(
            hypothesis_id=hyp.id,
            distance=distance,
            posterior=0.0,  # Filled in by normalize_posteriors
            n_sims=1000,
            diagnostics={
                "method": "amortized_npe",
                "log_prob": log_prob,
                "posterior_mean": samples.mean(0).tolist(),
                "posterior_std": samples.std(0).tolist(),
            },
        )

    def _score_kernel(
        self, hyp: Hypothesis, real: pd.DataFrame, simulated: list[pd.DataFrame]
    ) -> SimResult:
        """Fallback scoring using summary-statistic kernel density comparison.

        Computes Gaussian kernel density in summary-statistic space between
        real telemetry and simulated ensembles.
        """
        real_stats = _extract_summary_stats(real, self.channels)
        sim_stats = np.array([_extract_summary_stats(s, self.channels) for s in simulated])

        # Normalize features
        combined = np.vstack([real_stats.reshape(1, -1), sim_stats])
        mu = combined.mean(axis=0)
        sigma = combined.std(axis=0)
        sigma[sigma < 1e-10] = 1.0

        real_norm = (real_stats - mu) / sigma
        sim_norm = (sim_stats - mu) / sigma

        # Gaussian kernel density estimate at the real observation
        diffs = sim_norm - real_norm
        sq_dists = np.sum(diffs**2, axis=1)
        kernel_vals = np.exp(-sq_dists / (2 * self.bandwidth**2))
        log_density = float(np.log(kernel_vals.mean() + 1e-30))

        distance = float(np.mean(np.sqrt(sq_dists)))

        return SimResult(
            hypothesis_id=hyp.id,
            distance=distance,
            posterior=0.0,  # Filled in by normalize_posteriors
            n_sims=len(simulated),
            diagnostics={
                "method": "kernel_density",
                "log_density": log_density,
                "mean_sq_dist": float(np.mean(sq_dists)),
                "bandwidth": self.bandwidth,
            },
        )
