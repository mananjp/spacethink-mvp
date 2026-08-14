"""Gaussian synthetic-likelihood posterior over a scalar fault parameter.

The method (Wood 2010) is simulation-based inference without neural networks: for a
grid of parameter values, run the twin repeatedly and record the *mean and spread* of
its summary statistics. That spread is the simulator's own noise, measured rather
than assumed, so the resulting posterior is as wide as the data genuinely warrant —
no arbitrary distance cutoff to trade calibration against sharpness.

Inference is amortized: training simulates once, after which scoring an observation
is a table lookup and costs no simulation at all.

Deliberately pure numpy/scipy. Hermetic CI must pass with neither torch nor sbi
installed, and this is the path that runs when they are absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from domain import FaultParameter, SimMapping
from evaluate.sbi_scorer import extract_summary_stats
from twin.simulator import ToySimulator

DEFAULT_CHANNELS = ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]

# Floor on any estimated per-feature standard deviation, as a fraction of that
# feature's spread across the grid. Without it, a feature that happens to look
# noiseless in a small sample dominates the likelihood and forces overconfidence.
_SD_FLOOR_FRACTION = 1e-3


@dataclass(frozen=True)
class SyntheticLikelihoodPosterior:
    """Amortized posterior p(theta | summary statistics) on a fixed parameter grid.

    ``grid`` holds cell *centres*, so a half-cell jitter at sampling time stays
    inside the prior support without clipping (clipping would pile probability mass
    on the boundaries and corrupt the SBC rank statistics).
    """

    grid: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    prior_low: float
    prior_high: float
    param_name: str = "friction"
    duration_s: float = 400
    channels: tuple[str, ...] = tuple(DEFAULT_CHANNELS)

    @property
    def cell_width(self) -> float:
        return (self.prior_high - self.prior_low) / len(self.grid)

    def simulate(self, theta: float, rng: np.random.Generator) -> np.ndarray:
        """Draw one observation from the twin at ``theta``, as summary statistics."""
        frame = simulate_frame(
            theta,
            param_name=self.param_name,
            duration_s=self.duration_s,
            seed=int(rng.integers(0, 100_000)),
        )
        return extract_summary_stats(frame, list(self.channels))

    def log_likelihood(self, stats: np.ndarray) -> np.ndarray:
        """Gaussian log-likelihood of ``stats`` at every grid point."""
        observed = np.asarray(stats, dtype=float).ravel()
        z = (observed[None, :] - self.means) / self.sds
        return -0.5 * np.sum(z**2 + 2.0 * np.log(self.sds), axis=1)

    def log_posterior(self, stats: np.ndarray) -> np.ndarray:
        """Unnormalized log posterior. The prior is uniform, so it is the likelihood."""
        return self.log_likelihood(stats)

    def weights(self, stats: np.ndarray) -> np.ndarray:
        """Normalized posterior probability over the grid."""
        log_p = self.log_posterior(stats)
        shifted = np.exp(log_p - log_p.max())
        total = shifted.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(len(self.grid), 1.0 / len(self.grid))
        return shifted / total

    def sample(self, stats: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` posterior samples: grid cell by weight, then uniform within it."""
        weights = self.weights(stats)
        idx = rng.choice(len(self.grid), size=n, p=weights)
        half = self.cell_width / 2.0
        return self.grid[idx] + rng.uniform(-half, half, size=n)


def simulate_frame(
    theta: float,
    param_name: str = "friction",
    duration_s: float = 400,
    seed: int = 0,
    twin_cls: type = ToySimulator,
) -> pd.DataFrame:
    """Run the twin once under a single scalar fault parameter."""
    twin = twin_cls()
    twin.configure(
        SimMapping(
            subsystem="reaction_wheel",
            fault_params=(FaultParameter(param_name, float(theta)),),
        )
    )
    return twin.run(duration_s=duration_s, seed=seed)


def train_synthetic_likelihood(
    prior_low: float = 0.1,
    prior_high: float = 2.0,
    n_grid: int = 32,
    n_reps: int = 6,
    duration_s: float = 400,
    seed: int = 42,
    param_name: str = "friction",
    channels: list[str] | None = None,
    twin_cls: type = ToySimulator,
) -> SyntheticLikelihoodPosterior:
    """Estimate summary-statistic mean and spread across a grid of parameter values.

    ``n_reps`` repeats per grid point estimate the simulator noise. The spread is
    inflated by sqrt(1 + 1/n_reps) because a future observation must be compared
    against an *estimated* mean, not the true one — the correction that keeps a
    small ``n_reps`` from manufacturing overconfidence.
    """
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    if n_reps < 2:
        raise ValueError("n_reps must be >= 2 to estimate a spread")
    if prior_high <= prior_low:
        raise ValueError("prior_high must exceed prior_low")

    channels = list(channels or DEFAULT_CHANNELS)
    rng = np.random.default_rng(seed)

    width = (prior_high - prior_low) / n_grid
    grid = prior_low + (np.arange(n_grid) + 0.5) * width

    means = []
    sds = []
    for theta in grid:
        reps = np.array(
            [
                extract_summary_stats(
                    simulate_frame(
                        theta,
                        param_name=param_name,
                        duration_s=duration_s,
                        seed=int(rng.integers(0, 1_000_000)),
                        twin_cls=twin_cls,
                    ),
                    channels,
                )
                for _ in range(n_reps)
            ]
        )
        means.append(reps.mean(axis=0))
        sds.append(reps.std(axis=0, ddof=1))

    means_arr = np.array(means)
    sds_arr = np.array(sds) * np.sqrt(1.0 + 1.0 / n_reps)

    # Floor each feature's noise relative to how much that feature moves across the
    # grid, so a feature that looks noiseless by chance cannot dominate.
    signal_scale = means_arr.std(axis=0)
    floor = np.maximum(signal_scale * _SD_FLOOR_FRACTION, 1e-12)
    sds_arr = np.maximum(sds_arr, floor[None, :])

    return SyntheticLikelihoodPosterior(
        grid=grid,
        means=means_arr,
        sds=sds_arr,
        prior_low=prior_low,
        prior_high=prior_high,
        param_name=param_name,
        duration_s=duration_s,
        channels=tuple(channels),
    )
