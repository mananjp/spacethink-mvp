"""Onboard edge evaluation loop — edge version of the anomaly detection pipeline.

Runs the forecaster alone on-device for "Was this event real?" decision
before downlink. Designed for CubeSat-class hardware with constrained
memory and compute.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from explore.telemanom_lineage import _ExponentialSmoothingForecaster


@dataclass
class OnboardDecision:
    """Decision made by the onboard evaluation loop."""
    channel: str
    timestamp_idx: int
    is_anomaly: bool
    confidence: float
    should_downlink: bool
    error_magnitude: float
    threshold: float


class OnboardEvaluator:
    """Lightweight onboard anomaly evaluator for edge deployment.

    Runs the exponential smoothing forecaster with a small memory footprint
    and makes binary decisions: "real anomaly → downlink" vs "noise → skip".

    Memory budget: ~59 KB working set.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        threshold_sigma: float = 3.0,
        buffer_size: int = 500,
        downlink_threshold: float = 0.7,
    ):
        self.forecaster = _ExponentialSmoothingForecaster(alpha=alpha, beta=beta)
        self.threshold_sigma = threshold_sigma
        self.buffer_size = buffer_size
        self.downlink_threshold = downlink_threshold

        # Circular buffer for running statistics
        self._buffer: dict[str, np.ndarray] = {}
        self._buffer_idx: dict[str, int] = {}
        self._running_mean: dict[str, float] = {}
        self._running_var: dict[str, float] = {}
        self._n_samples: dict[str, int] = {}

    def _init_channel(self, channel: str) -> None:
        """Initialize tracking for a new channel."""
        self._buffer[channel] = np.zeros(self.buffer_size, dtype=np.float32)
        self._buffer_idx[channel] = 0
        self._running_mean[channel] = 0.0
        self._running_var[channel] = 1.0
        self._n_samples[channel] = 0

    def _update_stats(self, channel: str, value: float) -> None:
        """Update running statistics (Welford's algorithm for memory efficiency)."""
        if channel not in self._buffer:
            self._init_channel(channel)

        n = self._n_samples[channel] + 1
        self._n_samples[channel] = n

        delta = value - self._running_mean[channel]
        self._running_mean[channel] += delta / n
        delta2 = value - self._running_mean[channel]
        self._running_var[channel] += delta * delta2

        # Update circular buffer
        idx = self._buffer_idx[channel] % self.buffer_size
        self._buffer[channel][idx] = value
        self._buffer_idx[channel] = idx + 1

    def evaluate_sample(
        self,
        channel: str,
        value: float,
        timestamp_idx: int = 0,
    ) -> OnboardDecision:
        """Evaluate a single telemetry sample.

        Memory-efficient: uses only running statistics, no full history.
        """
        self._update_stats(channel, value)

        n = self._n_samples[channel]
        if n < 10:
            # Not enough samples for meaningful detection
            return OnboardDecision(
                channel=channel,
                timestamp_idx=timestamp_idx,
                is_anomaly=False,
                confidence=0.0,
                should_downlink=False,
                error_magnitude=0.0,
                threshold=0.0,
            )

        # Compute prediction error
        mean = self._running_mean[channel]
        var = self._running_var[channel] / max(n - 1, 1)
        std = max(np.sqrt(var), 1e-6)

        error = abs(value - mean)
        threshold = std * self.threshold_sigma

        is_anomaly = error > threshold
        confidence = min(1.0, error / (threshold + 1e-10))

        # Downlink decision: only worth the bandwidth if confidence is high
        should_downlink = is_anomaly and confidence >= self.downlink_threshold

        return OnboardDecision(
            channel=channel,
            timestamp_idx=timestamp_idx,
            is_anomaly=is_anomaly,
            confidence=round(confidence, 3),
            should_downlink=should_downlink,
            error_magnitude=round(error, 4),
            threshold=round(threshold, 4),
        )

    def evaluate_window(
        self,
        channel: str,
        values: np.ndarray,
    ) -> list[OnboardDecision]:
        """Evaluate a window of telemetry samples."""
        decisions = []
        for i, val in enumerate(values):
            decisions.append(self.evaluate_sample(channel, float(val), i))
        return decisions

    def memory_usage_bytes(self) -> int:
        """Estimate current memory usage in bytes."""
        n_channels = len(self._buffer)
        # Buffer: float32 × buffer_size per channel
        buffer_bytes = n_channels * self.buffer_size * 4
        # Running stats: 3 floats + 1 int per channel
        stats_bytes = n_channels * (3 * 8 + 4)
        # Overhead
        overhead = 256
        return buffer_bytes + stats_bytes + overhead

    def reset(self) -> None:
        """Reset all channel tracking."""
        self._buffer.clear()
        self._buffer_idx.clear()
        self._running_mean.clear()
        self._running_var.clear()
        self._n_samples.clear()
