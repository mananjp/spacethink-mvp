"""Exploration layer — Detector protocol + baseline implementations.

Per the literature review: the winning pattern for the MVP is a lightweight,
well-understood detector (LSTM-forecaster + dynamic thresholding lineage, or a
simple robust z-score for v0) rather than feeding raw telemetry to an LLM.
ThresholdDetector below is the deterministic stub used by other tracks' tests;
ZScoreDetector is the first real, still-simple baseline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from domain import EventOfInterest, Severity, new_id


class Detector(Protocol):
    def detect(self, df: pd.DataFrame, channels: list[str]) -> list[EventOfInterest]: ...


class ThresholdDetector:
    """Deterministic stub for tests/CI — flags nothing, always returns []."""

    name = "stub_threshold"

    def detect(self, df: pd.DataFrame, channels: list[str]) -> list[EventOfInterest]:
        return []


class ZScoreDetector:
    """Baseline-referenced robust z-score detector.

    Each channel is compared against robust statistics (median / MAD) of a
    *nominal baseline* taken from the start of the run, rather than a short
    trailing window. This matters:

    - A trailing rolling window lags the telemetry's normal sinusoidal structure
      and crosses the threshold constantly, so v0 flagged ~150 spurious events per
      run (it was detecting the oscillation, not faults).
    - A trailing window also *adapts to* a slow drift (e.g. bearing friction) and
      stops flagging it — the exact fault we most want to catch.

    A fixed nominal baseline avoids both: normal oscillation stays within the
    baseline's spread, while sustained level shifts and sharp events read as large
    robust-z deviations. Adjacent flags are grouped into events, blips separated by
    less than ``merge_gap`` are merged, and events shorter than ``min_len`` dropped.

    Assumes the run begins with a nominal stretch (fault, if any, starts after the
    baseline). That holds for the synthetic reaction-wheel data and is the same
    assumption the signature scorer makes; real telemetry gets the LSTM-forecaster
    swap planned for Phase 1b.
    """

    name = "zscore_v1"

    def __init__(
        self,
        baseline_frac: float = 0.2,
        min_baseline: int = 200,
        z_thresh: float = 6.0,
        merge_gap: int = 300,
        min_len: int = 1,
        window: int | None = None,  # accepted for backward compatibility; unused
    ):
        self.baseline_frac = baseline_frac
        self.min_baseline = min_baseline
        self.z_thresh = z_thresh
        self.merge_gap = merge_gap
        self.min_len = min_len
        # `window` was the old trailing-window size; the baseline-referenced
        # detector no longer uses it. Kept so existing call sites don't break.
        self.window = window

    def _robust_z(self, series: np.ndarray, baseline_n: int) -> np.ndarray:
        """|value - baseline median| / (1.4826 * baseline MAD)."""
        baseline = series[:baseline_n]
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        scale = max(1.4826 * mad, 1e-9)
        return np.abs((series - median) / scale)

    def _group_events(self, flagged: np.ndarray) -> list[tuple[int, int]]:
        """Contiguous flagged runs, merged across gaps <= merge_gap, filtered by min_len."""
        raw: list[list[int]] = []
        i, n = 0, len(flagged)
        while i < n:
            if flagged[i]:
                j = i
                while j < n and flagged[j]:
                    j += 1
                raw.append([i, j])
                i = j
            else:
                i += 1
        merged: list[list[int]] = []
        for start, end in raw:
            if merged and start - merged[-1][1] <= self.merge_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged if e - s >= self.min_len]

    def detect(
        self, df: pd.DataFrame, channels: list[str], run_id: str = "local"
    ) -> list[EventOfInterest]:
        events: list[EventOfInterest] = []
        base_time = datetime.now(timezone.utc)
        n = len(df)
        baseline_n = min(max(self.min_baseline, int(n * self.baseline_frac)), n // 2)
        if baseline_n < 2:
            return events

        for ch in channels:
            series = df[ch].to_numpy(dtype=float)
            z = self._robust_z(series, baseline_n)
            flagged = z > self.z_thresh

            for start_idx, end_idx in self._group_events(flagged):
                score = float(z[start_idx:end_idx].max())
                severity = (
                    Severity.HIGH if score > 12 else Severity.MEDIUM if score > 8 else Severity.LOW
                )
                events.append(
                    EventOfInterest(
                        id=new_id(),
                        run_id=run_id,
                        channel=ch,
                        start_ts=base_time + timedelta(seconds=int(start_idx)),
                        end_ts=base_time + timedelta(seconds=int(end_idx)),
                        score=score,
                        severity=severity,
                        detector_name=self.name,
                        # start_idx/end_idx let downstream stages / the dashboard
                        # locate the event window in the original series.
                        metadata={
                            "z_thresh": self.z_thresh,
                            "baseline_n": baseline_n,
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                        },
                    )
                )
        return events
