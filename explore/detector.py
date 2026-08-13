"""Exploration layer — Detector protocol + baseline implementations.

Per the literature review: the winning pattern for the MVP is a lightweight,
well-understood detector (LSTM-forecaster + dynamic thresholding lineage, or a
simple robust z-score for v0) rather than feeding raw telemetry to an LLM.
ThresholdDetector below is the deterministic stub used by other tracks' tests;
ZScoreDetector is the first real, still-simple baseline.
"""

from __future__ import annotations

import warnings
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

    "Normal oscillation stays within the baseline's spread" only holds if the baseline
    is long enough to *contain* a full oscillation, and a single run-length fraction
    cannot guarantee that across channels of different speeds. Sizing every baseline
    at 20% of the run spanned 2.00 periods of wheel speed and current but only 0.50
    periods of temperature, so the baseline never observed the thermal trough and the
    entire negative lobe of normal behaviour read as a level shift — 1720 points
    flagged per nominal run, at max robust-z 10.8, on every seed. Event-wise detection
    precision was 58.7% with 11 false alarms across the nominal runs.

    Each channel's baseline is therefore sized independently, by
    :meth:`baseline_length`, and validated against held-out nominal data rather than
    assumed adequate. Precision on the same suite is 100% with 0 false alarms, recall
    unchanged at 100%.

    Real telemetry sharpens this rather than softening it: orbital (~90 min) and
    thermal cycles run far slower than the attitude-control channels sharing a frame.

    Assumes the run begins with a nominal stretch (fault, if any, starts after the
    baseline). That holds for the synthetic reaction-wheel data and is the same
    assumption the signature scorer makes; real telemetry gets the LSTM-forecaster
    swap planned for Phase 1b. When no baseline in that region can be validated the
    channel is *under-characterised*: the detector warns, falls back to the
    conservative default budget, and marks the resulting events — see :meth:`detect`
    for why it does not simply abstain.
    """

    name = "zscore_v1"

    #: Fraction of the nominal region a baseline may occupy. The remainder is held
    #: out to validate it, so a baseline is never checked against itself.
    _HELD_OUT_SPLIT = 0.8
    #: Candidate baseline lengths tried between the default budget and that ceiling.
    _BASELINE_SEARCH_STEPS = 12

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

    def _default_budget(self, n: int) -> int:
        """Baseline length before any validation: a fixed fraction of the run."""
        return min(max(self.min_baseline, int(n * self.baseline_frac)), n // 2)

    def baseline_length(self, series: np.ndarray) -> int:
        """Longest baseline that still explains the rest of the presumed-nominal region.

        Candidates run from the default budget up to a ceiling that always leaves a
        held-out tail, and the largest whose held-out remainder stays under
        ``z_thresh`` wins. More of normal is strictly better for the robust statistics
        — seeing too little of it is precisely what caused the temperature regression —
        but growth is bounded by evidence rather than taken on faith, so a baseline is
        never assumed to generalise. Withholding a tail also stops a baseline being
        validated against itself, which every baseline would trivially pass.

        This is the invariant the detector actually needs: "normal, as characterised,
        covers all of normal", tested on unseen nominal data.

        Estimating each channel's period and demanding the baseline span one cycle was
        the obvious route and does not survive contact with the data. The periodogram
        cannot distinguish a period near the record length from one far beyond it —
        both peak in the lowest resolvable bin — and mean-crossing counts are dominated
        by noise rather than by the oscillation: 133 crossings in a 600-sample window
        whose true period is 2000. Held-out error needs no period at all.

        Returns 0 when no candidate validates, which :meth:`detect` treats as
        under-characterised rather than as a reason to abstain.
        """
        n = len(series)
        cap = n // 2  # presumed-nominal region; the fault, if any, starts after it
        if cap < 4:
            return 0

        budget = min(self._default_budget(n), cap - 1)
        if budget < 2:
            return 0

        longest = max(budget, min(int(cap * self._HELD_OUT_SPLIT), cap - 1))
        region = series[:cap]

        for candidate in sorted(
            set(np.linspace(budget, longest, self._BASELINE_SEARCH_STEPS).astype(int)),
            reverse=True,
        ):
            if candidate < 2:
                continue
            held_out = self._robust_z(region, int(candidate))[int(candidate) :]
            if held_out.size and float(held_out.max()) <= self.z_thresh:
                return int(candidate)

        return 0

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

        for ch in channels:
            series = df[ch].to_numpy(dtype=float)
            baseline_n = self.baseline_length(series)
            validated = baseline_n >= 2

            if not validated:
                # Two causes are indistinguishable from inside the region: the channel
                # oscillates more slowly than the region is long, or a fault already
                # started within it. Abstaining would silently drop every real fault of
                # the second kind, so fall back to the conservative default budget and
                # make the reduced confidence visible instead of losing recall.
                baseline_n = self._default_budget(len(series))
                if baseline_n < 2:
                    continue
                warnings.warn(
                    f"channel {ch!r} is under-characterised: no baseline inside the "
                    f"presumed-nominal region explains the rest of it, so 'normal' for "
                    f"this channel rests on an unvalidated {baseline_n}-sample default. "
                    f"Its events carry lower confidence; supply a longer nominal stretch.",
                    UserWarning,
                    stacklevel=2,
                )

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
                            # The warning is transient; this rides with the event, so
                            # downstream stages can tell a validated baseline from a
                            # fallback one long after detection.
                            "baseline_validated": validated,
                        },
                    )
                )
        return events
