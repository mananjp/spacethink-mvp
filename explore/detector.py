"""Exploration layer — Detector protocol + baseline implementations.

Per the literature review: the winning pattern for the MVP is a lightweight,
well-understood detector (LSTM-forecaster + dynamic thresholding lineage, or
simple z-score/EWMA for v0) rather than feeding raw telemetry to an LLM.
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
    def detect(self, df: pd.DataFrame, channels: list[str]) -> list[EventOfInterest]:
        ...


class ThresholdDetector:
    """Deterministic stub for tests/CI — flags nothing, always returns []."""

    name = "stub_threshold"

    def detect(self, df: pd.DataFrame, channels: list[str]) -> list[EventOfInterest]:
        return []


class ZScoreDetector:
    """Rolling z-score detector with dynamic thresholding.

    Simple, explainable, and — per ESA-ADB findings — a reasonable floor
    baseline before swapping in an LSTM forecaster in Phase 1b.
    """

    name = "zscore_v0"

    def __init__(self, window: int = 200, z_thresh: float = 3.5):
        self.window = window
        self.z_thresh = z_thresh

    def detect(self, df: pd.DataFrame, channels: list[str], run_id: str = "local") -> list[EventOfInterest]:
        events: list[EventOfInterest] = []
        base_time = datetime.now(timezone.utc)

        for ch in channels:
            series = df[ch].to_numpy(dtype=float)
            roll_mean = pd.Series(series).rolling(self.window, min_periods=self.window // 2).mean()
            roll_std = pd.Series(series).rolling(self.window, min_periods=self.window // 2).std().replace(0, np.nan)
            z = (pd.Series(series) - roll_mean) / roll_std
            flagged = z.abs() > self.z_thresh

            in_event = False
            start_idx = None
            for i, is_flag in enumerate(flagged.fillna(False)):
                if is_flag and not in_event:
                    in_event = True
                    start_idx = i
                elif not is_flag and in_event:
                    in_event = False
                    end_idx = i
                    score = float(z.iloc[start_idx:end_idx].abs().max())
                    severity = Severity.HIGH if score > 6 else Severity.MEDIUM if score > 4.5 else Severity.LOW
                    events.append(EventOfInterest(
                        id=new_id(),
                        run_id=run_id,
                        channel=ch,
                        start_ts=base_time + timedelta(seconds=int(start_idx)),
                        end_ts=base_time + timedelta(seconds=int(end_idx)),
                        score=score,
                        severity=severity,
                        detector_name=self.name,
                        metadata={"window": self.window, "z_thresh": self.z_thresh},
                    ))
            if in_event:
                end_idx = len(series) - 1
                score = float(z.iloc[start_idx:end_idx].abs().max())
                severity = Severity.HIGH if score > 6 else Severity.MEDIUM if score > 4.5 else Severity.LOW
                events.append(EventOfInterest(
                    id=new_id(),
                    run_id=run_id,
                    channel=ch,
                    start_ts=base_time + timedelta(seconds=int(start_idx)),
                    end_ts=base_time + timedelta(seconds=int(end_idx)),
                    score=score,
                    severity=severity,
                    detector_name=self.name,
                    metadata={"window": self.window, "z_thresh": self.z_thresh},
                ))
        return events
