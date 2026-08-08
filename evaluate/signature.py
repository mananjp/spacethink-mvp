"""Fault-signature extraction — the physics link between detection and diagnosis.

The v0 scorer compared *raw* telemetry series and was dominated by noise, so it
could not tell one fault from another (and "diagnosed" faults on nominal data).
A fault signature is instead a small, physically meaningful feature vector that
describes *how a channel deviates from its own pre-fault baseline* during the
event window:

  - level shift and trend per channel (a slow current/temperature rise with a
    speed droop  ->  bearing friction),
  - fraction of near-zero speed reads               ->  encoder dropout,
  - rate of sharp speed step-downs paired with       ->  stiction.
    current spikes

Both the real event and each hypothesis's twin simulation are reduced to the
*same* signature, so scoring compares fault shapes, not noisy absolute values.

Normalization uses fixed, domain-meaningful scales (not each series' own std)
so signatures stay comparable even when the real data and the twin have
different noise levels — the systematic deviation is what carries the signal,
and it survives averaging over the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPEED = "wheel_speed_rpm"
CURRENT = "wheel_current_a"
TEMP = "wheel_temp_c"

# Absolute normalization scales (roughly one "meaningful unit" per channel).
CHANNEL_SCALES = {SPEED: 100.0, CURRENT: 0.1, TEMP: 5.0}

# Thresholds for the discrete event-shape features.
_STICTION_STEP_RPM = 200.0  # a speed drop larger than this in one step
_CURRENT_SPIKE_A = 0.2  # current this far above baseline = a spike

# Rare-event rates are passed through sqrt(rate) * gain, not rate * gain. The
# concave transform (a) lifts these tiny rates (0.001–0.02) onto the same scale
# as the level/trend features so they carry weight in the distance, and (b) makes
# *presence* matter more than exact count — a real fault with only a few sparse
# events still reads as that fault, not as nominal, even though the twin ensemble
# produces a stronger, more consistent signature.
_ZERO_FRAC_GAIN = 4.0
_STICTION_GAIN = 8.0
_SPIKE_GAIN = 6.0

FEATURE_NAMES: tuple[str, ...] = (
    "speed_dlevel",
    "speed_trend",
    "speed_zero_frac",
    "speed_stiction_rate",
    "current_dlevel",
    "current_trend",
    "current_spike_rate",
    "temp_dlevel",
    "temp_trend",
)


def _trend(win: np.ndarray, scale: float) -> float:
    """Second-half minus first-half mean — a noise-robust monotonic-trend proxy."""
    if win.size < 4:
        return 0.0
    half = win.size // 2
    return float((win[half:].mean() - win[:half].mean()) / scale)


def extract_signature(window: pd.DataFrame, baseline: pd.DataFrame) -> np.ndarray:
    """Reduce a telemetry window (vs. a pre-event baseline) to a fault signature.

    ``window`` is the anomalous segment; ``baseline`` is nominal telemetry from
    before the event (empty baseline falls back to the window's own early stats).
    Returns a fixed-length float vector aligned with ``FEATURE_NAMES``.
    """
    s = window[SPEED].to_numpy(dtype=float)
    c = window[CURRENT].to_numpy(dtype=float)
    tp = window[TEMP].to_numpy(dtype=float)

    def _base_mean(col: str, fallback: np.ndarray) -> float:
        b = baseline[col].to_numpy(dtype=float) if len(baseline) else fallback
        return float(b.mean()) if b.size else float(fallback.mean())

    speed_base = _base_mean(SPEED, s)
    current_base = _base_mean(CURRENT, c)
    temp_base = _base_mean(TEMP, tp)

    ss, cs, ts = CHANNEL_SCALES[SPEED], CHANNEL_SCALES[CURRENT], CHANNEL_SCALES[TEMP]

    # Speed: droop level + trend, dropout-to-zero fraction, stiction step-downs.
    zero_thresh = 0.5 * speed_base
    speed_dlevel = (s.mean() - speed_base) / ss
    speed_trend = _trend(s, ss)
    speed_zero_frac = (
        np.sqrt(float(np.mean(s < zero_thresh))) * _ZERO_FRAC_GAIN if s.size else 0.0
    )
    if s.size >= 2:
        ds = np.diff(s)
        not_dropout = s[1:] > zero_thresh  # exclude drops *to* zero (that's dropout)
        stiction_raw = float(np.mean((ds < -_STICTION_STEP_RPM) & not_dropout))
        speed_stiction_rate = np.sqrt(stiction_raw) * _STICTION_GAIN
    else:
        speed_stiction_rate = 0.0

    # Current: rise level + trend (friction), spike fraction (stiction).
    current_dlevel = (c.mean() - current_base) / cs
    current_trend = _trend(c, cs)
    current_spike_rate = (
        np.sqrt(float(np.mean(c > current_base + _CURRENT_SPIKE_A))) * _SPIKE_GAIN
        if c.size
        else 0.0
    )

    # Temperature: rise level + trend (friction heating).
    temp_dlevel = (tp.mean() - temp_base) / ts
    temp_trend = _trend(tp, ts)

    return np.array(
        [
            speed_dlevel,
            speed_trend,
            speed_zero_frac,
            speed_stiction_rate,
            current_dlevel,
            current_trend,
            current_spike_rate,
            temp_dlevel,
            temp_trend,
        ],
        dtype=float,
    )


def split_baseline_window(
    df: pd.DataFrame, baseline_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a full run into (baseline, window) for extracting a simulated
    signature: the first ``baseline_frac`` is treated as nominal, the rest as
    the region where a developing fault would show up.
    """
    n = len(df)
    cut = max(1, int(n * baseline_frac))
    return df.iloc[:cut], df.iloc[cut:]
