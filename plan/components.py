"""Selectable pipeline components — detectors and scorers by name.

Lets the CLI / harness pick a detector and scorer without the planner importing
heavy optional dependencies (torch for the Telemanom lineage, sbi for NPE) at
module load. Each entry is a zero-arg factory imported lazily, so the default
path stays hermetic and fast, and the advanced components load only when asked.

Defaults are the proven path from the core-diagnosis fix:
    detector = "zscore"   (baseline-referenced robust z-score)
    scorer   = "signature" (event-window fault-signature distance)

The alternatives are wired in but must be *measured* before becoming default
(see docs/NEXT_STEPS.md) — SBIScorer/Telemanom score/detect differently and may
or may not beat the default on a given dataset.
"""

from __future__ import annotations

from typing import Callable

DEFAULT_DETECTOR = "zscore"
DEFAULT_SCORER = "signature"


def _zscore():
    from explore.detector import ZScoreDetector

    return ZScoreDetector()


def _telemanom():
    # Optional torch dependency; falls back to exponential-smoothing internally.
    from explore.telemanom_lineage import TelemanomLineageDetector

    return TelemanomLineageDetector()


def _signature():
    from evaluate.scorer import SignatureScorer

    return SignatureScorer()


def _distance():
    from evaluate.scorer import DistanceScorer

    return DistanceScorer()


def _sbi():
    # Optional sbi dependency; falls back to a kernel scorer internally.
    from evaluate.sbi_scorer import SBIScorer

    return SBIScorer()


DETECTORS: dict[str, Callable[[], object]] = {
    "zscore": _zscore,
    "telemanom": _telemanom,
}

SCORERS: dict[str, Callable[[], object]] = {
    "signature": _signature,
    "distance": _distance,
    "sbi": _sbi,
}


def build_detector(name: str | None):
    """Instantiate a detector by name (None -> default)."""
    factory = DETECTORS.get(name or DEFAULT_DETECTOR)
    if factory is None:
        raise ValueError(f"unknown detector '{name}'. options: {sorted(DETECTORS)}")
    return factory()


def build_scorer(name: str | None):
    """Instantiate a scorer by name (None -> default)."""
    factory = SCORERS.get(name or DEFAULT_SCORER)
    if factory is None:
        raise ValueError(f"unknown scorer '{name}'. options: {sorted(SCORERS)}")
    return factory()
