"""Detection-evaluation harness — honest event-wise scoring of the *detector*.

Separate from `evaluate/harness.py` (which scores end-to-end *diagnosis*). This
scores only the exploration stage: does the detector flag the labelled anomaly
regions, without crying wolf on nominal data? Reported as range-based
precision / recall / F0.5 (never point-adjusted — see `detection_metrics.py`).

Works on any `TelemetrySource`, so the same command evaluates the synthetic
generator today and real ESA-ADB / OPS-SAT-AD data once a `CsvTelemetrySource` is
pointed at it:

    python -m evaluate.detection_harness            # synthetic
    # from real data: CsvTelemetrySource("data/esa_adb", "data/esa_adb/labels.csv")
"""

from __future__ import annotations

import pandas as pd

from evaluate.detection_metrics import (
    DetectionScore,
    Interval,
    aggregate_scores,
    merge_intervals,
    score_detection,
)
from explore.detector import Detector, ZScoreDetector
from ingest.sources import LabelledRun, SyntheticSource, TelemetrySource

# Columns that are time axes, not telemetry channels.
_TIME_COLUMNS = {"t", "time", "timestamp", "ts"}


def infer_channels(telemetry: pd.DataFrame) -> list[str]:
    """Numeric telemetry columns (everything except a time axis)."""
    return [
        c
        for c in telemetry.columns
        if c.lower() not in _TIME_COLUMNS
        and pd.api.types.is_numeric_dtype(telemetry[c])
    ]


def _detected_intervals(events: list) -> list[Interval]:
    intervals = [
        (int(e.metadata["start_idx"]), int(e.metadata["end_idx"]))
        for e in events
        if "start_idx" in e.metadata and "end_idx" in e.metadata
    ]
    return merge_intervals(intervals)


def evaluate_detection(
    source: TelemetrySource | None = None,
    detector: Detector | None = None,
    beta: float = 0.5,
) -> dict:
    """Score a detector over all runs from a source. Returns the corpus score plus
    per-run detection/label counts.
    """
    source = source or SyntheticSource()
    detector = detector or ZScoreDetector()

    per_run: list[dict] = []
    scores: list[DetectionScore] = []
    for run in source.runs():
        run_score = _score_run(run, detector, beta)
        scores.append(run_score)
        per_run.append(
            {
                "run_id": run.run_id,
                "n_detected": run_score.n_detected,
                "n_labelled": run_score.n_labelled,
                "recalled": run_score.recalled,
                "false_alarms": run_score.n_detected - run_score.tp_detections,
            }
        )

    corpus = aggregate_scores(scores, beta=beta)
    return {"corpus": corpus.as_dict(), "per_run": per_run}


def _score_run(run: LabelledRun, detector: Detector, beta: float) -> DetectionScore:
    channels = infer_channels(run.telemetry)
    events = detector.detect(run.telemetry, channels, run_id=run.run_id)  # type: ignore[call-arg]
    detected = _detected_intervals(events)
    return score_detection(detected, run.anomaly_intervals, beta=beta)


def format_report(result: dict) -> str:
    c = result["corpus"]
    nominal_false_alarms = sum(
        r["false_alarms"] for r in result["per_run"] if r["n_labelled"] == 0
    )
    lines = [
        "=" * 60,
        "spaceThink DETECTION evaluation (event-wise, range-based)",
        "=" * 60,
        f"precision : {c['precision']:.1%}   ({c['tp_detections']}/{c['n_detected']} detections hit a real anomaly)",
        f"recall    : {c['recall']:.1%}   ({c['recalled']}/{c['n_labelled']} labelled anomalies caught)",
        f"F{c['beta']:g}      : {c['fbeta']:.3f}",
        f"false alarms on nominal runs: {nominal_false_alarms}",
        "(never point-adjusted; a detection counts only if it overlaps a labelled anomaly)",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    print(format_report(evaluate_detection()))
