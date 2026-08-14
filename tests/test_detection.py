"""Tests for event-wise detection metrics, the dataset sources, and the CSV seam."""

from __future__ import annotations

import pandas as pd

from evaluate.detection_harness import evaluate_detection, infer_channels
from evaluate.detection_metrics import (
    aggregate_scores,
    merge_intervals,
    score_detection,
)
from ingest.sources import CsvTelemetrySource, SyntheticSource


def test_score_detection_perfect_overlap():
    s = score_detection(detected=[(100, 200)], labelled=[(150, 250)])
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.fbeta == 1.0


def test_score_detection_false_alarm_on_nominal():
    # No labelled anomaly, but the detector fired -> precision 0, recall defined 1.
    s = score_detection(detected=[(10, 20)], labelled=[])
    assert s.precision == 0.0
    assert s.tp_detections == 0


def test_score_detection_missed_and_partial():
    # Two labelled anomalies, only the first overlapped; one detection is spurious.
    s = score_detection(
        detected=[(100, 150), (500, 520)], labelled=[(120, 200), (800, 900)]
    )
    assert s.recalled == 1  # only the first anomaly is hit
    assert s.tp_detections == 1  # only the (100,150) detection overlaps a real one
    assert s.recall == 0.5
    assert s.precision == 0.5


def test_merge_intervals():
    assert merge_intervals([(0, 10), (5, 15), (20, 25)]) == [(0, 15), (20, 25)]
    assert merge_intervals([]) == []


def test_aggregate_scores_sums_counts():
    a = score_detection([(0, 10)], [(0, 10)])  # tp
    b = score_detection([(0, 10)], [])  # false alarm
    agg = aggregate_scores([a, b])
    assert agg.n_detected == 2
    assert agg.tp_detections == 1
    assert agg.precision == 0.5


def test_synthetic_source_labels_match_fault():
    runs = list(SyntheticSource(n_per_class=1).runs())
    by_mech = {r.truth_mechanism: r for r in runs}
    # Nominal has no labelled interval; faults have exactly one.
    assert by_mech["nominal_no_fault"].anomaly_intervals == []
    assert len(by_mech["bearing_friction_increase"].anomaly_intervals) == 1


def test_detection_harness_recall_and_precision():
    result = evaluate_detection(SyntheticSource(n_per_class=5))
    c = result["corpus"]
    assert c["recall"] == 1.0  # every injected fault region is caught
    assert c["precision"] > 0.4  # honest: spurious detections drag this down
    assert c["fbeta"] > 0.5


def test_csv_source_roundtrip_is_the_esa_adb_seam(tmp_path):
    # Write synthetic runs + a labels CSV in the real-data on-disk format, then
    # load them back through CsvTelemetrySource and confirm they evaluate.
    src = SyntheticSource(n_per_class=1)
    data_dir = tmp_path / "runs"
    data_dir.mkdir()
    label_rows = []
    for run in src.runs():
        run.telemetry.to_csv(data_dir / f"{run.run_id}.csv", index=False)
        for start, end in run.anomaly_intervals:
            label_rows.append(
                {"run_id": run.run_id, "start_idx": start, "end_idx": end}
            )
    pd.DataFrame(label_rows).to_csv(tmp_path / "labels.csv", index=False)

    csv_source = CsvTelemetrySource(data_dir, tmp_path / "labels.csv")
    loaded = {r.run_id: r for r in csv_source.runs()}
    assert "friction_increase_000" in loaded
    assert loaded["friction_increase_000"].anomaly_intervals  # labels came through
    assert infer_channels(loaded["friction_increase_000"].telemetry) == [
        "wheel_speed_rpm",
        "wheel_current_a",
        "wheel_temp_c",
    ]

    result = evaluate_detection(csv_source)
    assert result["corpus"]["recall"] == 1.0
