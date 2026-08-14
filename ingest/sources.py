"""Dataset sources — the seam between the pipeline and where telemetry comes from.

Everything downstream (detector, twin, scorer) consumes a ``LabelledRun``: a
channel DataFrame plus optional ground-truth anomaly intervals and, for synthetic
data, the true fault mechanism. Swapping the synthetic generator for a real
benchmark (ESA-ADB / OPS-SAT-AD) is then just a different ``TelemetrySource`` — no
change to the detector or the evaluation harness.

- ``SyntheticSource`` wraps the reaction-wheel generator and derives the labelled
  anomaly interval from the injected fault.
- ``CsvTelemetrySource`` loads real telemetry: one CSV of channels per run, plus an
  optional labels CSV (``run_id,start_idx,end_idx`` — the ESA-ADB anomaly-interval
  format). **This is where real ESA-ADB / OPS-SAT-AD data plugs in.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import pandas as pd

from ingest.synthetic_generator import generate_reaction_wheel_telemetry

Interval = tuple[int, int]  # half-open [start, end) in row indices

# Canonical map from an injected synthetic fault to the mechanism the loop should
# conclude. Nominal ("none") has no labelled anomaly interval.
FAULT_TO_MECHANISM = {
    "none": "nominal_no_fault",
    "friction_increase": "bearing_friction_increase",
    "encoder_dropout": "encoder_dropout",
    "stiction": "stiction",
}


@dataclass(frozen=True)
class LabelledRun:
    run_id: str
    telemetry: pd.DataFrame
    anomaly_intervals: list[Interval] = field(default_factory=list)
    truth_mechanism: str | None = None  # None when unknown (real unlabeled data)


class TelemetrySource(Protocol):
    def runs(self) -> Iterator[LabelledRun]: ...


class SyntheticSource:
    """Deterministic synthetic reaction-wheel runs with derived anomaly labels."""

    def __init__(
        self,
        n_per_class: int = 5,
        n_points: int = 4000,
        fault_start: int = 2000,
        base_seed: int = 100,
        fault_types: list[str] | None = None,
    ):
        self.n_per_class = n_per_class
        self.n_points = n_points
        self.fault_start = fault_start
        self.base_seed = base_seed
        self.fault_types = fault_types or list(FAULT_TO_MECHANISM.keys())

    def runs(self) -> Iterator[LabelledRun]:
        seed = self.base_seed
        for fault_type in self.fault_types:
            for i in range(self.n_per_class):
                df = generate_reaction_wheel_telemetry(
                    fault_type=fault_type,
                    n_points=self.n_points,
                    fault_start=self.fault_start,
                    seed=seed,
                )
                seed += 1
                fault_start = df.attrs.get("fault_start")
                intervals: list[Interval] = (
                    [(int(fault_start), int(len(df)))]
                    if fault_start is not None
                    else []
                )
                yield LabelledRun(
                    run_id=f"{fault_type}_{i:03d}",
                    telemetry=df,
                    anomaly_intervals=intervals,
                    truth_mechanism=FAULT_TO_MECHANISM[fault_type],
                )


class CsvTelemetrySource:
    """Load real telemetry from disk — the ESA-ADB / OPS-SAT-AD plug-in point.

    ``data_dir`` holds one CSV per run (each column is a channel). ``labels_path``
    is an optional CSV with columns ``run_id,start_idx,end_idx`` giving labelled
    anomaly intervals (multiple rows per run allowed). Runs with no label rows are
    treated as unlabelled (``anomaly_intervals=[]``, ``truth_mechanism=None``).
    """

    def __init__(self, data_dir: str | Path, labels_path: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.labels_path = Path(labels_path) if labels_path else None

    def _load_labels(self) -> dict[str, list[Interval]]:
        if not self.labels_path or not self.labels_path.exists():
            return {}
        labels: dict[str, list[Interval]] = {}
        df = pd.read_csv(self.labels_path)
        for _, row in df.iterrows():
            labels.setdefault(str(row["run_id"]), []).append(
                (int(row["start_idx"]), int(row["end_idx"]))
            )
        return labels

    def runs(self) -> Iterator[LabelledRun]:
        labels = self._load_labels()
        for csv_path in sorted(self.data_dir.glob("*.csv")):
            run_id = csv_path.stem
            telemetry = pd.read_csv(csv_path)
            yield LabelledRun(
                run_id=run_id,
                telemetry=telemetry,
                anomaly_intervals=labels.get(run_id, []),
                truth_mechanism=None,
            )
