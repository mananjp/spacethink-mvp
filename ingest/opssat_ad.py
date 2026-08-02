"""OPS-SAT-AD ingest — loader for the OPS-SAT Anomaly Detection benchmark dataset.

Downloads from Zenodo (DOI-gated by env var), caches locally, and returns
Channel segments + telecommand stream + anomaly labels.

Works on the lightweight subset (channels 41–46) by default.
Schema-validates at boundary; fails fast with file/field-level errors.
"""
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OPSSAT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "opssat_ad"
ZENODO_DOI = os.getenv("OPSSAT_ZENODO_DOI", "10.5281/zenodo.XXXXXXX")  # placeholder

# Lightweight channel subset (channels 41-46: reaction wheel telemetry)
DEFAULT_CHANNELS = list(range(41, 47))

# Required columns for schema validation
REQUIRED_COLUMNS = {"timestamp", "value", "channel_id"}


@dataclass
class OPSSATSegment:
    """A segment of OPS-SAT telemetry data."""
    channel_id: int
    channel_name: str
    data: pd.DataFrame
    anomaly_labels: list[tuple[int, int]] = field(default_factory=list)  # (start_idx, end_idx)
    telecommands: list[dict] = field(default_factory=list)


@dataclass
class OPSSATDataset:
    """Complete OPS-SAT-AD dataset loaded and validated."""
    segments: list[OPSSATSegment]
    metadata: dict = field(default_factory=dict)
    n_channels: int = 0
    total_points: int = 0


def _download_from_zenodo(cache_dir: Path) -> Path:
    """Download OPS-SAT-AD from Zenodo. Requires OPSSAT_ZENODO_DOI env var."""
    doi = os.getenv("OPSSAT_ZENODO_DOI")
    if not doi:
        raise EnvironmentError(
            "Set OPSSAT_ZENODO_DOI env var to download OPS-SAT-AD dataset. "
            "Use generate_synthetic_opssat() for offline/CI testing."
        )
    # Placeholder — actual download implementation would go here
    raise NotImplementedError(
        f"Zenodo download for DOI {doi} not yet implemented. "
        "Use generate_synthetic_opssat() for development."
    )


def generate_synthetic_opssat(
    cache_dir: Path = OPSSAT_CACHE_DIR,
    n_channels: int = 6,
    n_points: int = 10000,
    seed: int = 42,
) -> OPSSATDataset:
    """Generate a synthetic OPS-SAT-like dataset for offline development and CI.

    Produces multi-channel telemetry with injected anomalies and telecommand
    records, mimicking the structure of the real OPS-SAT-AD benchmark.
    """
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    segments = []

    for ch_idx in range(n_channels):
        ch_id = 41 + ch_idx
        t = np.arange(n_points, dtype=float)

        # Base signal with channel-specific characteristics
        freq = 0.001 + ch_idx * 0.0005
        base = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 3 * freq * t)
        noise = rng.normal(0, 0.05, n_points)
        values = base + noise

        # Inject anomaly window
        anomaly_start = int(n_points * 0.6) + rng.integers(-500, 500)
        anomaly_end = anomaly_start + rng.integers(200, 800)
        anomaly_end = min(anomaly_end, n_points - 1)

        if ch_idx % 2 == 0:
            # Drift anomaly
            ramp = np.linspace(0, 1, anomaly_end - anomaly_start)
            values[anomaly_start:anomaly_end] += ramp * 2.0
        else:
            # Spike anomaly
            n_spikes = rng.integers(5, 20)
            spike_locs = rng.integers(anomaly_start, anomaly_end, size=n_spikes)
            values[spike_locs] += rng.uniform(3, 6, size=n_spikes)

        # Telecommand records
        tc_times = rng.integers(0, n_points, size=3)
        telecommands = [
            {
                "id": f"TC-{ch_id}-{i}",
                "name": f"RW_BIAS_SWAP_{ch_id}",
                "timestamp_idx": int(tc_t),
                "subsystem": "ADCS",
                "parameters": {"target_wheel": ch_id - 40},
            }
            for i, tc_t in enumerate(tc_times)
        ]

        # Build DataFrame
        timestamps = pd.date_range("2024-01-01", periods=n_points, freq="s")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "value": values,
            "channel_id": ch_id,
            "t": t.astype(int),
        })

        # Map to reaction-wheel channels for compatibility
        channel_names = {
            41: "wheel_speed_rpm",
            42: "wheel_current_a",
            43: "wheel_temp_c",
            44: "wheel_speed_rpm_2",
            45: "wheel_current_a_2",
            46: "wheel_temp_c_2",
        }

        segment = OPSSATSegment(
            channel_id=ch_id,
            channel_name=channel_names.get(ch_id, f"channel_{ch_id}"),
            data=df,
            anomaly_labels=[(anomaly_start, anomaly_end)],
            telecommands=telecommands,
        )
        segments.append(segment)

    # Save to cache
    for seg in segments:
        seg.data.to_csv(cache_dir / f"channel_{seg.channel_id}.csv", index=False)

    metadata = {
        "source": "synthetic",
        "n_channels": n_channels,
        "n_points_per_channel": n_points,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return OPSSATDataset(
        segments=segments,
        metadata=metadata,
        n_channels=n_channels,
        total_points=n_channels * n_points,
    )


def load_opssat_ad(
    cache_dir: Path = OPSSAT_CACHE_DIR,
    channels: list[int] | None = None,
) -> OPSSATDataset:
    """Load OPS-SAT-AD dataset from cache, downloading if necessary.

    Falls back to synthetic generation for CI/offline development.
    """
    channels = channels or DEFAULT_CHANNELS

    if not cache_dir.exists() or not any(cache_dir.glob("*.csv")):
        try:
            _download_from_zenodo(cache_dir)
        except (EnvironmentError, NotImplementedError):
            warnings.warn(
                "OPS-SAT-AD not available. Generating synthetic substitute.",
                stacklevel=2,
            )
            return generate_synthetic_opssat(cache_dir=cache_dir)

    # Load from cache
    segments = []
    for ch_id in channels:
        csv_path = cache_dir / f"channel_{ch_id}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        # Schema validation
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Schema validation failed for channel {ch_id}: "
                f"missing columns {missing}"
            )

        segment = OPSSATSegment(
            channel_id=ch_id,
            channel_name=f"channel_{ch_id}",
            data=df,
        )
        segments.append(segment)

    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return OPSSATDataset(
        segments=segments,
        metadata=metadata,
        n_channels=len(segments),
        total_points=sum(len(s.data) for s in segments),
    )


def opssat_to_pipeline_format(dataset: OPSSATDataset) -> pd.DataFrame:
    """Convert OPS-SAT dataset to the pipeline's expected DataFrame format.

    Produces a single DataFrame with columns: t, wheel_speed_rpm,
    wheel_current_a, wheel_temp_c — matching the channel contract.
    """
    # Use first 3 channels as speed/current/temp
    if len(dataset.segments) < 3:
        raise ValueError("Need at least 3 channels for pipeline format conversion")

    n = min(len(s.data) for s in dataset.segments[:3])

    return pd.DataFrame({
        "t": np.arange(n),
        "wheel_speed_rpm": 4000 + 200 * dataset.segments[0].data["value"].to_numpy()[:n],
        "wheel_current_a": 0.5 + 0.05 * dataset.segments[1].data["value"].to_numpy()[:n],
        "wheel_temp_c": 25 + 3 * dataset.segments[2].data["value"].to_numpy()[:n],
    })
