"""Synthetic telemetry generator — stands in for ESA-ADB / OPS-SAT-AD while
we build the pipeline. Produces multichannel time series with injected faults
(reaction-wheel friction increase, encoder drop-out, stiction) so every layer
(detect -> hypothesize -> test -> score) can be exercised end to end without
any real satellite data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42


def _base_signal(
    n: int, rng: np.random.Generator, freq: float = 0.01, noise: float = 0.05
) -> np.ndarray:
    t = np.arange(n)
    signal = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 3 * freq * t)
    return signal + rng.normal(0, noise, size=n)


def generate_reaction_wheel_telemetry(
    n_points: int = 5000,
    fault_start: int = 3000,
    fault_type: str = "friction_increase",
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Generate synthetic reaction-wheel channels: speed, current, temperature.

    fault_type in {"friction_increase", "encoder_dropout", "stiction", "none"}.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    # Noise is added *before* the amplitude scaling, so keep the `noise=` args
    # small: effective per-channel noise is (amplitude * noise). These give
    # realistic magnitudes (speed ~12 rpm, current ~0.01 A, temp ~0.3 C) so the
    # injected faults sit clearly above the noise floor. The twin in
    # ``twin/simulator.py`` mirrors these levels.
    speed = 4000 + 200 * _base_signal(n_points, rng, freq=0.002, noise=0.06)
    current = 0.5 + 0.05 * _base_signal(n_points, rng, freq=0.002, noise=0.2)
    temperature = 25 + 3 * _base_signal(n_points, rng, freq=0.0005, noise=0.1)

    if fault_type == "friction_increase":
        ramp = np.clip((t - fault_start) / (n_points - fault_start), 0, 1)
        current = current + ramp * 0.4
        temperature = temperature + ramp * 15
        speed = speed - ramp * 150
    elif fault_type == "encoder_dropout":
        dropout_mask = (t > fault_start) & (rng.random(n_points) < 0.01)
        speed[dropout_mask] = 0.0
    elif fault_type == "stiction":
        stiction_events = (t > fault_start) & (rng.random(n_points) < 0.003)
        speed[stiction_events] -= rng.uniform(300, 600, size=stiction_events.sum())
        current[stiction_events] += rng.uniform(0.3, 0.8, size=stiction_events.sum())

    df = pd.DataFrame(
        {
            "t": t,
            "wheel_speed_rpm": speed,
            "wheel_current_a": current,
            "wheel_temp_c": temperature,
        }
    )
    df.attrs["fault_type"] = fault_type
    df.attrs["fault_start"] = fault_start if fault_type != "none" else None
    return df


def generate_dataset(
    out_dir: str = "data/synthetic", n_runs: int = 12, seed: int = RNG_SEED
) -> None:
    from pathlib import Path

    rng = np.random.default_rng(seed)
    fault_types = ["none", "friction_increase", "encoder_dropout", "stiction"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(n_runs):
        ft = fault_types[i % len(fault_types)]
        df = generate_reaction_wheel_telemetry(
            fault_type=ft,
            fault_start=int(rng.integers(2000, 3800)),
            seed=int(rng.integers(0, 1_000_000)),
        )
        df.to_csv(out / f"run_{i:03d}_{ft}.csv", index=False)

    print(f"spacethink: wrote {n_runs} synthetic runs to {out.resolve()}")


if __name__ == "__main__":
    generate_dataset()
