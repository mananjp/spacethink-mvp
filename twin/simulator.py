"""Digital twin layer — Twin protocol + a toy analytic simulator.

ToySimulator is the swap-interface stub referenced in the parallel-work-split
doc: an analytic ODE approximation of reaction-wheel dynamics that runs in
milliseconds, so the Testing/Hypothesis layers can be built and tested before
a high-fidelity simulator (Basilisk, per the research) is wired in later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from domain import SimMapping


class Twin(Protocol):
    def configure(self, mapping: SimMapping) -> "Twin":
        ...

    def run(self, duration_s: float, seed: int) -> pd.DataFrame:
        ...


@dataclass
class ToySimulator:
    """Analytic reaction-wheel model: speed/current/temperature under a
    configurable fault (friction coefficient, encoder dropout rate, stiction
    probability). Not physically exact — a fast, swappable placeholder for
    Basilisk-grade fidelity.
    """

    friction: float = 0.0        # 0 = nominal, >0 = degraded bearing
    dropout_rate: float = 0.0    # probability per step of encoder zero-read
    stiction_rate: float = 0.0   # probability per step of stiction event
    _configured: bool = False

    def configure(self, mapping: SimMapping) -> "ToySimulator":
        params = {p.name: p.value for p in mapping.fault_params}
        self.friction = params.get("friction", 0.0)
        self.dropout_rate = params.get("dropout_rate", 0.0)
        self.stiction_rate = params.get("stiction_rate", 0.0)
        self._configured = True
        return self

    def run(self, duration_s: float = 5000, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = int(duration_s)
        t = np.arange(n)

        speed = 4000 + 200 * np.sin(2 * np.pi * 0.002 * t) + rng.normal(0, 8, n)
        current = 0.5 + 0.05 * np.sin(2 * np.pi * 0.002 * t) + rng.normal(0, 0.01, n)
        temperature = 25 + 3 * np.sin(2 * np.pi * 0.0005 * t) + rng.normal(0, 0.2, n)

        ramp = np.linspace(0, 1, n)
        current += ramp * self.friction * 0.4
        temperature += ramp * self.friction * 15
        speed -= ramp * self.friction * 150

        if self.dropout_rate > 0:
            mask = rng.random(n) < self.dropout_rate
            speed[mask] = 0.0

        if self.stiction_rate > 0:
            mask = rng.random(n) < self.stiction_rate
            speed[mask] -= rng.uniform(300, 600, size=mask.sum())
            current[mask] += rng.uniform(0.3, 0.8, size=mask.sum())

        return pd.DataFrame({
            "t": t,
            "wheel_speed_rpm": speed,
            "wheel_current_a": current,
            "wheel_temp_c": temperature,
        })

    def run_ensemble(self, n_sims: int, duration_s: float, base_seed: int = 0) -> list[pd.DataFrame]:
        return [self.run(duration_s=duration_s, seed=base_seed + i) for i in range(n_sims)]
