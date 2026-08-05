"""Reaction wheel domain verifier.

Generalizes twin simulation + distance/SBI scoring under the Verifier protocol.
Its ``CalibrationStatus`` is *computed* from real SBC/PPC (evaluate/calibration.py),
not reported by the scorer — that is the property the AutonomyGate depends on.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from domain import (
    CalibrationStatus,
    Evidence,
    Hypothesis,
    VerificationResult,
    Verifier,
)
from evaluate.calibration import derive_calibration_status, run_ppc, run_sbc
from evaluate.scorer import DistanceScorer, Scorer
from twin.simulator import ToySimulator, Twin


@lru_cache(maxsize=1)
def default_reaction_wheel_calibration() -> CalibrationStatus:
    """Reaction-wheel CalibrationStatus from REAL SBC + PPC, computed once.

    SBC/PPC are a property of the twin family (deterministic, fixed seed), so the
    result is cached and reused across runs. The current distance/SBI scorer FAILS
    SBC (rank statistics cluster -> not rank-uniform), so this returns
    ``passed=False`` by design until the SBI scorer is trained to be rank-calibrated.
    That is the calibration gate doing its job, not a defect — and it is the concrete
    bar the next milestone (train SBIScorer) must clear.
    """
    sbc = run_sbc(DistanceScorer(), n_prior_samples=12, n_sims=6, seed=42)
    ppc = run_ppc(DistanceScorer(), n_sims=20, seed=42)
    return derive_calibration_status("reaction_wheel", sbc, ppc, method="SBC+PPC")


class ReactionWheelVerifier:
    """Reaction wheel domain hypothesis verifier.

    Combines physical twin simulation (BasiliskTwin or ToySimulator) with statistical
    scoring (SBIScorer or DistanceScorer). ``calibration_status()`` derives its
    confidence from SBC/PPC diagnostics unless an explicit status is injected.
    """

    domain_name = "reaction_wheel"

    def __init__(
        self,
        twin: Twin | None = None,
        scorer: Scorer | None = None,
        calibrated: bool | None = None,
        confidence: float | None = None,
        diagnostics: dict[str, Any] | None = None,
    ):
        self.twin = twin or ToySimulator()
        self.scorer = scorer or DistanceScorer()
        # Explicit injection (tests / known-good configs). When left as None,
        # calibration_status() computes the REAL status from SBC + PPC.
        self._calibrated = calibrated
        self._confidence = confidence
        self._diagnostics = diagnostics

    def verify(self, hypothesis: Hypothesis, evidence: Evidence) -> VerificationResult:
        """Verify a reaction wheel hypothesis against evidence telemetry."""
        real_df: pd.DataFrame = evidence.raw_data.get("real", pd.DataFrame())
        simulated: list[pd.DataFrame] = evidence.raw_data.get("simulated", [])

        if real_df.empty:
            return VerificationResult(
                hypothesis_id=hypothesis.id,
                verified=False,
                fit_score=float("inf"),
                posterior=0.0,
                diagnostics={"error": "Empty evidence telemetry"},
            )

        if not simulated:
            # Generate simulated telemetry via twin if not explicitly provided
            simulated = [self.twin.run(duration_s=100, seed=i) for i in range(5)]

        sim_res = self.scorer.score(hypothesis, real_df, simulated)
        is_verified = sim_res.distance < 1.5  # threshold check

        return VerificationResult(
            hypothesis_id=hypothesis.id,
            verified=is_verified,
            fit_score=sim_res.distance,
            posterior=sim_res.posterior,
            diagnostics=sim_res.diagnostics,
        )

    def calibration_status(self) -> CalibrationStatus:
        """Standardized calibration status, derived from SBC/PPC (or injected)."""
        if self._confidence is not None or self._calibrated is not None:
            return CalibrationStatus(
                domain=self.domain_name,
                passed=bool(self._calibrated) if self._calibrated is not None else True,
                confidence=float(self._confidence) if self._confidence is not None else 1.0,
                method="SBC+PPC",
                diagnostics=self._diagnostics or {"source": "injected"},
            )
        return default_reaction_wheel_calibration()


# Ensure ReactionWheelVerifier implements Verifier protocol at import time
_check_rw: Verifier = ReactionWheelVerifier()
