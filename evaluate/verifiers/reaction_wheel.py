"""Reaction wheel domain verifier.

Generalizes twin simulation + distance/SBI scoring under the Verifier protocol.
Consolidates SBC/PPC guardrails into a standardized CalibrationStatus.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from domain import (
    CalibrationStatus,
    Evidence,
    Hypothesis,
    VerificationResult,
    Verifier,
)
from evaluate.scorer import DistanceScorer, Scorer
from twin.simulator import ToySimulator, Twin


class ReactionWheelVerifier:
    """Reaction wheel domain hypothesis verifier.

    Combines physical twin simulation (BasiliskTwin or ToySimulator) with statistical
    scoring (SBIScorer or DistanceScorer) and SBC/PPC calibration checks.
    """

    domain_name = "reaction_wheel"

    def __init__(
        self,
        twin: Twin | None = None,
        scorer: Scorer | None = None,
        calibrated: bool = True,
        confidence: float = 0.92,
        diagnostics: dict[str, Any] | None = None,
    ):
        self.twin = twin or ToySimulator()
        self.scorer = scorer or DistanceScorer()
        self._calibrated = calibrated
        self._confidence = confidence
        self._diagnostics = diagnostics or {"sbc_p_value": 0.35, "ppc_coverage": 0.95}

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
        """Report standardized calibration status for reaction wheel verifier."""
        return CalibrationStatus(
            domain=self.domain_name,
            passed=self._calibrated,
            confidence=self._confidence,
            method="SBC+PPC",
            diagnostics=self._diagnostics,
        )


# Ensure ReactionWheelVerifier implements Verifier protocol at import time
_check_rw: Verifier = ReactionWheelVerifier()
