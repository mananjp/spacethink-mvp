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
from evaluate.calibration import (
    derive_calibration_status,
    run_ppc_with_posterior,
    run_sbc_with_posterior,
)
from evaluate.synthetic_likelihood import train_synthetic_likelihood
from evaluate.scorer import DistanceScorer, Scorer
from twin.simulator import ToySimulator, Twin


@lru_cache(maxsize=1)
def default_reaction_wheel_calibration() -> CalibrationStatus:
    """Reaction-wheel CalibrationStatus from real SBC + PPC, computed once.

    SBC/PPC are a property of the twin family (deterministic, fixed seed), so the
    result is cached and reused across runs.

    The posterior under test is the amortized synthetic likelihood, not ABC on the
    raw distance. ABC could not pass SBC and stay informative at the same time: at a
    tight tolerance its ranks piled in the centre (overconfident), and it only reached
    rank-uniformity by widening to ~85% of the prior — calibrated by knowing nothing.
    The synthetic likelihood measures the twin's own summary-statistic noise, so its
    width is earned rather than tuned, and it clears both bars together
    (p ~ 0.09, sharpness ~ 0.97).

    ``passed`` here therefore means calibrated *and* sharp. That pairing is the point:
    rank-uniformity alone is trivially satisfied by returning the prior, so a gate
    built on it would hand PASSIVE autonomy to a scorer with no information.

    Both legs now measure **the same posterior**. The PPC leg previously called
    ``run_ppc(DistanceScorer())``, which consulted neither this posterior nor that
    scorer: it compared the twin at a fixed parameter against the same twin at the same
    parameter, so it returned ``passed=True`` at coverage 1.0 for any input — a
    constant-zero distance function at a physically absurd friction of 50.0 included —
    while carrying 0.4 of the confidence weight that opens the gate. One of the three
    legs holding up the calibration claim could not fail.

    The three legs are complementary rather than redundant, which is why all three are
    required. SBC catches overconfidence, and the predictive check cannot: simulator
    noise is common to the true and estimated predictive distributions, so a posterior
    10x too narrow yields a predictive only ~18% too narrow. Conversely SBC's
    rank-uniformity passes a posterior that ignores its input entirely, and the
    conditional predictive check rejects it outright.
    """
    posterior = train_synthetic_likelihood(n_grid=32, n_reps=6, duration_s=400, seed=17)
    sbc = run_sbc_with_posterior(posterior, n_prior_samples=200, n_sims=19, seed=23)
    ppc = run_ppc_with_posterior(
        posterior, n_trials=200, n_predictive_draws=39, seed=42, family=sbc.family
    )
    # `method` names the calibration method and is a stable part of the
    # CalibrationStatus contract; which posterior was tested is a diagnostic.
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
