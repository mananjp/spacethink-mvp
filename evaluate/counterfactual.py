"""Counterfactual Replay — closed-loop diagnosis replay with alternative hypotheses.

"Was this antenna-power anomaly consistent with your reaction-wheel friction
 hypothesis?" → yes/no with sim fit.

Foundation for the ViaSat-3-class narrative. Replays a run with alternative
hypothesis parameters enabled and compares the fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from domain import FaultParameter, Hypothesis, SimMapping, SimResult
from evaluate.scorer import DistanceScorer, normalize_posteriors
from twin.simulator import ToySimulator


@dataclass
class CounterfactualResult:
    """Result of a counterfactual replay."""
    original_hypothesis_id: str
    original_mechanism: str
    original_distance: float
    original_posterior: float
    counterfactual_hypotheses: list[CounterfactualHypothesisResult]
    verdict: str  # "consistent" | "inconsistent" | "ambiguous"
    explanation: str


@dataclass
class CounterfactualHypothesisResult:
    """Result for a single counterfactual hypothesis."""
    hypothesis_id: str
    mechanism: str
    distance: float
    posterior: float
    delta_distance: float  # vs original
    fit_quality: str  # "better" | "worse" | "similar"


def run_counterfactual(
    real_telemetry: pd.DataFrame,
    original_hypothesis: Hypothesis,
    original_sim_result: SimResult,
    alternative_hypotheses: list[Hypothesis],
    twin_cls: type = ToySimulator,
    scorer_cls: type = DistanceScorer,
    n_sims: int = 10,
    channels: list[str] | None = None,
) -> CounterfactualResult:
    """Replay a diagnosis with alternative hypotheses and compare fit.

    Parameters
    ----------
    real_telemetry : The original real telemetry data
    original_hypothesis : The top-ranked hypothesis from the original run
    original_sim_result : The SimResult from the original scoring
    alternative_hypotheses : Alternative hypotheses to test
    twin_cls : Digital twin class (default ToySimulator)
    scorer_cls : Scorer class (default DistanceScorer)
    n_sims : Number of simulations per hypothesis
    channels : Telemetry channels to compare

    Returns
    -------
    CounterfactualResult with comparison data
    """
    channels = channels or ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
    scorer = scorer_cls(channels=channels) if hasattr(scorer_cls, '__init__') else scorer_cls()
    duration = min(len(real_telemetry), 5000)

    cf_results = []
    all_sim_results = []

    for alt_hyp in alternative_hypotheses:
        # Run twin simulation under alternative hypothesis
        twin = twin_cls()
        twin.configure(SimMapping(
            subsystem="reaction_wheel",
            fault_params=alt_hyp.fault_params,
        ))
        sims = twin.run_ensemble(n_sims=n_sims, duration_s=duration)

        # Score
        sim_result = scorer.score(alt_hyp, real_telemetry.iloc[:duration], sims)
        all_sim_results.append(sim_result)

    # Re-normalize posteriors including original
    combined = [
        SimResult(
            hypothesis_id=original_hypothesis.id,
            distance=original_sim_result.distance,
            posterior=0.0,
            n_sims=original_sim_result.n_sims,
            diagnostics=original_sim_result.diagnostics,
        )
    ] + all_sim_results

    ranked = normalize_posteriors(combined)

    # Build counterfactual results
    for alt_hyp, sim_result in zip(alternative_hypotheses, all_sim_results):
        ranked_result = next(r for r in ranked if r.hypothesis_id == alt_hyp.id)
        delta = sim_result.distance - original_sim_result.distance

        if abs(delta) < 0.1 * original_sim_result.distance:
            fit = "similar"
        elif delta < 0:
            fit = "better"
        else:
            fit = "worse"

        cf_results.append(CounterfactualHypothesisResult(
            hypothesis_id=alt_hyp.id,
            mechanism=alt_hyp.mechanism,
            distance=sim_result.distance,
            posterior=ranked_result.posterior,
            delta_distance=delta,
            fit_quality=fit,
        ))

    # Determine verdict
    original_ranked = next(r for r in ranked if r.hypothesis_id == original_hypothesis.id)
    any_better = any(r.fit_quality == "better" for r in cf_results)
    all_worse = all(r.fit_quality == "worse" for r in cf_results)

    if all_worse or original_ranked.posterior > 0.5:
        verdict = "consistent"
        explanation = (
            f"The original diagnosis '{original_hypothesis.mechanism}' remains the best fit. "
            f"All counterfactual hypotheses produced worse fits (posterior: {original_ranked.posterior:.3f})."
        )
    elif any_better:
        better = [r for r in cf_results if r.fit_quality == "better"]
        verdict = "inconsistent"
        explanation = (
            f"Counterfactual analysis found {len(better)} alternative hypothesis(es) with better fit: "
            f"{', '.join(r.mechanism for r in better)}. "
            f"Original '{original_hypothesis.mechanism}' posterior dropped to {original_ranked.posterior:.3f}."
        )
    else:
        verdict = "ambiguous"
        explanation = (
            f"Counterfactual analysis is inconclusive. Multiple hypotheses have similar fit. "
            f"Original posterior: {original_ranked.posterior:.3f}."
        )

    return CounterfactualResult(
        original_hypothesis_id=original_hypothesis.id,
        original_mechanism=original_hypothesis.mechanism,
        original_distance=original_sim_result.distance,
        original_posterior=original_ranked.posterior,
        counterfactual_hypotheses=cf_results,
        verdict=verdict,
        explanation=explanation,
    )


def format_counterfactual_report(result: CounterfactualResult) -> str:
    """Format a counterfactual result as Markdown."""
    lines = [
        "## 🔄 Counterfactual Replay Analysis",
        "",
        f"**Original Hypothesis:** `{result.original_mechanism}` "
        f"(distance: {result.original_distance:.3f}, posterior: {result.original_posterior:.3f})",
        "",
        f"**Verdict:** **{result.verdict.upper()}**",
        f"> {result.explanation}",
        "",
        "### Alternative Hypotheses",
        "",
        "| Mechanism | Distance | Posterior | Δ Distance | Fit |",
        "|-----------|----------|----------|------------|-----|",
    ]

    for cf in result.counterfactual_hypotheses:
        icon = "✅" if cf.fit_quality == "worse" else "⚠️" if cf.fit_quality == "similar" else "❌"
        lines.append(
            f"| {cf.mechanism} | {cf.distance:.3f} | {cf.posterior:.3f} | "
            f"{cf.delta_distance:+.3f} | {icon} {cf.fit_quality} |"
        )

    return "\n".join(lines)
