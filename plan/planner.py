"""Planner — orchestrates the full closed loop for one run:

    ingest -> detect (explore) -> hypothesize -> twin-simulate -> score (evaluate)
    -> rank hypotheses -> generate Groq EXHYTE timeline -> persist to RunStore

This is the "engine" described in the dossier: the same loop regardless of
which strategic direction (A/B/C) the product ultimately points at.

Scoring is signature-based (see ``evaluate/signature.py``): the detector's event
window fixes *where* the fault is, a real fault signature is extracted relative
to a pre-fault baseline, and each hypothesis is simulated in the twin and reduced
to the same signature. Diagnosis is therefore driven by the detected event and by
fault *shape*, not by raw noisy values — which is what lets it tell the
mechanisms apart and recognize genuinely nominal telemetry.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from domain import (
    FaultParameter,
    OversightMode,
    OversightPolicy,
    RunManifest,
    SimMapping,
    ValidationStatus,
)
from evaluate.autonomy_gate import decide_oversight
from evaluate.human_gate import evaluate_human_gate
from evaluate.llm_council import LLMCouncil
from evaluate.scorer import SignatureScorer, normalize_posteriors
from evaluate.signature import extract_signature, split_baseline_window
from evaluate.verifiers import ReactionWheelVerifier
from explore.detector import ZScoreDetector
from hypothesize.generator import StubLlm
from hypothesize.groq_explainer import generate_exhyte_timeline
from runstore import RunStore
from twin.simulator import ToySimulator

CHANNELS = ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]


def _fault_key(
    fault_params: tuple[FaultParameter, ...],
) -> tuple[tuple[str, float], ...]:
    """Hashable key for a hypothesis's fault parameters (mechanism-independent)."""
    return tuple(sorted((p.name, round(p.value, 6)) for p in fault_params))


def _simulated_signatures(
    fault_params: tuple[FaultParameter, ...],
    n_sims: int,
    duration: int,
) -> list[np.ndarray]:
    """Signature ensemble for one hypothesis's fault parameters — each simulated
    run reduced to a fault signature (nominal early segment vs. later segment).
    """
    mapping = SimMapping(subsystem="reaction_wheel", fault_params=fault_params)
    sigs: list[np.ndarray] = []
    for i in range(n_sims):
        sim = ToySimulator().configure(mapping).run(duration_s=duration, seed=1000 + i)
        base, win = split_baseline_window(sim)
        sigs.append(extract_signature(win, base))
    return sigs


def _real_signature(telemetry: pd.DataFrame) -> np.ndarray:
    """Signature of the real run, extracted the *same* way as the simulated
    signatures (first-fraction baseline vs. the rest). This symmetry — rather
    than relying on the detector's noisy per-event onset — is what makes the
    real-vs-simulated comparison fair. Assumes the fault (if any) begins after
    the baseline fraction, which holds for the synthetic reaction-wheel data.
    """
    baseline, window = split_baseline_window(telemetry)
    return extract_signature(window, baseline)


def _simulated_windows(
    fault_params: tuple[FaultParameter, ...],
    n_sims: int,
    duration: int,
) -> list[pd.DataFrame]:
    """Post-baseline window of each simulated run, for DataFrame-based scorers
    (DistanceScorer, SBIScorer). Uses the same event-localized window the
    signature path uses, so those scorers are compared on the fault region
    rather than the whole raw series (the original bug)."""
    mapping = SimMapping(subsystem="reaction_wheel", fault_params=fault_params)
    windows: list[pd.DataFrame] = []
    for i in range(n_sims):
        sim = ToySimulator().configure(mapping).run(duration_s=duration, seed=1000 + i)
        _, win = split_baseline_window(sim)
        windows.append(win.reset_index(drop=True))
    return windows


def run_closed_loop(
    telemetry: pd.DataFrame,
    run_id: str | None = None,
    n_sims_per_hypothesis: int = 20,
    store: RunStore | None = None,
    groq_api_key: str | None = None,
    detector: object | None = None,
    scorer: object | None = None,
    oversight_policy: OversightPolicy | None = None,
    calibration_status: object | None = None,
) -> dict:
    """Run the EXHYTE closed loop.

    ``detector`` and ``scorer`` are injectable (defaults: baseline ``ZScoreDetector``
    and event-window ``SignatureScorer``). Any object with the ``Scorer`` protocol
    (``DistanceScorer``, ``SBIScorer``) is also accepted — the planner detects it by
    the absence of an ``expects_signatures`` marker and feeds it the event-window
    DataFrames instead of signature vectors. See ``plan/components.py``.
    """
    run_id = run_id or str(uuid.uuid4())
    store = store or RunStore()

    detector = detector or ZScoreDetector()
    llm = StubLlm()
    scorer = scorer or SignatureScorer()
    uses_signatures = getattr(scorer, "expects_signatures", False)
    council = LLMCouncil(api_key=groq_api_key)

    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        dataset="synthetic",
        detector_name=getattr(detector, "name", type(detector).__name__),
        twin_name="ToySimulator",
        llm_name=llm.name,
    )
    store.put(run_id, "manifest", manifest, key="manifest")

    events = detector.detect(telemetry, CHANNELS, run_id=run_id)
    store.put(run_id, "events", events, key="events")

    report: dict = {"run_id": run_id, "n_events": len(events), "events": []}
    if not events:
        return report

    # --- Calibration-gated autonomy (AutonomyGate) ---
    # One CalibrationStatus per run: injected, else derived from the reaction-wheel
    # Verifier's real SBC/PPC (cached, deterministic). The gate reads ONLY this
    # status and the policy -- never raw scores or domain data.
    policy = oversight_policy or OversightPolicy()
    cal_status = calibration_status or ReactionWheelVerifier().calibration_status()
    report["calibration"] = {
        "domain": cal_status.domain,
        "passed": cal_status.passed,
        "confidence": cal_status.confidence,
        "method": cal_status.method,
    }
    store.put(run_id, "calibration_status", cal_status, key="calibration")

    # One fault per run: build the event-localized real input once, and cache
    # the per-mechanism simulated inputs (they depend only on fault parameters).
    duration = min(len(telemetry), 5000)
    if uses_signatures:
        real_sig = _real_signature(telemetry)
        sim_sig_cache: dict[tuple, list[np.ndarray]] = {}
    else:
        _, real_window_df = split_baseline_window(telemetry)
        real_window_df = real_window_df.reset_index(drop=True)
        sim_win_cache: dict[tuple, list[pd.DataFrame]] = {}

    for event in events:
        hypotheses = llm.generate(event)
        gated = [h for h in hypotheses if llm.critique(h).plausible]

        results = []
        for hyp in gated:
            key = _fault_key(hyp.fault_params)
            if uses_signatures:
                if key not in sim_sig_cache:
                    sim_sig_cache[key] = _simulated_signatures(
                        hyp.fault_params, n_sims_per_hypothesis, duration
                    )
                results.append(scorer.score(hyp, real_sig, sim_sig_cache[key]))
            else:
                if key not in sim_win_cache:
                    sim_win_cache[key] = _simulated_windows(
                        hyp.fault_params, n_sims_per_hypothesis, duration
                    )
                results.append(scorer.score(hyp, real_window_df, sim_win_cache[key]))

        ranked = sorted(normalize_posteriors(results), key=lambda r: -r.posterior)
        store.put(run_id, "hypotheses", gated, key=f"hyp_{event.id}")
        store.put(run_id, "sim_results", ranked, key=f"sim_{event.id}")

        top = ranked[0] if ranked else None
        top_hyp = next((h for h in gated if h.id == top.hypothesis_id), None) if top else None

        # --- LLM Council Deliberation & Human Validation Gate ---
        consensus = council.deliberate(event, top_hyp, top)
        validation_status = evaluate_human_gate(event, consensus, top.posterior if top else None)

        # AutonomyGate: oversight mode from the calibration status alone.
        oversight_mode = decide_oversight(cal_status, policy)
        requires_human = (
            validation_status == ValidationStatus.ESCALATED_PENDING_HUMAN
            or oversight_mode == OversightMode.ACTIVE
        )
        store.put(run_id, "oversight_mode", oversight_mode, key=f"oversight_{event.id}")

        store.put(run_id, "council_consensus", consensus, key=f"council_{event.id}")
        store.put(run_id, "validation_status", validation_status, key=f"val_{event.id}")

        ranked_mechanisms = [
            (
                next(h.mechanism for h in gated if h.id == r.hypothesis_id),
                round(r.posterior, 3),
            )
            for r in ranked
        ]

        ai_timeline = generate_exhyte_timeline(
            event_id=event.id,
            channel=event.channel,
            severity=event.severity.value,
            score=event.score,
            top_hypothesis=top_hyp.mechanism if top_hyp else None,
            top_text=top_hyp.text if top_hyp else None,
            posterior=top.posterior if top else None,
            ranked_mechanisms=ranked_mechanisms,
            api_key=groq_api_key,
        )

        report["events"].append(
            {
                "event_id": event.id,
                "channel": event.channel,
                "severity": event.severity.value,
                "score": event.score,
                "top_hypothesis": top_hyp.mechanism if top_hyp else None,
                "top_hypothesis_text": top_hyp.text if top_hyp else None,
                "posterior": top.posterior if top else None,
                "ranked_mechanisms": ranked_mechanisms,
                "council_consensus": {
                    "consensus_score": consensus.consensus_score,
                    "verdict": consensus.verdict.value,
                    "summary": consensus.summary,
                    "individual_votes": [
                        {
                            "role": v.role.value,
                            "agrees": v.agrees_with_top_hyp,
                            "confidence": v.confidence,
                            "rationale": v.rationale,
                        }
                        for v in consensus.individual_votes
                    ],
                },
                "validation_status": validation_status.value,
                "autonomy_mode": oversight_mode.value,
                "calibrated_confidence": cal_status.confidence,
                "calibration_passed": cal_status.passed,
                "calibration_method": cal_status.method,
                "requires_human": requires_human,
                "ai_timeline": ai_timeline,
            }
        )

    return report
