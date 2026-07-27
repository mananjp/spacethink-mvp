"""Planner — orchestrates the full closed loop for one run:

    ingest -> detect (explore) -> hypothesize -> twin-simulate -> score (evaluate)
    -> rank hypotheses -> generate Groq EXHYTE timeline -> persist to RunStore

This is the "engine" described in the dossier: the same loop regardless of
which strategic direction (A/B/C) the product ultimately points at.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from domain import RunManifest, SimMapping
from explore.detector import ZScoreDetector
from evaluate.scorer import DistanceScorer, normalize_posteriors
from evaluate.llm_council import LLMCouncil
from evaluate.human_gate import evaluate_human_gate
from hypothesize.generator import StubLlm
from hypothesize.groq_explainer import generate_exhyte_timeline
from runstore import RunStore
from twin.simulator import ToySimulator


CHANNELS = ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]


def run_closed_loop(
    telemetry: pd.DataFrame,
    run_id: str | None = None,
    n_sims_per_hypothesis: int = 20,
    store: RunStore | None = None,
    groq_api_key: str | None = None,
) -> dict:
    run_id = run_id or str(uuid.uuid4())
    store = store or RunStore()

    detector = ZScoreDetector()
    llm = StubLlm()
    scorer = DistanceScorer(channels=CHANNELS)
    council = LLMCouncil(api_key=groq_api_key)

    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        dataset="synthetic",
        detector_name=detector.name,
        twin_name="ToySimulator",
        llm_name=llm.name,
    )
    store.put(run_id, "manifest", manifest, key="manifest")

    events = detector.detect(telemetry, CHANNELS, run_id=run_id)
    store.put(run_id, "events", events, key="events")

    report = {"run_id": run_id, "n_events": len(events), "events": []}

    for event in events:
        hypotheses = llm.generate(event)
        gated = [h for h in hypotheses if llm.critique(h).plausible]

        results = []
        for hyp in gated:
            twin = ToySimulator().configure(SimMapping(subsystem="reaction_wheel", fault_params=hyp.fault_params))
            duration = min(len(telemetry), 5000)
            sims = twin.run_ensemble(n_sims=n_sims_per_hypothesis, duration_s=duration)
            results.append(scorer.score(hyp, telemetry.iloc[:duration], sims))

        ranked = sorted(normalize_posteriors(results), key=lambda r: -r.posterior)
        store.put(run_id, "hypotheses", gated, key=f"hyp_{event.id}")
        store.put(run_id, "sim_results", ranked, key=f"sim_{event.id}")

        top = ranked[0] if ranked else None
        top_hyp = next((h for h in gated if h.id == top.hypothesis_id), None) if top else None

        # --- LLM Council Deliberation & Human Validation Gate ---
        consensus = council.deliberate(event, top_hyp, top)
        validation_status = evaluate_human_gate(event, consensus, top.posterior if top else None)

        store.put(run_id, "council_consensus", consensus, key=f"council_{event.id}")
        store.put(run_id, "validation_status", validation_status, key=f"val_{event.id}")

        ranked_mechanisms = [
            (next(h.mechanism for h in gated if h.id == r.hypothesis_id), round(r.posterior, 3))
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

        report["events"].append({
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
            "ai_timeline": ai_timeline,
        })

    return report

