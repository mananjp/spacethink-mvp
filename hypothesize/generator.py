"""Hypothesis layer — LlmClient protocol + template-based generator (StubLlm).

Per the literature review: hypothesis generation should be template/rule-based
first and LLM-driven later, gated by a causal-graph truthfulness check so the
model cannot propose mechanisms that contradict known physics. StubLlm below
returns fixed templated hypotheses keyed to detector metadata, which is enough
to exercise the full closed loop with synthetic data before wiring in a real
LLM API call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain import EventOfInterest, FaultParameter, Hypothesis, new_id


@dataclass(frozen=True)
class Critique:
    plausible: bool
    reason: str


class LlmClient(Protocol):
    def generate(self, event: EventOfInterest, context: dict) -> list[Hypothesis]:
        ...

    def critique(self, hyp: Hypothesis) -> Critique:
        ...


# Candidate mechanisms for the reaction-wheel wedge (from the fault library).
TEMPLATES = [
    {
        "mechanism": "bearing_friction_increase",
        "text": "Wheel current and temperature are rising while speed drifts down — "
                "consistent with increasing bearing friction (lubricant degradation).",
        "params": [FaultParameter("friction", 0.6)],
        "prior": 0.4,
    },
    {
        "mechanism": "encoder_dropout",
        "text": "Speed channel shows brief zero-reads with no corresponding current/"
                "temperature change — consistent with intermittent encoder dropout, "
                "not a real mechanical fault.",
        "params": [FaultParameter("dropout_rate", 0.01)],
        "prior": 0.3,
    },
    {
        "mechanism": "stiction",
        "text": "Speed shows sharp step-drops paired with current spikes — "
                "consistent with stiction (static friction) events in the wheel bearing.",
        "params": [FaultParameter("stiction_rate", 0.003)],
        "prior": 0.3,
    },
]


class StubLlm:
    """Deterministic templated generator — the offline stub used in CI and for
    the initial synthetic-data closed loop. No network calls.
    """

    name = "stub_llm"

    def generate(self, event: EventOfInterest, context: dict | None = None) -> list[Hypothesis]:
        return [
            Hypothesis(
                id=new_id(),
                event_id=event.id,
                text=t["text"],
                mechanism=t["mechanism"],
                fault_params=tuple(t["params"]),
                prior=t["prior"],
                generator="template",
            )
            for t in TEMPLATES
        ]

    def critique(self, hyp: Hypothesis) -> Critique:
        # Trivial gate: reject anything with a negative fault parameter (nonsensical).
        if any(p.value < 0 for p in hyp.fault_params):
            return Critique(plausible=False, reason="negative fault parameter")
        return Critique(plausible=True, reason="passes basic sanity gate")
