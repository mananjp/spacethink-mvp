"""LLM Council — Multi-agent peer review & consensus evaluation panel.

Evaluates detected events, top hypotheses, and digital twin simulation posteriors
from 4 distinct expert personas:
1. Spacecraft Systems Engineer
2. Data & Sensor Quality Analyst
3. Digital Twin Physics Verifier
4. Adversarial Red-Team Skeptic

Calculates a consensus score and verdict to auto-triage routine events or escalate
major discoveries / split decisions to the Human Validation Gate.
"""
from __future__ import annotations

import os
from typing import Optional
from domain import (
    CouncilConsensus,
    CouncilRole,
    CouncilVerdict,
    CouncilVote,
    EventOfInterest,
    Hypothesis,
    SimResult,
)

try:
    import groq
except ImportError:
    groq = None


class LLMCouncil:
    """Multi-agent LLM Council for telemetry anomaly and hypothesis validation."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile", offline: bool = False):
        self.api_key = None if offline else (api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
        self.offline = offline

    def deliberate(
        self,
        event: EventOfInterest,
        hypothesis: Optional[Hypothesis],
        sim_result: Optional[SimResult],
    ) -> CouncilConsensus:
        """Run 4-persona deliberation on the event and top hypothesis fit."""
        if not hypothesis or not sim_result:
            return self._empty_consensus()

        if self.api_key and groq:
            try:
                return self._deliberate_groq(event, hypothesis, sim_result)
            except Exception as e:
                # Fallback to offline multi-persona rule engine if API fails
                return self._deliberate_offline(event, hypothesis, sim_result, note=f"(Groq fallback: {e})")
        else:
            return self._deliberate_offline(event, hypothesis, sim_result)

    def _deliberate_offline(
        self,
        event: EventOfInterest,
        hypothesis: Hypothesis,
        sim_result: SimResult,
        note: str = "",
    ) -> CouncilConsensus:
        """Offline deterministic persona evaluator for reproducible pipeline runs & CI."""
        posterior = sim_result.posterior
        distance = sim_result.distance
        mech = hypothesis.mechanism

        # 1. Systems Engineer
        sys_agrees = posterior >= 0.35 and distance <= 2.5
        sys_conf = min(0.95, round(0.50 + posterior * 0.45, 2))
        sys_rationale = (
            f"Subsystem telemetry dynamics align with '{mech}'. Thermal/current profiles are physically consistent."
            if sys_agrees
            else f"Telemetry profile fit for '{mech}' displays unusual channel drift."
        )

        # 2. Data Quality Analyst
        data_agrees = event.score >= 1.5
        data_conf = 0.90 if event.score > 2.5 else 0.75
        data_rationale = (
            f"Telemetry anomaly score (Z-Score: {event.score:.2f}) indicates true physical state change, not sensor dropout."
            if data_agrees
            else "Low Z-score elevation suggests potential transient telemetry noise."
        )

        # 3. Physics Verifier
        phys_agrees = distance < 1.8 and posterior >= 0.40
        phys_conf = min(0.98, round(1.0 - (distance / 5.0), 2))
        phys_rationale = (
            f"Digital Twin simulation match is strong (RMSE distance: {distance:.2f}, Posterior: {posterior:.3f})."
            if phys_agrees
            else f"Simulation residual error (distance: {distance:.2f}) indicates physics deviation."
        )

        # 4. Red Team Skeptic
        # Skeptic is stricter on mechanism prior and posterior cutoff
        red_agrees = posterior >= 0.45 and hypothesis.prior >= 0.30
        red_conf = round(0.60 + (posterior * 0.35), 2)
        red_rationale = (
            f"No major contradictions found in alternative mechanisms for '{mech}'. High confidence diagnosis."
            if red_agrees
            else f"Challenging diagnosis: Mechanism prior ({hypothesis.prior}) or posterior ({posterior:.3f}) requires human audit."
        )

        votes = (
            CouncilVote(CouncilRole.SYSTEMS_ENGINEER, sys_agrees, sys_conf, sys_rationale),
            CouncilVote(CouncilRole.DATA_QUALITY_ANALYST, data_agrees, data_conf, data_rationale),
            CouncilVote(CouncilRole.TWIN_PHYSICS_VERIFIER, phys_agrees, phys_conf, phys_rationale),
            CouncilVote(CouncilRole.RED_TEAM_SKEPTIC, red_agrees, red_conf, red_rationale),
        )

        return self._build_consensus(votes, note)

    def _deliberate_groq(
        self,
        event: EventOfInterest,
        hypothesis: Hypothesis,
        sim_result: SimResult,
    ) -> CouncilConsensus:
        """Call Groq API to obtain live AI Council deliberation."""
        client = groq.Groq(api_key=self.api_key)

        prompt = f"""
You are the EXHYTE Space Think LLM Council, consisting of 4 expert personas:
1. SYSTEMS_ENGINEER: Evaluates spacecraft mechanical/thermal subsystem limits.
2. DATA_QUALITY_ANALYST: Evaluates telemetry noise, sensor dropouts, and signal integrity.
3. TWIN_PHYSICS_VERIFIER: Evaluates digital twin simulation fit and residual distances.
4. RED_TEAM_SKEPTIC: Challenges assumptions and looks for alternative explanations.

TELEMETRY EVENT:
- Channel: {event.channel}
- Severity: {event.severity.value}
- Anomaly Z-Score: {event.score:.2f}

TOP HYPOTHESIS DIAGNOSIS:
- Mechanism: {hypothesis.mechanism}
- Explanation: {hypothesis.text}
- Prior Probability: {hypothesis.prior:.2f}
- Digital Twin Fit Distance: {sim_result.distance:.3f}
- Posterior Score: {sim_result.posterior:.3f}

Perform multi-persona deliberation. For each of the 4 roles, respond with JSON format:
{{
  "votes": [
    {{"role": "systems_engineer", "agree": true/false, "confidence": 0.0-1.0, "rationale": "..."}},
    {{"role": "data_quality_analyst", "agree": true/false, "confidence": 0.0-1.0, "rationale": "..."}},
    {{"role": "twin_physics_verifier", "agree": true/false, "confidence": 0.0-1.0, "rationale": "..."}},
    {{"role": "red_team_skeptic", "agree": true/false, "confidence": 0.0-1.0, "rationale": "..."}}
  ]
}}
Only return valid JSON.
"""
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        import json
        data = json.loads(response.choices[0].message.content)
        raw_votes = data.get("votes", [])

        role_map = {
            "systems_engineer": CouncilRole.SYSTEMS_ENGINEER,
            "data_quality_analyst": CouncilRole.DATA_QUALITY_ANALYST,
            "twin_physics_verifier": CouncilRole.TWIN_PHYSICS_VERIFIER,
            "red_team_skeptic": CouncilRole.RED_TEAM_SKEPTIC,
        }

        parsed_votes = []
        for rv in raw_votes:
            role = role_map.get(rv.get("role"), CouncilRole.SYSTEMS_ENGINEER)
            parsed_votes.append(
                CouncilVote(
                    role=role,
                    agrees_with_top_hyp=bool(rv.get("agree", True)),
                    confidence=float(rv.get("confidence", 0.8)),
                    rationale=str(rv.get("rationale", "")),
                )
            )

        if len(parsed_votes) < 4:
            return self._deliberate_offline(event, hypothesis, sim_result, note="(Groq response incomplete)")

        return self._build_consensus(tuple(parsed_votes))

    def _build_consensus(self, votes: tuple[CouncilVote, ...], note: str = "") -> CouncilConsensus:
        n_agree = sum(1 for v in votes if v.agrees_with_top_hyp)
        total_conf = sum(v.confidence for v in votes)
        avg_conf = total_conf / len(votes) if votes else 0.0

        if n_agree == 4:
            verdict = CouncilVerdict.UNANIMOUS_AGREEMENT
            summary = f"All 4 Council personas unanimously confirm the top diagnosis. {note}".strip()
        elif n_agree == 3:
            verdict = CouncilVerdict.STRONG_CONSENSUS
            summary = f"Strong Council consensus (3/4 agree). Minor dissent noted. {note}".strip()
        elif n_agree == 2:
            verdict = CouncilVerdict.SPLIT_COUNCIL
            summary = f"Split Council decision (2/4 agree). Human review required. {note}".strip()
        else:
            verdict = CouncilVerdict.REJECTED_BY_COUNCIL
            summary = f"Council rejected the top hypothesis ({n_agree}/4 agree). Further simulation needed. {note}".strip()

        # Score balances agreement count and average persona confidence
        consensus_score = round((n_agree / len(votes)) * 0.7 + (avg_conf * 0.3), 3)

        return CouncilConsensus(
            consensus_score=consensus_score,
            verdict=verdict,
            summary=summary,
            individual_votes=votes,
        )

    def _empty_consensus(self) -> CouncilConsensus:
        return CouncilConsensus(
            consensus_score=0.0,
            verdict=CouncilVerdict.SPLIT_COUNCIL,
            summary="No hypothesis or simulation results available for council review.",
            individual_votes=(),
        )
