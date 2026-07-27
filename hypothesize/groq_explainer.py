"""Groq API Integration — closed-loop EXHYTE timeline generation & scientific reasoning.

Provides LLM-driven scientific reasoning over spacecraft telemetry events,
summarizing:
1. Exploration (Anomaly Detection signature)
2. Hypothesis Generation (Mechanisms & physical priors)
3. Digital Twin Simulation (Tests run & matching)
4. Evaluation & Verdict (Anomaly classification vs. Novel discovery)
"""
from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_groq_client() -> Any | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"[groq_explainer] Failed to initialize Groq client: {e}")
        return None


def generate_exhyte_timeline(
    event_id: str,
    channel: str,
    severity: str,
    score: float,
    top_hypothesis: str | None,
    top_text: str | None,
    posterior: float | None,
    ranked_mechanisms: list[tuple[str, float]],
    api_key: str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Generate a rich EXHYTE timeline report using Groq API (or offline fallback)."""
    client = None
    effective_key = api_key or os.getenv("GROQ_API_KEY")
    if effective_key:
        try:
            from groq import Groq
            client = Groq(api_key=effective_key)
        except Exception:
            client = None

    if client:
        try:
            post_val = f"{posterior:.2f}" if posterior is not None else "N/A"
            prompt = f"""You are spaceThink's Autonomous Spacecraft Diagnostic & Discovery Agent operating under the EXHYTE framework.
Analyze the following telemetry event diagnostic data and generate a clear, structured Markdown report.

--- EVENT DATA ---
Event ID: {event_id}
Channel Flagged: {channel}
Severity: {severity.upper()}
Anomaly Z-Score: {score:.2f}
Top Diagnosed Hypothesis: {top_hypothesis or 'None'}
Top Rationale: {top_text or 'None'}
Top Hypothesis Posterior Probability: {post_val}
All Ranked Candidate Mechanisms: {ranked_mechanisms}

--- REQUIRED STRUCTURE ---
Provide a professional, executive-ready EXHYTE Timeline & Discovery Analysis using the exact headings below:

### ⏱️ EXHYTE Closed-Loop Timeline
- **1. EXPLORE (Telemetry Anomaly Detection)**: Describe what was detected on channel '{channel}', the severity level, and deviation score ({score:.2f} z-score).
- **2. HYPOTHESIZE (Physical Mechanism Generation)**: Detail the physical mechanisms proposed (e.g. lubricant breakdown, stiction, sensor noise, control gain drift).
- **3. TEST (Digital Twin Simulation)**: Explain how parameter-driven ensembles were simulated in the digital twin for each candidate mechanism.
- **4. REFINE & SCORE (Posterior Ranking)**: Explain why the top candidate ({top_hypothesis}) achieved a posterior probability of {post_val}.

### 🔍 Discovery vs. Anomaly Assessment
Classify whether this event is:
- **Known Hardware Anomaly**: Expected wear, degradation, or sensor fault.
- **Novel / Unexpected Phenomenon**: Deviation requiring deeper scientific investigation or mission operator intervention.

### 🚀 Actionable Flight Controller Recommendation
Provide 2-3 concise bullet points for ground station operators (e.g. telecommand adjustments, sensor zeroing, bearing thermal monitoring).
"""

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are spaceThink, an expert deep-space spacecraft health & scientific reasoning agent."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as err:
            print(f"[groq_explainer] Groq API call failed ({err}), falling back to template timeline.")

    # Offline Fallback Template Timeline
    is_dropout = top_hypothesis == "encoder_dropout"
    is_stiction = top_hypothesis == "stiction"
    is_friction = top_hypothesis == "bearing_friction_increase"

    discovery_type = (
        "Known Sensor Noise / Artifact" if is_dropout else
        "Mechanical Component Stress Anomaly" if (is_stiction or is_friction) else
        "Unclassified Subsystem Anomaly"
    )

    ranked_str = "\n".join([f"  - **{m}**: Posterior = {p:.3f}" for m, p in ranked_mechanisms])
    post_str = f"{posterior:.2f}" if posterior is not None else "0.00"

    return f"""### ⏱️ EXHYTE Closed-Loop Timeline (Offline Mode)

- **1. EXPLORE (Telemetry Anomaly Detection)**:
  Flagged anomalous activity on channel `{channel}` with severity **{severity.upper()}** (peak Z-score: **{score:.2f}**).

- **2. HYPOTHESIZE (Physical Mechanism Generation)**:
  Generated {len(ranked_mechanisms)} candidate mechanistic hypotheses with prior fault parameter distributions:
  - Bearing Friction Increase (lubricant degradation)
  - Encoder Dropout (intermittent signal loss)
  - Bearing Stiction (static friction spikes)

- **3. TEST (Digital Twin Simulation)**:
  Executed Monte Carlo digital-twin simulation ensembles (ToySimulator reaction-wheel ODE model) under each candidate parameter configuration.

- **4. REFINE & SCORE (Posterior Ranking)**:
  Compared simulated telemetry vs. real telemetry using normalized RMSE distance:
{ranked_str}

### 🔍 Discovery vs. Anomaly Assessment
- **Classification**: **{discovery_type}**
- **Top Diagnosis**: `{top_hypothesis}` (Posterior: **{post_str}**)
- **Verdict**: {top_text}

### 🚀 Actionable Flight Controller Recommendation
1. Monitor channel `{channel}` for sustained trend progression over subsequent orbits.
2. { "Check encoder wiring & command counter buffer for transient dropouts." if is_dropout else "Consider thermal bias adjustment or momentum dumping to reduce reaction-wheel bearing load." }
3. Human-in-the-loop review recommended before executing automated telecommands.
*(Note: To enable live LLM reasoning, add `GROQ_API_KEY` to your `.env` file).*
"""
