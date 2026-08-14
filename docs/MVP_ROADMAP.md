# MVP Roadmap — YC Framing

Date: 2026-07-19. Companion to `docs/IMPLEMENTATION_PLAN.md` (build phases) and
`docs/research/04-market-landscape.md` (full market evidence).

---

## The pitch

> Satellite fleets triple every five years, but anomaly response is still a war room.
> spaceThink is the AI flight-controller that detects an anomaly, hypothesizes the physical
> root cause, and **proves it against a digital twin** before an operator touches a console.

Fallback framing: *"Verified root-cause-as-a-service for spacecraft — we don't just flag
anomalies, we close them."*

**Why now** (all 2024–2026 facts, cited in research/04):
- ~10.5–14.5K active satellites, 3× since 2019, 50–100K projected by 2030; manual ops break
  past ~5–10 sats.
- ViaSat-3 anomaly: $420M insurance claim, largest in history; underwriters exiting.
- YC is actively funding LLM-diagnostics-on-machine-data (Telemetron, Deeptrace, Daqstra)
  and space ops (Quindar S22, Constellation Space W26). The intersection is unclaimed.
- The architecture just became publicly feasible (EXHYTE survey Dec 2025; AURA Nov 2025;
  amortized-SBI fault diagnosis 2026) — first credible vertical product wins the category.

**The moat question ("why won't Quindar/Intella do this?")**: they stop at detection and
workflow. Our loop is *generative + verified*: mechanistic hypotheses tested in simulation
before anyone sees them. That requires physics/SBI + agent engineering they don't have, and
it produces the asset they can't copy quickly — a **fault-signature library** (fault
parameterizations + trained amortized scorers per subsystem) that compounds with every
customer anomaly we close.

## What we demo (two demos, both from the implementation plan's Phase 5)

1. **Real ESA anomaly, closed live** — stream OPS-SAT benchmark telemetry; spaceThink
   detects the anomaly, generates 3 ranked mechanistic hypotheses, runs each in the Basilisk
   twin, and one visibly reproduces the real telemetry signature. Detection → verified root
   cause in <5 minutes, side-by-side with "human anomaly review board: days-to-weeks."
2. **The muted-alarm demo** — a day of noisy telemetry, hundreds of threshold alerts;
   spaceThink triages ~98% as benign-with-explanation (telecommand-aware) and escalates the
   two real faults with twin-verified causes. This attacks the pain Intella's own CCO says
   is the real cost sink: triage, not detection.

Honesty rule for both: metrics shown are event-wise F0.5 on ESA-ADB protocol and top-1/top-2
hypothesis accuracy on the fault-injection suite — no point-adjusted F1, ever. In a field
notorious for inflated claims, evaluation rigor is itself a differentiator.

## Timeline (12 weeks to demo-ready; two founders)

| Weeks | Track A (data/physics) | Track B (inference/agent) | Milestone |
|---|---|---|---|
| 1–3 | Ingest + detector on OPS-SAT-AD | Runstore, Basilisk scenario + fault library | Detector reproduces published baselines |
| 4–5 | ESA-ADB Mission1 subset + metrics | Ensemble runner + synthetic fault suite | "First light": ranked events on real ESA data |
| 6–7 | Detector hardening, telecommand FP kills | SBI scorers + calibration | True fault top-1 ≥60% on synthetic suite |
| 8–9 | Knowledge/RAG wrapper | Templates + LLM generator/critic/ranker + causal gate | Full loop closes end-to-end |
| 10–12 | Dashboard + demo scripts | Planner (EIG menu) + polish | Both YC demos run from clean checkout |

Weekly integration on the runstore contract; borromeo gate green at every milestone.

## Success metrics (what "MVP works" means — from SPEC QAs)

- Detection: within ±0.05 event-wise F0.5 of Telemanom-ESA-Pruned on ESA-ADB M1 subset.
- Diagnosis: true injected fault top-1 ≥60%, top-2 ≥80% (≥30 scenarios, ≥5 fault classes).
- Triage story: >90% of rare-nominal (telecommand-driven) events auto-explained on Mission2
  annotations.
- Ops: full loop ≤5 min/event on a laptop; ≤$2 LLM spend/run; dashboard replay instant.

## First customers & GTM sequence

1. **Design partners (now)**: 2–3 LEO constellation operators, 5–50 sats, 3–15 person ops
   teams (EO/IoT/comms, 2–5 yrs post first launch) — offer free pilot on their historical
   anomaly archive ("we'll re-diagnose your last 10 anomalies").
2. **Channel (post-batch)**: GSaaS/ops-outsourcing providers (KSAT/Telespazio-type) — their
   margin is operator hours; one deal covers many fleets.
3. **Year 2**: defense non-dilutive layer (SpaceWERX/SDA SBIR — Sedaro's proven path);
   insurance angle (root-cause evidence for claims/premiums) as a wedge with GEO operators.

Pricing sketch: per-satellite/month SaaS with a free triage tier (Kayhan freemium
precedent), diagnosis/verification metered on the paid tier.

## Investor objections → prepared answers

| Objection | Answer |
|---|---|
| "Isn't this Quindar/Intella/Constellation Space?" | They detect or automate workflow; nobody verifies hypotheses in simulation. The closed loop is the product. |
| "Where does the twin come from per customer?" | Fidelity-tiered: parametric Basilisk-class twins auto-calibrated from the customer's own telemetry for common subsystems; partner/import path (Sedaro, TrueTwin) for high fidelity. The fault library transfers across customers even when twins don't. |
| "TAM is small (~$2B/yr ops software)" | Constellation growth (10K→50K+ sats), $420M per-anomaly stakes + insurance-aligned pricing, and the same engine expands into industrial predictive maintenance ($105B by 2035). |
| "LLM hallucination in safety-critical ops" | The twin is the hallucination filter — nothing unverified reaches an operator; causal-graph gate rejects impossible mechanisms pre-simulation; human-on-the-loop by default. |
| "Data access / ITAR" | Start with public ESA benchmarks + allied commercial operators under the relaxed 2024 EAR rules; cloud-hosted with data classification designed in from day one. |

## What we are NOT building for the batch (scope contract, SPEC §3)

No flight software, no real-time mission-scale streaming, no commanding real spacecraft,
no multi-mission transfer learning, no instrument-science extension. Offline/replay +
simulation only. Every one of these is a roadmap slide, not an MVP requirement.
