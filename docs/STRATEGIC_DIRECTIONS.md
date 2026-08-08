# Strategic Directions

Status: open decision (2026-07-20). This document is deliberately **not** a commitment.
It records the several directions spaceThink can take and what each one changes, so the
founders can choose consciously rather than drift into one by default.

> **Read this first.** Parts II–III of this dossier (the Specification and Implementation
> Plan) and the MVP Roadmap currently detail **Direction C (ops-first)** as the most
> fully-worked instantiation — because that is where the market research pointed for
> near-term customers and revenue. Directions A and B below are equally valid framings of
> the *same underlying system*; they are documented here but not yet specced to the same
> depth. Choosing A or B reshapes the users, data, demos, and success metrics in the later
> Parts, but not the core engine.

---

## 1. The invariant core (true in every direction)

Whatever we choose, the machine is the same — the EXHYTE closed loop:

```
   Explore telemetry  →  Hypothesize a mechanism  →  Test it (simulate / observe)  →  Refine
        ▲                                                                              │
        └──────────────────────────── loop: refine / pivot / sunset ───────────────────┘
```

What never changes: the exploration/anomaly engine, the machine-readable hypothesis
objects, the digital-twin testing layer, the simulation-as-truth gate (nothing unverified
reaches a human), and the human-on-the-loop posture. **What changes between directions is
the *target* of the hypotheses, and therefore who cares, what data feeds it, and how you
prove it works.**

The bridge between "engineering" and "science" is one idea: an anomaly that is *neither a
known failure mode nor a commanded maneuver* is exactly where new physics hides. The health
machinery is what rules out the boring explanations so the genuinely novel ones surface. So
science does not require a different engine — it is the ceiling built on the same floor.

## 2. Two axes, not one choice

The directions live on two independent axes:

```
                       Onboard / acting autonomously
                                    ▲
             (C′) onboard fault     │     (A′) autonomous science probe
                  response          │          — the full original vision
                                    │
   ─ Engineering / health ──────────┼────────────── Scientific discovery ─▶
                                    │
             (C) ground-based       │     (A) ground-based discovery
                 anomaly triage     │         assistant (rediscovery)
                 [current docs]     │
                                    ▼
                        Offline / ground analysis
```

- **Axis 1 — Purpose:** diagnose the spacecraft (engineering) ↔ discover things about the
  world (science).
- **Axis 2 — Autonomy:** analyze on the ground after the fact ↔ reason and *act* onboard
  under communication delay (re-point an instrument, schedule a follow-up).

Direction **B is the diagonal**: start bottom-left (C), move toward top-right (A′) as data,
customers, and trust accrue. The MVP stays in the bottom row (offline) for *all three*
directions — onboard autonomy is a later-stage move gated by safety, and nothing about it
needs to be decided now.

---

## 3. Direction A — Science-First (Autonomous Discovery)

**Thesis.** An AI scientist for spacecraft: the full discovery loop run over telemetry *and
instrument data*, autonomously surfacing and testing hypotheses about real physical
phenomena. AEGIS generalized — from images to telemetry-wide reasoning, and from
target-selection to mechanism discovery. This is the closest match to the original vision:
"robots and satellites doing closed-loop experiments that yield new scientific discoveries."

- **Who cares / who pays:** planetary-science and heliophysics mission teams, instrument
  PIs, agency science directorates (NASA/ESA), science-operations centers. Revenue is
  grant- and mission-funded (NASA ROSES, science-topic SBIRs), not commercial SaaS.
- **Data:** science/instrument telemetry (magnetometers, plasma/particle detectors,
  spectrometers, radiation monitors) plus housekeeping for context; archival mission data
  (NASA PDS, CDAWeb/heliophysics, ESA PSA).
- **MVP demo — *rediscovery*:** take a dataset where a phenomenon was *later confirmed* (a
  known space-weather event, an instrument artifact that turned out to be real physics, a
  discovered periodicity), hold the answer out, and show the agent proposing and confirming
  it. This is evaluable and it directly shows the vision.
- **How you prove it works:** rediscovery rate on a curated set of known discoveries;
  novelty *and* truthfulness scoring (per TruthHypo — a novel-sounding claim that
  contradicts known physics is worse than useless); expert judgment.
- **Fundability:** hardest near-term revenue answer; strongest mission/ambition narrative.
  YC will love the ambition and press hard on "who pays this year."
- **Hardest risk:** the oracle problem — what *counts* as a discovery, and how to evaluate
  it; long science feedback loops; agency sales cycles; instrument-data access.
- **Delta from current docs:** swap users/data/demos; keep the engine and the twin; add
  instrument-data ingestion and a novelty-vs-known-physics gate; success metrics become
  rediscovery-based, not fault-ranking.

## 4. Direction B — Science as Mission, Ops as Wedge (Recommended default)

**Thesis.** Same engine, sequenced. Commercialize health/anomaly diagnosis first (the wedge
that has customers, data, and clean evaluation today) while the *stated mission and roadmap*
is autonomous science. The ops product funds the moonshot and generates the operator
relationships and telemetry access that the discovery capability later needs.

- **Who cares / who pays:** commercial operators now → science teams / agencies later.
- **Data:** housekeeping telemetry now (ESA-ADB, OPS-SAT-AD) → instrument/science data later.
- **MVP demo — *both*:** the ops triage / verified-root-cause demo **and** one rediscovery
  thread, so investors see the floor (revenue) *and* the ceiling (the mission). The
  discovery thread must be genuinely built, not a slide.
- **How you prove it works:** fault top-1/top-2 accuracy now, plus one rediscovery case.
- **Fundability:** strongest overall — a near-term revenue answer *and* a moonshot. This is
  how most credible deep-tech companies frame themselves (wedge product, mission ambition).
- **Hardest risk:** split focus and two customer types; discipline to keep the discovery
  thread real rather than letting the ops product quietly become the whole company (which is
  exactly the drift that produced the first version of this dossier).
- **Delta from current docs:** the current SPEC/Plan stand as *Phase 1 of B*; add the
  science north-star to the roadmap narrative and one instrument-data rediscovery demo to
  the MVP.

## 5. Direction C — Ops-First (Verified Anomaly Diagnosis)

**Thesis.** A verified anomaly-diagnosis product for fleet operations: detect → hypothesize
the fault → verify it in the twin → recommend the next action. Scientific discovery stays in
Future Work, exactly as the founding paper's "Scope and assumptions" section scoped it. **This
is what Parts II–III of this dossier currently specify.**

- **Who cares / who pays:** LEO constellation operators (5–50 sats, 3–15 person ops teams);
  GSaaS / ops-outsourcing providers as a channel; defense SBIR as a year-2 layer.
- **Data:** ESA-ADB, OPS-SAT-AD housekeeping telemetry.
- **MVP demo:** the two demos already in the roadmap (live ESA-anomaly closed loop; the
  muted-alarm triage demo).
- **How you prove it works:** true injected fault top-1 ≥60% / top-2 ≥80%; detection within
  ±0.05 event-wise F0.5 of the ESA-ADB SOTA baseline.
- **Fundability:** clearest customer and revenue wedge of the three.
- **Hardest risk:** it is not the vision the founders described; commoditization pressure
  from Intella, Quindar, Constellation Space, and adjacent well-funded players.
- **Delta from current docs:** none — it is the current plan.

---

## 6. Side-by-side

| Axis | A — Science-first | B — Mission+Wedge | C — Ops-first |
|---|---|---|---|
| Primary user | Science/mission teams, agencies | Operators now → science later | Constellation operators |
| Data | Instrument + science telemetry | Housekeeping now → instrument later | Housekeeping telemetry |
| MVP demo | Autonomous rediscovery | Ops triage **+** one rediscovery | Ops triage / verified root cause |
| Evaluation | Rediscovery rate; novelty+truth | Fault accuracy **+** one rediscovery | Fault top-1/top-2; event-wise F0.5 |
| Near-term revenue | Grants / mission-funded | Commercial SaaS (funds mission) | Commercial SaaS |
| Vision fidelity | Highest | High | Low |
| Hardest risk | Evaluating "a discovery" | Focus / real discovery thread | Not the vision; commoditization |
| Current spec depth | Framing only | Framing + C = phase 1 | Fully specced (Parts II–III) |

## 7. How to decide (not now, but soon)

- If a **revenue + YC answer within ~12 months is a hard constraint** → C, or B's phase 1.
- If **the mission is the point** and you can fund it through grants/agency partnerships,
  and you're willing to accept fuzzier evaluation → A.
- If you want **both** and are willing to sequence them → B (and treat the current
  Parts II–III as B's first phase).
- **Autonomy (onboard, acting) is orthogonal** and is a later-stage move in every case; the
  MVP stays offline/replay regardless, for safety and tractability.

These are not permanently mutually exclusive. B is literally "C then A." The one path to
*avoid* is choosing C by accident — shipping the ops product and waking up in two years as a
satellite-ops diagnostics vendor because the discovery mission never got a line in the plan.
This document exists so that doesn't happen silently.

## 8. Recommended next step

Pick a **primary** direction to detail next (rewrite the SPEC §2 goals, the roadmap, and the
success metrics to match), while keeping this document as the standing record of the paths
not taken. If undecided, default to **B**: it preserves the vision as the actual point of the
company while keeping the fundable wedge already worked out in Parts II–III.
