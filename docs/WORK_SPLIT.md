# Parallel Work Split

How to divide the build across 2, 3, or 4 people so they work concurrently instead of
serially. Companion to `docs/IMPLEMENTATION_PLAN.md` (the phases) — this document assigns the
phases to people.

Grounding principle (CS130 / Parnas / Conway): **a module is an independent work
assignment.** The architecture already hides each likely-to-change decision behind one
module, so the team boundaries should follow those seams. Align people to interfaces, freeze
the interfaces first, and everyone can build behind a stub without waiting on anyone else.

> Direction note: this split is written for the currently-specced path (Direction C / phase 1
> of B in `docs/STRATEGIC_DIRECTIONS.md`). §7 covers how it shifts for Direction A (science).

---

## 1. The one rule that makes parallel work possible

**Freeze the shared contracts in Week 0, before splitting.** Three things are touched by
everyone; if they drift, every track breaks:

1. `domain/` entities (already built) — the shared vocabulary.
2. `runstore/` artifact schemas — how stages hand work to each other.
3. The three swap-interfaces — `Detector`, `Twin`, `LlmClient`.

Once these exist with **stubs on both sides**, each person builds their real implementation
against the other side's stub. Nobody blocks. The stubs are not throwaway — they are the
permanent test doubles that keep CI hermetic (no network, no Basilisk, no API keys).

## 2. The interface contracts (build these first, together)

Small, explicit, and frozen early. Signatures (Python-ish) — the real work hides behind them:

```
# domain/  — already implemented (immutable entities)
EventOfInterest, Hypothesis, FaultParameter, SimMapping, RunManifest

# runstore/  — the integration seam. Everyone reads/writes typed artifacts by run_id.
class RunStore:
    def put(self, run_id: str, kind: str, artifact) -> ArtifactRef
    def get(self, ref: ArtifactRef) -> artifact
    def list(self, run_id: str, kind: str) -> list[ArtifactRef]

# explore/  — Track A owns; Track B consumes events
class Detector(Protocol):
    def detect(self, channels, telecommands) -> list[EventOfInterest]
#   stub: ThresholdDetector (trivial, deterministic) for other tracks' tests

# twin/  — Track A owns; Track B's SBI trains on its ensembles
class Twin(Protocol):
    def configure(self, mapping: SimMapping) -> "Twin"
    def run(self, duration_s: float, seed: int) -> Channels
#   stub: ToySimulator (analytic ODE, milliseconds) — lets Track B start before Basilisk

# hypothesize/  — Track B owns
class LlmClient(Protocol):
    def generate(self, event_features, context) -> list[Hypothesis]
    def critique(self, hyp: Hypothesis) -> Critique
#   stub: StubLlm (returns fixed templated hypotheses) — keeps CI offline

# evaluate/  — Track B owns
class Scorer(Protocol):
    def score(self, hyp: Hypothesis, real: Channels) -> SimResult   # posterior/fit
```

**Definition of "an interface is done":** a contract test exists that any implementation
(real or stub) must pass. The contract test — not the implementation — is the source of
truth both tracks code against.

## 3. Two people (the founders) — primary plan

Split along the deepest seam in the system: **numbers/physics** vs **reasoning/agent**. Each
founder owns a vertical half of the loop end to end.

| | **Track A — Data & Physics** | **Track B — Inference & Agent** |
|---|---|---|
| Owns modules | `ingest/`, `explore/`, `twin/`, fault library, synthetic suite, `evaluate/` (detection metrics) | `evaluate/` (SBI scoring), `hypothesize/`, `knowledge/`, `plan/` |
| Phase 0 (shared) | Co-build `domain`+`runstore`+interfaces+CI (pair or split 1 day) | same |
| Phase 1 | Ingest OPS-SAT-AD/ESA-ADB; forecaster + dynamic thresholding; event-wise metrics | Scaffold `hypothesize/` + `evaluate/` against `ToySimulator` + `StubLlm` stubs |
| Phase 2 | Basilisk scenario + fault library + ensemble runner + **synthetic fault suite** | Build SBI training/inference plumbing on toy ensembles; calibration harness |
| Phase 3 | Hand real ensembles to B; help validate sim-vs-real on OPS-SAT/ADAPT | Train amortized NPE + model comparison on A's ensembles; SBC/PPC guardrails |
| Phase 4 | Provide fault-parameter priors + causal-graph facts | LLM roles (generate/critic/rank), templates, causal-graph truthfulness gate, RAG |
| Phase 5 | Detector polish; demo data prep | `plan/` (EIG menu); dashboard + demo scripts (or split with A) |
| Weekly sync | **runstore contract test must stay green** — the one meeting that matters | same |

**The single real cross-track dependency:** Track B's SBI scorers (Phase 3) need Track A's
Basilisk ensembles (Phase 2). Mitigation is built in — B works against `ToySimulator` from
day one and only swaps in Basilisk ensembles when A ships them, so B is never idle. A should
ship a crude "v0 ensemble" early (even 100 low-fidelity runs) to unblock B's real training
sooner.

**Shared/either-owner work:** `cli/` (orchestration) and `dashboard/` (read-only viewer) sit
on top of the runstore and can be built by whoever has slack, or split (A takes CLI, B takes
dashboard — dashboard is closer to B's hypothesis/uncertainty presentation).

## 4. Three people

Split Track A's physics from its data, because they're different skill sets:

- **Person 1 — Data/Detection:** `ingest/`, `explore/` (forecaster + thresholding),
  detection metrics. Deliverable: ranked events in the runstore.
- **Person 2 — Twin/Physics:** `twin/` (Basilisk), fault library, ensemble runner, synthetic
  suite, **and** the SBI scoring in `evaluate/` (natural home — the person who owns the
  simulator owns the inference that consumes it). Deliverable: verified fault posteriors.
- **Person 3 — Agent/Product:** `hypothesize/` (LLM roles, templates, causal gate),
  `knowledge/`, `plan/`, `dashboard/`, `cli/`. Deliverable: the closed loop + the demo.

Interfaces between them are exactly the contracts in §2, so the coordination cost stays low.
Critical path is Person 2 (twin → suite → SBI feeds both the detector's context and the
agent's verification). Person 2 is the one to unblock first; if you hire one contractor, hire
to relieve Person 2.

## 5. Four people

Add a dedicated inference owner and a dedicated product owner:

- **Data/Detection** (as above).
- **Twin/Simulation:** `twin/`, fault library, ensembles, synthetic suite.
- **Inference/SBI:** `evaluate/` — NPE, model comparison, calibration. Consumes the twin's
  ensembles; the cleanest standalone research role (portable ML skills, no Basilisk needed
  past the ensemble contract).
- **Agent/Product:** `hypothesize/`, `knowledge/`, `plan/`, `dashboard/`, `cli/`.

Beyond 4, parallelism stops paying: the loop only has so many independent seams, and
coordination cost (Brooks) starts to dominate. Scale by deepening a track (more fault
classes, more detectors, more subsystems), not by adding people to the same module.

## 6. Parallel schedule (2-person, 12 weeks)

```
Wk  Track A (Data & Physics)                Track B (Inference & Agent)          Integration
0   ── co-build domain + runstore + interfaces + CI (both) ──                    contracts frozen
1   ingest OPS-SAT-AD                        hypothesize/ + evaluate/ on stubs    stub loop runs e2e
2   forecaster + thresholding                SBI plumbing on ToySimulator        —
3   Basilisk scenario + first faults         calibration harness                 v0 ensemble handoff
4   ESA-ADB subset + metrics                 NPE trains on v0 ensemble            first real scores
5   fault library + synthetic suite ✦        model comparison across hyps         suite = shared truth
6   detector hardening (telecommand FPs)     SBC/PPC guardrails                   top-1 ≥60% checkpoint
7   sim-vs-real on OPS-SAT/ADAPT             scorer validation on real data       calibration report
8   fault priors + causal-graph facts        LLM generate/critic/rank + gate      full loop closes ✦
9   demo data prep                           templates + RAG (knowledge/)         —
10  → dashboard (or CLI)                     plan/ EIG menu                        —
11  demo script 1 (ops triage)              demo script 2 (+ rediscovery if B)   both demos green ✦
12  ── harden, rehearse, buffer (both) ──                                         gate green, demo-ready
```

✦ = milestone with a hard exit criterion (see IMPLEMENTATION_PLAN.md phase gates).

## 7. If the direction changes (A / science-first)

Direction A adds one workstream and reshapes evaluation, but keeps the same seams:

- **New stream — Science Data & Rediscovery** (a person or a shared responsibility): ingest
  instrument/science telemetry (magnetometer/plasma/spectrometer archives), and build the
  **rediscovery evaluation harness** (curated known-discovery datasets with answers held
  out). This replaces the fault-injection suite as the source of ground truth.
- Track A's `twin/` gains environmental/physical models (not just fault models); Track B's
  `hypothesize/` gains the novelty-vs-known-physics gate.
- The 2-person split still holds: A owns data+physics(+science data), B owns
  inference+agent(+rediscovery scoring). It's the same two halves, aimed at discovery instead
  of diagnosis.

## 8. Anti-patterns to avoid

- **Two people editing `domain/` or the runstore schema at once** → the seam drifts and both
  tracks break. Changes to shared contracts are a scheduled, agreed event, not a casual edit.
- **Blocking on a real implementation when a stub exists** → if you're waiting on Basilisk or
  the LLM, you skipped your stub. The stub is the point.
- **Letting the synthetic suite live in one person's head** → it's the shared definition of
  "does the science work"; it belongs in `tests/` in the runstore, owned jointly.
- **Adding people to a blocked track** → Brooks's Law. Unblock the critical path (the twin)
  first; only then does a new hand help.
