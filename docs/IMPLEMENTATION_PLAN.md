# spaceThink — Implementation Plan

Direction **B** (Ops wedge → Science rediscovery) · YC demo target · ADCS-only vertical · Open-source thin-wrapper RAG.

Built for an AI agent (or a human teammate) to execute against. Every phase ends at a verifiable gate. Cross-cutting rules live at the bottom.

---

## 0. Frame

| Decision | Value |
|---|---|
| Strategic direction | B — Ops wedge now, Science rediscovery later |
| Demo target | YC (live + muted-alarm) |
| Vertical scope | ADCS / reaction wheels only |
| Knowledge layer | Open-source thin wrapper (chroma / pgvector) |
| Stack | Python 3.11, Postgres via Docker, FastAPI surface (when wired), Streamlit dashboard, Groq LLM (with OpenAI fallback) |

The plan is sequenced so the **YC demo lands at the end of Phase 2**. Phases 3–5 harden the moat. Phase 6 is the strategic-option track.

### Stable contracts (Day 0)

Three things freeze early and never drift:

1. `domain/` — immutable entities (already in place).
2. `runstore/` — extend to **content-addressed artifacts + SQLite index** so runs are replay-deterministic.
3. Four Protocols behind which everything else builds:
   - `Detector` ← `TelemanomLineageDetector` (Phase 2) plugs in beside `ZScoreDetector`.
   - `Twin` ← `BasiliskTwin` (Phase 1) plugs in beside `ToySimulator`.
   - `Scorer` ← `SBIScorer` (Phase 1) plugs in beside `DistanceScorer`.
   - `Adjudicator` *(new, Phase 3)* — per-mechanistic-step uncertainty + claim-evidence graph.

Stubs are not throwaway. They are the permanent test doubles that keep CI hermetic.

---

## 1. Track assignment (Conway-friendly 2-person split)

| Track | Owns | Phases |
|---|---|---|
| **A — Data & Physics** | `ingest/`, `explore/`, `twin/`, fault library, synthetic suite, detection metrics | 1, 2, 4, 6 |
| **B — Inference & Agent** | `evaluate/` (inference half), `hypothesize/`, `knowledge/`, `plan/`, `api/`, `dashboard/`, `cli/` orchestration | 3, 5 |

Weekly sync: the `runstore` contract test must stay green. Phases 1 and 3 can run in parallel; Phase 2 needs Phase 1 ensembles.

---

## Phase 1 — The Twin-Verified Moat (4 weeks)

**Goal:** replace the toy. Until this lands, the "verified" claim is vapor.

### 1.1 `BasiliskTwin` (`twin/basilisk_twin.py`)

- New class implementing the `Twin` Protocol.
- Scenario: 3-axis spacecraft, 4 reaction wheels, sun-pointing attitude (via Basilisk Python wrapper, ISC licensed).
- Fault-parameter map:
  - `friction` → RW friction multiplier 5–20× nominal (per Carneiro 2022).
  - `dropout_rate` → encoder zero-read probability per step.
  - `stiction_rate` → static-friction spike probability per step.
- Output DataFrame uses the same channel contract as `ToySimulator` (`t`, `wheel_speed_rpm`, `wheel_current_a`, `wheel_temp_c`).
- `run_ensemble(n, duration, base_seed)` parallelized via process pool.
- Imports: if Basilisk is missing, raise `ImportError` with a clear install hint. CI hermetic mode still passes via `ToySimulator`.

### 1.2 `SBIScorer` (`evaluate/sbi_scorer.py`)

Two offline-trained components; online inference in milliseconds:

1. **Amortized NPE per fault family** (`sbi` v0.26.x, Apache-2.0):
   - Generate 10 000 sims per family via `BasiliskTwin.run_ensemble`.
   - Train with an embedding network over `[wheel_speed_rpm, wheel_current_a, wheel_temp_c]` windows.
   - Output: posterior over fault magnitude.
2. **Amortized Bayesian model comparison across hypotheses**:
   - BayesFlow-style neural classifier (or sbi `NRE` fallback).
   - Output: posterior model probabilities / Bayes factors.

Train script: `scripts/train_sbi_scorer.py`. Reuses Phase 1 ensembles; produces `models/sbi/<family>/{posterior,calibration,ppc}`.

### 1.3 Calibration guardrails (`evaluate/calibration.py`)

- **Simulation-based calibration (SBC)** — 1 000 prior samples × 1 000 sims per family.
- **Posterior-predictive checks (PPC)** — generated vs. real summary stats.
- **A scorer that fails SBC or PPC is blocked**, not deployed. Diagnostics land in `runstore` next to every `SimResult`.
- These checks run in CI as blocking gates.

### 1.4 Honest metrics module (`evaluate/esa_metrics.py`)

Wrap the MIT-licensed ESA-ADB metrics code:

- Corrected **event-wise F0.5** (precision-weighted).
- Alarming precision (anti-fragmentation).
- ADTQC (timing quality).
- Channel-aware and subsystem-aware scoring.

**PA-F1 banned in code.** A `pytest`-time grep gate fails the build if `point_adjust` or `pa_f1` appears anywhere under `src/` or `tests/`. Internal scalar: PR-based or `PA%K`.

Implements the `Scorer` Protocol — `ESAMetricsScorer` plugs in beside `DistanceScorer` / `SBIScorer`.

### 1.5 Synthetic fault suite (`tests/synthetic_fault_suite.py`)

The regression bed for every downstream claim:

- ≥ 30 scenarios × ≥ 5 fault classes (friction, encoder-dropout, stiction, gyro-bias, sensor-noise).
- Baseline truth: injected fault class + magnitude.
- **CI gate:** top-1 ≥ 60 %, top-2 ≥ 80 %. Failing either blocks the PR.
- Artifacts land in `runstore`.

### 1.6 Phase 1 exit criteria

- `spacethink twin run --fault rw_friction --magnitude 8x` produces plausible faulty telemetry.
- 100-sim ensemble ≤ 10 min on a laptop.
- Synthetic suite ≥ 60 % top-1, ≥ 80 % top-2.
- SBC + PPC reports committed to `runstore` per scorer.
- All tests green with `BasiliskTwin` + `SBIScorer`. Hermetic CI (no Basilisk installed) still passes via `ToySimulator`.

---

## Phase 2 — The Muted-Alarm Triage Killer (4 weeks)

**Goal:** the YC demo. A day of telemetry → 98 % benign auto-explain, 2 % escalated with verified cause.

### 2.1 OPS-SAT-AD ingest (`ingest/opssat_ad.py`)

- Download via Zenodo DOI gated by env var; cache locally.
- Returns `Channel` segments + telecommand stream + anomaly labels.
- Schema-validate at boundary; fail fast with file/field-level errors.
- Works on the lightweight subset (channels 41–46) up front.
- First real-telemetry test set for the sim-vs-real bridge (Phase 1 risk R1).

### 2.2 `TelemanomLineageDetector` (`explore/telemanom_lineage.py`)

- Channel-wise GRU/LSTM one-step forecaster + nonparametric dynamic thresholding.
- **Telecommand conditioning from day one** (exogenous inputs) — the #1 false-positive killer per the dossier.
- Error smoothing, detection merging, pruning — engineering budget equal to the net.
- Implements `Detector` Protocol. `ZScoreDetector` retained for tests.

### 2.3 Benign auto-explain pass (`hypothesize/telecommand_explainer.py`)

- Detected events that align with active telecommands get routed to "expected nominal" with citation (telecommand record + retrieved doc).
- The single most persuasive YC moment: *"this is a commanded reaction-wheel bias swap, not a fault."*
- For v1, a small static telecommand table covers the YC case; later, `knowledge/` feeds this.

### 2.4 Triage dashboard mode (`dashboard/triage_mode.py`)

- **Page 1 — bulk triage**: "1 day × 1 fleet → 412 alerts → 406 auto-explained, 6 escalated with verified cause."
- **Page 2 — per-event deep-dive**: refactored from existing `dashboard/app.py`.
- Per-event cost ceiling: **≤ $0.20 LLM spend, ≤ 60 s compute** (the dossier's $2/run level scaled to per-event).

### 2.5 Discrete-menu EIG planner (stub) (`plan/eig_planner.py`)

- Menu: `{keep observing, high-rate downlink A/B, diagnostic slew, wheel spin-up/down test}`.
- Score via Pyro `contrib.oed` `marginal_eig`, reusing Phase 1 ensembles.
- Safety allowlist schema enforced for any proposed command (QA-5).
- For the YC demo, planner output is shown, **not** enabled.

### 2.6 Phase 2 exit criteria

- `python -m cli.main run --telemetry opssat_ad` reproduces a labeled anomaly within 5 min on a laptop.
- Triage dashboard renders 98 % auto-explain on a noisy day.
- LLM spend ≤ $0.20 / event, ≤ 60 s compute / event.
- Both YC demo scripts run from a clean checkout.

---

## Phase 3 — The Insurance Evidence Pack (3 weeks)

**Goal:** the price-multiplier. Pays for the rest of the roadmap. The dossier's counterfactual-replay demo idea.

### 3.1 Counterfactual replay (`evaluate/counterfactual.py`)

- Closed-loop diagnosis → replay the run with alternative hypotheses enabled.
- *"Was this antenna-power anomaly consistent with your reaction-wheel friction hypothesis?"* → yes / no with sim fit.
- Foundation for the ViaSat-3-class narrative.

### 3.2 Audit ledger (`runstore/ledger.py`)

- Append-only, hash-chained log of every artifact.
- Schema: `{prev_hash, ts, actor, kind, artifact_hash, prompt_version, model_version, sim_params, seed}`.
- Each `run_id` has a verifiable provenance trail.
- Tamper demo: flip one byte → chain breaks at the next step.

### 3.3 Claim-pack export (`cli/claim_pack.py`)

- `spacethink claim-pack <run_id> --format pdf,json` → regulator/underwriter-friendly artifact.
- Links every claim to its evidence (claim-evidence graph) + audit ledger digest.
- **Per-mechanistic-step uncertainty surfaced** (the dossier's 2026 uncertainty-granularity finding — trust-builder).

### 3.4 Phase 3 exit criteria

- A claim pack for a synthetic fault passes a "blind read" test — an outsider can verify the chain without prior context.
- Ledger tamper-detection demo: one byte flipped → chain breaks.

---

## Phase 3.5 — The Calibration-Gated Verification Contract (2 weeks)

**Goal:** a domain-agnostic, calibration-checked verification contract that heterogeneous domain-specific verifiers implement, whose calibration status — not a generic confidence score — is the signal a human-oversight policy actually gates on.

### 3.5.1 `VerifierProtocol` (`domain/__init__.py`)

- Generalizes `Twin` + `Scorer` into a unified domain verification protocol (`Verifier`).
- `verify(hypothesis: Hypothesis, evidence: Evidence) -> VerificationResult`.
- `calibration_status() -> CalibrationStatus`.

### 3.5.2 `CalibrationStatus` (`domain/__init__.py`)

- Standardized boundary object crossing the domain boundary:
  `{domain, passed, confidence, method, diagnostics}`.
- Every `Verifier` reports whether *it* has passed its own domain-appropriate calibration check (e.g. SBC+PPC, catalog benchmark) before downstream code trusts its output.

### 3.5.3 `AutonomyGate` (`evaluate/autonomy_gate.py`)

- Gating function `decide_oversight(status: CalibrationStatus, policy: OversightPolicy) -> OversightMode`.
- Reads **only** `CalibrationStatus` — never raw domain data or domain-specific heuristic scores.
- `OversightMode.ACTIVE` → hold for human approval before any action/claim.
- `OversightMode.PASSIVE` → proceed autonomously, log to evidence ledger, notify async.

### 3.5.4 Heterogeneous Domain Verifier Implementations (`evaluate/verifiers/`)

- **Primary Verifier**: `ReactionWheelVerifier` (`evaluate/verifiers/reaction_wheel.py`) wrapping reaction-wheel twin simulation + SBI scoring + SBC/PPC calibration checks.
- **Second Verifier**: `AstroCatalogVerifier` (`evaluate/verifiers/astro_catalog.py`) wrapping astronomical transient catalog cross-matching + physical lightcurve plausibility checks to prove domain agnosticism.

### 3.5.5 Phase 3.5 exit criteria

- `ReactionWheelVerifier` and `AstroCatalogVerifier` both pass contract compliance tests for `VerifierProtocol`.
- `AutonomyGate` evaluates calibration status correctly across both domains without branching on raw data types.
- PR-level unit tests verify `decide_oversight` active vs. passive transitions strictly enforce calibration bounds.

---


## Phase 4 — Multi-Tenant / Fleet-at-Scale (4 weeks)

**Goal:** channel leverage. One deal, many fleets.

### 4.1 Per-customer twin calibration (`twin/calibrator.py`)

- Auto-calibrated parametric `BasiliskTwin` from a customer's telemetry (channels + labeled anomalies).
- Fidelity tiers:
  - **Tier 1** — parametric, auto-calibrated, common subsystems (default).
  - **Tier 2** — `Sedaro` / `TrueTwin` import path for high-fidelity.
- Cold-start motion: "first 10 anomalies re-diagnosed" for design partners.

### 4.2 FastAPI surface (`api/app.py`)

`fastapi` and `uvicorn` are already in `requirements.txt` but unwired. Now wire them:

- Endpoints:
  - `POST /v1/events` — ingest a window.
  - `GET /v1/runs/{run_id}` — fetch a run.
  - `POST /v1/claim-pack` — request an insurance pack.
  - `GET /v1/health`, `/v1/ready`.
- OpenAPI schema auto-generated.
- Rate limiting middleware (token bucket per tenant).
- Tenant-isolated API-key auth.
- Consistent error envelope: `{"error": {"code": "...", "message": "...", "details": [...]}}`.
- Request-ID propagation for tracing.

### 4.3 Cross-fleet anomaly clustering (`evaluate/clustering.py`)

- *"This signature appeared on 7 of your satellites in 24 h."*
- Recommends a coordinated response (commands, downlink priority).

### 4.4 Phase 4 exit criteria

- Two design partners onboarded on Tier 1 twins.
- OpenAPI schema published; contract tests cover ≥ 80 % of endpoints.
- Security audit clean (`pip-audit` clean; auth paths covered by tests).

---

## Phase 5 — Knowledge Layer + Reproducibility (3 weeks)

**Goal:** the long-term asset per customer.

### 5.1 `knowledge/` module (`knowledge/rag.py`)

Thin wrapper — buy, don't build:

- Dev: `chromadb` local persistence.
- Prod: `pgvector` (fits the existing Postgres stack via the `db/models.py` infrastructure).
- Embeddings: `all-MiniLM-L6-v2` offline; text-embedding-3-small online (OpenAI fallback adapter must coexist with Groq).
- Ingest sources: past anomaly reports + spacecraft manuals.
- Retrieval at event-time → injected into hypothesis generation (the dossier's "buy EXPLORE" rule).

### 5.2 `spacethink rerun <run_id>` (`cli/rerun.py`)

- Re-runs from runstore + captured seed + frozen model versions.
- Same outputs bit-for-bit within stochastic tolerance.
- Required for partner due-diligence.

### 5.3 Run diff

- `spacethink rerun <id1> --diff <id2>` → why did these differ?
- Sources the audit ledger + run store.

### 5.4 Phase 5 exit criteria

- `spacethink rerun` returns the same `top_hypothesis` within stochastic tolerance.
- RAG improves top-1 accuracy on a held-out set of past anomaly reports.

---

## Phase 6 — Edge Path + Onboard Autonomy (4 weeks, parallel)

**Goal:** strategic option, not prerequisite.

### 6.1 TFLite / ONNX export (`explore/export.py`)

- Quantize the Telemanom forecaster.
- Target: **59 KB RAM, OPS-SAT-class Cortex** (the dossier's Berkenkamp 2026 result).

### 6.2 Onboard loop (research)

- Edge version of the loop running on the forecaster alone.
- "Was this event real?" decision before downlink.

### 6.3 Phase 6 exit criteria

- RAM + latency report on real hardware targets.
- A second demo: "the same code path runs on the ground and on CubeSat-class hardware."

---

## Cross-cutting rules

These ship continuously, not at phase end.

- **Quality gate green** at every phase end (existing CI).
- **Test pyramid**: contract tests for Protocols → unit tests for stubs → synthetic suite → OPS-SAT-AD replay.
- **Stubs are permanent test doubles** — the LLM/SDI/sim path must build hermetically without network or paid services.
- **No `point.adjust` anywhere** — a grep gate enforces it.
- **Per-mechanistic-step uncertainty** baked into every claim artifact from Phase 3 onward.
- **No secrets in code**; LLM spend caps enforced.
- **Modifiability invariant**: swapping a detector / LLM / simulator / scorer touches exactly one adapter module (each Protocol).
- **12-Factor config**: all settings via env vars; `.env.example` carries every required key (already in place).
- **Graceful shutdown** in any long-running API process.

---

## Definition of Done (pre-PR)

- All tests pass (contract + unit + synthetic suite + integration).
- Coverage ≥ 80 % on `src/`; below is a fail.
- Lint + format clean (`ruff` / `ruff format` for Python).
- Type checks pass (`mypy --strict`).
- Docker build succeeds (`docker build .`).
- Security audit clean (`pip-audit`).
- Docs updated (this file, `README.md`, any new module README).
- No secrets committed.
- Health + ready endpoints verified if API changed.

**Build failure = stop.** Fix root cause, re-run from Step 1.

---

## What to do FIRST (kickoff checklist)

1. Initialize git and pin `.gitignore` (dossier marks this as pending owner decision on hosting/license).
2. Install `borromeo` locally and confirm the verify gate is green from a clean checkout.
3. Add `sbi`, `pyro` (with `contrib.oed`), `telemanom` reference, `chromadb`/`pgvector`, `tflite-runtime` (or `onnxruntime`) to `requirements.txt` (gated by extras where appropriate).
4. Freeze the four Protocols in writing before any new module is added.
5. Cut Phase 1 tickets; first PRs:
   - `runstore` content-addressing + SQLite index (precondition for everything).
   - `twin/basilisk_twin.py` alongside `ToySimulator`.
   - `evaluate/sbi_scorer.py` with the `Scorer` Protocol integration test as the contract spec.
   - `evaluate/esa_metrics.py` + the PA-F1 grep gate.
   - `tests/synthetic_fault_suite.py` with the 60 / 80 % CI gates wired.
6. Keep `ToySimulator` + `DistanceScorer` paths green at every commit — hermetic CI must not regress.

---

## Open items still to confirm with founders

- Basilisk install weight: ~1 GB Docker image, ≥ 2 min first build. OK to commit to?
- OpenAI/Anthropic fallback adapter in addition to Groq (one-line swap, no behavior change)?
- Hosting + license decision for the repo (MIT or Apache-2.0) before `git init`?
