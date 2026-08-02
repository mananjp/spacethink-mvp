# spaceThink — Autonomous Spacecraft Diagnostic & Discovery Agent

An EXHYTE-style closed-loop agent for spacecraft telemetry diagnosis and scientific discovery:
**Explore (Detect) → Hypothesize (Generate Mechanisms) → Test (Digital Twin) → Score (SBI) → LLM Council → Human Gate**.

Built for ADCS/reaction-wheel anomaly triage, insurance evidence generation, cross-fleet analysis, and onboard edge autonomy.

---

## Architecture & Features

### Phase 1 — Twin-Verified Moat & Honest Metrics
- **`BasiliskTwin` (`twin/basilisk_twin.py`)**: 4-wheel coupled attitude dynamics simulator with orbital thermal inertia, gyroscopic cross-coupling, and process pool parallelization (`run_ensemble`).
- **`SBIScorer` (`evaluate/sbi_scorer.py`)**: Simulation-Based Inference scorer with amortized NPE (Neural Posterior Estimation) and kernel density fallback.
- **Calibration Guardrails (`evaluate/calibration.py`)**: Simulation-Based Calibration (SBC) and Posterior Predictive Checks (PPC) gating scorer deployment.
- **ESA Honest Metrics (`evaluate/esa_metrics.py`)**: Event-wise F0.5 (precision-weighted), alarming precision (anti-fragmentation), and ADTQC. Strictly enforces the **PA-F1 Banned Grep Gate** (`tests/test_no_pa_f1.py`).
- **Synthetic Fault Suite (`tests/synthetic_fault_suite.py`)**: 30 scenarios across 5 fault classes enforcing Top-1 ≥ 60% and Top-2 ≥ 80% CI gates.

### Phase 2 — Muted-Alarm Triage Killer
- **OPS-SAT-AD Ingest (`ingest/opssat_ad.py`)**: Loader for OPS-SAT Anomaly Detection benchmark dataset (Zenodo DOI-gated with synthetic fallback for channels 41–46).
- **`TelemanomLineageDetector` (`explore/telemanom_lineage.py`)**: Forecaster with dynamic thresholding and **telecommand conditioning** (exogenous inputs — #1 false-positive killer).
- **Telecommand Explainer (`hypothesize/telecommand_explainer.py`)**: Benign auto-explain pass for telecommand-aligned operations (bias swaps, slews, safe mode).
- **EIG Planner (`plan/eig_planner.py`)**: Expected Information Gain action selector with QA-5 safety allowlist validation schema (`SAFETY_ALLOWLIST`).
- **Triage Dashboard (`dashboard/triage_mode.py`)**: Bulk triage view (98% auto-explain rate) and per-event deep dives.

### Phase 3 — Insurance Evidence Pack & Ledger
- **Audit Ledger (`runstore/ledger.py`)**: Append-only, hash-chained provenance log (`prev_hash`, `ts`, `actor`, `kind`, `artifact_hash`, `prompt_version`, `model_version`, `sim_params`, `seed`) with tamper verification (`verify_chain()`).
- **Content-Addressed Store (`runstore/store.py`)**: Content-addressed artifact store (SHA-256) backed by SQLite metadata index (`runstore_index.db`).
- **Counterfactual Replay (`evaluate/counterfactual.py`)**: Closed-loop diagnosis replay comparing alternative hypotheses fit.
- **Claim-Pack Exporter (`cli/claim_pack.py`)**: Underwriter/regulator claim pack exporter (JSON & Markdown) with claim-evidence graphs and per-mechanistic-step uncertainty.

### Phase 4 — Multi-Tenant FastAPI & Fleet Scale
- **FastAPI REST API (`api/app.py`)**: Endpoints for `/v1/events`, `/v1/runs/{run_id}`, `/v1/claim-pack`, `/v1/health`, `/v1/ready` with in-memory token-bucket rate limiting and API key auth.
- **Twin Calibrator (`twin/calibrator.py`)**: Parametric twin auto-calibration from customer telemetry.
- **Fleet Clustering (`evaluate/clustering.py`)**: Cross-fleet anomaly signature correlation and coordinated response recommendations.

### Phase 5 & 6 — Knowledge Layer, Reproducibility & Edge Autonomy
- **Knowledge RAG (`knowledge/rag.py`)**: Retrieval-augmented generation module supporting ChromaDB, sentence-transformers, and OpenAI fallback adapters.
- **CLI Rerun (`cli/rerun.py`)**: Deterministic run replay (`spacethink rerun <run_id>`) and run diffing (`--diff <id2>`).
- **Model Export (`explore/export.py`)**: ONNX/TFLite dynamic quantizer targeting 59 KB RAM edge budget.
- **Onboard Edge Loop (`explore/onboard.py`)**: Memory-efficient `OnboardEvaluator` running Welford's algorithm on CubeSat hardware.

---

## Project Structure

```
spacethink-mvp/
├── api/            FastAPI surface (/v1/events, /v1/runs, /v1/claim-pack, /v1/health)
├── cli/            Typer CLI commands (main.py, claim_pack.py, rerun.py)
├── dashboard/      Streamlit EXHYTE dashboard & triage mode (app.py, triage_mode.py)
├── db/             SQLAlchemy models + Postgres init scripts
├── domain/         Shared immutable entities (EventOfInterest, Hypothesis, AuditEntry, etc.)
├── evaluate/       Scorer, LLM Council, Human Gate, SBI, ESA metrics, Counterfactual, Clustering
├── explore/        Detectors (ZScore, TelemanomLineage), Model Export, Onboard Edge Loop
├── hypothesize/    Generator, Groq explainer, Telecommand explainer
├── ingest/         Synthetic telemetry generator & OPS-SAT-AD ingest
├── knowledge/      RAG wrapper (ChromaDB, sentence-transformers, OpenAI adapter)
├── plan/           Planner (closed loop orchestration) & EIG Planner
├── runstore/       Content-addressed store (store.py), Audit ledger (ledger.py)
├── scripts/        SBI Scorer training script (train_sbi_scorer.py)
├── tests/          Pytest suite (test_pipeline.py, test_llm_council.py, test_no_pa_f1.py, test_phases.py, synthetic_fault_suite.py)
└── twin/           Digital twins (ToySimulator, BasiliskTwin, Calibrator)
```

---

## Quickstart

### 1. Set Up Environment
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run CLI Commands

```bash
# Generate synthetic reaction-wheel telemetry
python -m cli.main generate-data

# Run closed-loop diagnosis on all synthetic runs
python -m cli.main run-all

# Run a digital twin simulation
python -m cli.main twin --fault rw_friction --magnitude 0.8

# Run bulk triage on OPS-SAT-AD telemetry
python -m cli.main triage --telemetry opssat_ad

# Export an insurance claim pack
python -m cli.main claim-pack <run_id> --format json

# Replay a run or diff two runs
python -m cli.main rerun <run_id>
python -m cli.main rerun <run_id_1> --diff <run_id_2>

# Export forecaster for edge deployment (ONNX / NumPy)
python -m cli.main export --format numpy

# Start the FastAPI server
python -m cli.main serve --port 8000
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to inspect detected events, LLM Council deliberation, Human Validation Gate sign-offs, and Groq AI timelines.

---

## Verification & Testing

Run the full automated test suite (26 tests covering unit, contract, and integration tests):

```bash
pytest -v
```

All tests run hermetically against test doubles without requiring network access, paid API keys, or GPU hardware.
