# spaceThink — MVP (offline, synthetic-data closed loop)

An EXHYTE-style closed-loop agent for spacecraft telemetry diagnosis:
**Explore (detect) → Hypothesize (generate mechanisms) → Test (digital-twin simulate) → Refine (score/rank)**.

This MVP runs entirely offline on **synthetic reaction-wheel telemetry** — no real
satellite data or network access required — so the full pipeline can be built,
tested, and demoed before wiring in real datasets (ESA-ADB / OPS-SAT-AD),
a high-fidelity simulator (Basilisk), or a real LLM API.

## Stack
- **Language:** Python 3.11+
- **DB:** PostgreSQL 16 (alpine) — run locally via Docker, or point at your own instance
- **ML:** NumPy/Pandas for signal processing, a template-based hypothesis generator (swappable for a real LLM), and a distance-based scorer (a simplified stand-in for full simulation-based inference)
- **CLI:** Typer
- **Dashboard:** Streamlit + Plotly

## Project layout
```
domain/         Shared entities (EventOfInterest, Hypothesis, SimResult, ...)
ingest/         Synthetic telemetry generator (reaction-wheel: friction, encoder dropout, stiction)
explore/        Detector protocol + ZScoreDetector (rolling z-score, dynamic threshold) + ThresholdDetector stub
twin/           Twin protocol + ToySimulator (fast analytic reaction-wheel model)
hypothesize/    LlmClient protocol + StubLlm (templated mechanism generator + sanity gate)
evaluate/       Scorer protocol + DistanceScorer (real-vs-simulated distance -> normalized posterior)
plan/           Planner — orchestrates the full closed loop for one run
runstore/       Filesystem-backed artifact store keyed by run_id
db/             SQLAlchemy models + init script for local Postgres persistence
cli/            Typer entry point (generate-data / run / run-all)
dashboard/      Streamlit viewer over run reports
tests/          Pytest contract + end-to-end smoke tests (all pass against stubs, no external services needed)
```

## Quickstart

### 1. Set up Python environment
```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Start local Alpine Postgres (optional — the closed loop itself doesn't require it yet;
persistence to Postgres is wired but the CLI demo path uses the filesystem RunStore by default)
```bash
cp .env.example .env
docker compose up -d db
python -m db.init_db      # creates tables in the local alpine-postgres container
```

### 3. Generate synthetic telemetry and run the closed loop
```bash
python -m cli.main generate-data          # writes 12 synthetic runs to data/synthetic/
python -m cli.main run-all                # runs detect->hypothesize->test->score on all of them
```
This produces `data/reports.json` with, for every detected event: the ranked candidate
mechanisms, their posterior probabilities, and the plain-language hypothesis text.

### 4. View results in the dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Run tests
```bash
pytest -v
```
All tests run against the deterministic stubs (`ThresholdDetector`, `ToySimulator`, `StubLlm`) —
no network, no real Basilisk, no API keys needed, matching the "hermetic CI" principle from the project plan.

## What's real vs. stubbed in this MVP
| Layer | This MVP | Swap-in later (per project research) |
|---|---|---|
| Detector | Rolling z-score + dynamic threshold | LSTM-forecaster + dynamic thresholding on ESA-ADB/OPS-SAT-AD |
| Twin | Analytic ODE approximation (`ToySimulator`) | Basilisk (high-fidelity spacecraft simulator) |
| Hypothesis generator | Fixed templates (`StubLlm`) | Real LLM (generate/critique/rank) + causal-graph truthfulness gate |
| Scorer | Mean normalized-RMSE distance -> softmax posterior | Amortized neural posterior estimation (`sbi` package) |
| Data | Synthetic reaction-wheel CSVs | Real telemetry (ESA-ADB, OPS-SAT-AD) |

Every module is built behind the same `Protocol` interfaces (`Detector`, `Twin`, `LlmClient`, `Scorer`)
that the project's parallel-work-split plan defines, so any of these swaps is a drop-in replacement —
no changes needed to `plan/planner.py` or the CLI/dashboard.

## Notes
- The z-score detector is intentionally simple and will over-trigger on nominal runs at default
  settings (window=200, z_thresh=3.5) — tune per-channel thresholds once real telemetry (ESA-ADB) is wired in.
- Postgres models exist in `db/models.py` but the CLI demo path currently persists via the filesystem
  `RunStore` for simplicity; switching the planner to also write through `db/models.py` is a small follow-up.
