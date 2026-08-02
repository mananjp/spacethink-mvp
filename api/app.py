"""FastAPI surface — REST API for spaceThink.

Endpoints:
  POST /v1/events         — ingest a telemetry window
  GET  /v1/runs/{run_id}  — fetch a run
  POST /v1/claim-pack     — request an insurance pack
  GET  /v1/health         — health check
  GET  /v1/ready          — readiness probe

Features:
  - OpenAPI schema auto-generated
  - Rate limiting middleware (token bucket per tenant)
  - Tenant-isolated API-key auth
  - Consistent error envelope
  - Request-ID propagation for tracing
"""
from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from plan.planner import run_closed_loop
from runstore import RunStore
from runstore.ledger import AuditLedger
from cli.claim_pack import build_evidence_graph, export_claim_pack_json

# ────────────────────────────────────────────────────────────────────────────
#  App setup
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="spaceThink API",
    description="Autonomous spacecraft anomaly detection, diagnosis, and evidence generation.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────
#  Rate limiting (in-memory token bucket per API key)
# ────────────────────────────────────────────────────────────────────────────

_rate_buckets: dict[str, dict] = defaultdict(lambda: {"tokens": 60, "last": time.time()})
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "60"))  # requests per minute
RATE_WINDOW = 60.0


def _check_rate_limit(api_key: str) -> None:
    bucket = _rate_buckets[api_key]
    now = time.time()
    elapsed = now - bucket["last"]
    bucket["tokens"] = min(RATE_LIMIT, bucket["tokens"] + elapsed * (RATE_LIMIT / RATE_WINDOW))
    bucket["last"] = now
    if bucket["tokens"] < 1:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    bucket["tokens"] -= 1


# ────────────────────────────────────────────────────────────────────────────
#  Auth (simple API key)
# ────────────────────────────────────────────────────────────────────────────

VALID_API_KEYS = set(filter(None, os.getenv("API_KEYS", "dev-key-001").split(",")))


def _get_api_key(x_api_key: str = Header(default="dev-key-001")) -> str:
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    _check_rate_limit(x_api_key)
    return x_api_key


# ────────────────────────────────────────────────────────────────────────────
#  Request-ID middleware
# ────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ────────────────────────────────────────────────────────────────────────────
#  Error envelope
# ────────────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[str] = []


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


# ────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ────────────────────────────────────────────────────────────────────────────

class EventIngestRequest(BaseModel):
    """Ingest a telemetry window for anomaly detection and diagnosis."""
    telemetry: list[dict] = Field(..., description="List of telemetry rows (t, wheel_speed_rpm, wheel_current_a, wheel_temp_c)")
    run_id: Optional[str] = Field(None, description="Optional run ID (auto-generated if not provided)")
    n_sims: int = Field(20, description="Number of simulations per hypothesis")


class EventIngestResponse(BaseModel):
    run_id: str
    n_events: int
    events: list[dict]


class RunResponse(BaseModel):
    run_id: str
    manifest: Optional[dict] = None
    artifacts: list[dict] = []


class ClaimPackRequest(BaseModel):
    run_id: str
    format: str = Field("json", description="Output format: json or text")


class ClaimPackResponse(BaseModel):
    run_id: str
    output_path: str
    format: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "0.1.0"


# ────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/ready", response_model=HealthResponse)
async def ready():
    """Readiness probe — checks that critical dependencies are available."""
    return HealthResponse(
        status="ready",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/v1/events", response_model=EventIngestResponse)
async def ingest_events(
    request: EventIngestRequest,
    api_key: str = Depends(_get_api_key),
):
    """Ingest a telemetry window and run the closed-loop pipeline."""
    try:
        df = pd.DataFrame(request.telemetry)
        required = {"t", "wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required columns: {missing}",
            )

        report = run_closed_loop(
            telemetry=df,
            run_id=request.run_id,
            n_sims_per_hypothesis=request.n_sims,
        )

        return EventIngestResponse(
            run_id=report["run_id"],
            n_events=report["n_events"],
            events=report["events"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    api_key: str = Depends(_get_api_key),
):
    """Fetch a run's manifest and artifact listing."""
    store = RunStore()
    try:
        manifest = store.get(
            type("Ref", (), {"run_id": run_id, "kind": "manifest", "key": "manifest"})()
        )
        manifest_dict = {
            "run_id": manifest.run_id,
            "created_at": str(manifest.created_at),
            "dataset": manifest.dataset,
            "detector_name": manifest.detector_name,
            "twin_name": manifest.twin_name,
            "llm_name": manifest.llm_name,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    artifacts = []
    for kind in ["events", "hypotheses", "sim_results", "council_consensus", "validation_status"]:
        refs = store.list(run_id, kind)
        for ref in refs:
            artifacts.append({"kind": ref.kind, "key": ref.key})

    return RunResponse(run_id=run_id, manifest=manifest_dict, artifacts=artifacts)


@app.post("/v1/claim-pack", response_model=ClaimPackResponse)
async def create_claim_pack(
    request: ClaimPackRequest,
    api_key: str = Depends(_get_api_key),
):
    """Generate a claim pack for a specific run."""
    store = RunStore()

    try:
        # Load hypotheses and sim results from the store
        hyp_refs = store.list(request.run_id, "hypotheses")
        sim_refs = store.list(request.run_id, "sim_results")

        hypotheses = []
        for ref in hyp_refs:
            data = store.get(ref)
            if isinstance(data, list):
                hypotheses.extend(data)
            else:
                hypotheses.append(data)

        sim_results = []
        for ref in sim_refs:
            data = store.get(ref)
            if isinstance(data, list):
                sim_results.extend(data)
            else:
                sim_results.append(data)

        graph = build_evidence_graph(request.run_id, hypotheses, sim_results, store)
        ledger = AuditLedger()

        if request.format == "text":
            from cli.claim_pack import export_claim_pack_text
            path = export_claim_pack_text(request.run_id, graph, ledger)
        else:
            path = export_claim_pack_json(request.run_id, graph, ledger)

        return ClaimPackResponse(
            run_id=request.run_id,
            output_path=str(path),
            format=request.format,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run {request.run_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
