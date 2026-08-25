"""Typer CLI — the single entry point for running the closed loop locally.

Usage:
    python -m cli.main generate-data
    python -m cli.main run --dataset data/synthetic/run_001_friction_increase.csv
    python -m cli.main run-all
    python -m cli.main twin --fault rw_friction --magnitude 8
    python -m cli.main triage --telemetry opssat_ad
    python -m cli.main claim-pack <run_id> --format json
    python -m cli.main rerun <run_id>
    python -m cli.main serve
    python -m cli.main export
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from dotenv import load_dotenv

from ingest.synthetic_generator import generate_dataset
from plan.planner import run_closed_loop

load_dotenv()

app = typer.Typer(help="spaceThink MVP — closed-loop EXHYTE agent (offline, synthetic data)")


# ────────────────────────────────────────────────────────────────────────────
#  Existing commands
# ────────────────────────────────────────────────────────────────────────────

@app.command()
def generate_data(out_dir: str = "data/synthetic", n_runs: int = 12):
    """Generate synthetic reaction-wheel telemetry (nominal + 3 fault types)."""
    generate_dataset(out_dir=out_dir, n_runs=n_runs)


@app.command()
def run(dataset: str, groq_key: Optional[str] = None, detector: Optional[str] = None, scorer: Optional[str] = None):
    """Run the closed loop (detect -> hypothesize -> twin-test -> score) on one CSV.

    --detector zscore|telemanom   --scorer signature|distance|sbi  (defaults: zscore, signature)
    """
    from plan.components import build_detector, build_scorer
    df = pd.read_csv(dataset)
    report = run_closed_loop(df, groq_api_key=groq_key, detector=build_detector(detector), scorer=build_scorer(scorer))
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command()
def run_all(data_dir: str = "data/synthetic", out_file: str = "data/reports.json", groq_key: Optional[str] = None, detector: Optional[str] = None, scorer: Optional[str] = None):
    """Run the closed loop over every synthetic CSV and write a combined report."""
    from plan.components import build_detector, build_scorer
    det, sc = build_detector(detector), build_scorer(scorer)
    reports = []
    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        report = run_closed_loop(df, groq_api_key=groq_key, detector=det, scorer=sc)
        report["source_file"] = csv_path.name
        reports.append(report)
        typer.echo(f"done: {csv_path.name} -> {report['n_events']} events")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(reports, indent=2, default=str))
    typer.echo(f"wrote {len(reports)} reports to {out_file}")


# ────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Twin run
# ────────────────────────────────────────────────────────────────────────────

@app.command()
def twin(
    fault: str = typer.Option("rw_friction", help="Fault type: rw_friction, encoder_dropout, stiction"),
    magnitude: float = typer.Option(0.8, help="Fault magnitude multiplier"),
    duration: float = typer.Option(5000, help="Simulation duration in seconds"),
    n_sims: int = typer.Option(1, help="Number of ensemble simulations"),
    output: Optional[str] = typer.Option(None, help="Output CSV path"),
    use_basilisk: bool = typer.Option(False, help="Use BasiliskTwin (requires Basilisk)"),
):
    """Run a digital twin simulation with specified fault parameters."""
    from domain import FaultParameter, SimMapping

    fault_map = {
        "rw_friction": "friction",
        "encoder_dropout": "dropout_rate",
        "stiction": "stiction_rate",
    }

    param_name = fault_map.get(fault, fault)
    mapping = SimMapping(
        subsystem="reaction_wheel",
        fault_params=(FaultParameter(param_name, magnitude),),
    )

    if use_basilisk:
        from twin.basilisk_twin import BasiliskTwin
        sim = BasiliskTwin().configure(mapping)
    else:
        from twin.simulator import ToySimulator
        sim = ToySimulator().configure(mapping)

    if n_sims > 1:
        results = sim.run_ensemble(n_sims=n_sims, duration_s=duration)
        typer.echo(f"Ran {n_sims} ensemble simulations ({duration}s each)")
        if output:
            results[0].to_csv(output, index=False)
            typer.echo(f"Saved first sim to {output}")
    else:
        result = sim.run(duration_s=duration, seed=42)
        typer.echo(f"Ran 1 simulation ({duration}s, {len(result)} samples)")
        if output:
            result.to_csv(output, index=False)
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(result.describe().to_string())


# ────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Triage
# ────────────────────────────────────────────────────────────────────────────

@app.command()
def triage(
    telemetry: str = typer.Option("opssat_ad", help="Telemetry source: opssat_ad or path to CSV"),
    groq_key: Optional[str] = typer.Option(None, help="Groq API key"),
):
    """Run bulk triage — auto-explain + escalate."""
    from explore.detector import ZScoreDetector
    from dashboard.triage_mode import TriageDashboardEngine, render_triage_page

    if telemetry == "opssat_ad":
        from ingest.opssat_ad import load_opssat_ad, opssat_to_pipeline_format
        dataset = load_opssat_ad()
        df = opssat_to_pipeline_format(dataset)
    else:
        df = pd.read_csv(telemetry)

    detector = ZScoreDetector()
    channels = ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]
    events = detector.detect(df, channels, run_id="triage-run")

    engine = TriageDashboardEngine()
    summary = engine.run_bulk_triage(events)

    typer.echo(render_triage_page(summary))


# ────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Claim pack
# ────────────────────────────────────────────────────────────────────────────

@app.command(name="claim-pack")
def claim_pack(
    run_id: str = typer.Argument(..., help="Run ID to generate claim pack for"),
    format: str = typer.Option("json", help="Output format: json or text"),
):
    """Generate an insurance/regulator claim pack for a run."""
    from runstore import RunStore
    from runstore.ledger import AuditLedger
    from cli.claim_pack import build_evidence_graph, export_claim_pack_json, export_claim_pack_text

    store = RunStore()
    ledger = AuditLedger()

    # Load run data
    from runstore import ArtifactRef
    hyp_refs = store.list(run_id, "hypotheses")
    sim_refs = store.list(run_id, "sim_results")

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

    graph = build_evidence_graph(run_id, hypotheses, sim_results, store)

    if format == "text":
        path = export_claim_pack_text(run_id, graph, ledger)
    else:
        path = export_claim_pack_json(run_id, graph, ledger)

    typer.echo(f"Claim pack written to {path}")


# ────────────────────────────────────────────────────────────────────────────
#  Phase 5 — Rerun
# ────────────────────────────────────────────────────────────────────────────

@app.command()
def rerun(
    run_id: str = typer.Argument(..., help="Run ID to replay"),
    diff: Optional[str] = typer.Option(None, help="Compare with another run ID"),
):
    """Deterministic run replay from runstore, or diff two runs."""
    from cli.rerun import rerun_from_store, diff_runs, format_diff_report

    if diff:
        result = diff_runs(run_id, diff)
        typer.echo(format_diff_report(result))
    else:
        report = rerun_from_store(run_id)
        typer.echo(json.dumps(report, indent=2, default=str))


# ────────────────────────────────────────────────────────────────────────────
#  Phase 4 & UI — Serve, Dashboard & Unified Stack
# ────────────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Start the FastAPI server."""
    import uvicorn
    typer.echo(f"Starting spaceThink API on {host}:{port}")
    uvicorn.run("api.app:app", host=host, port=port, reload=True)


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port for Streamlit dashboard"),
):
    """Start the Streamlit EXHYTE dashboard."""
    import subprocess
    import sys
    typer.echo(f"Starting spaceThink Dashboard on http://localhost:{port}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(port)])


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port_api: int = typer.Option(8000, help="Port for FastAPI server"),
    port_dashboard: int = typer.Option(8501, help="Port for Streamlit dashboard"),
    api_only: bool = typer.Option(False, help="Launch only the FastAPI server"),
    dashboard_only: bool = typer.Option(False, help="Launch only the Streamlit dashboard"),
    no_browser: bool = typer.Option(False, help="Do not open browser automatically"),
    generate_data: bool = typer.Option(False, help="Force regenerate synthetic datasets and reports"),
):
    """Start the entire spaceThink application stack (API + Streamlit UI Dashboard)."""
    from run import run_stack
    run_stack(
        host=host,
        port_api=port_api,
        port_dashboard=port_dashboard,
        api_only=api_only,
        dashboard_only=dashboard_only,
        open_browser=not no_browser,
        force_generate_data=generate_data,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Phase 6 — Export
# ────────────────────────────────────────────────────────────────────────────

@app.command()
def export(
    format: str = typer.Option("numpy", help="Export format: numpy or onnx"),
    model_name: str = typer.Option("telemanom_forecaster", help="Model name"),
):
    """Export forecaster model for edge deployment."""
    from explore.export import export_forecaster_numpy, export_forecaster_onnx, format_export_report

    if format == "onnx":
        report = export_forecaster_onnx(model_name=model_name)
    else:
        report = export_forecaster_numpy(model_name=model_name)

    typer.echo(format_export_report(report))


if __name__ == "__main__":
    app()

