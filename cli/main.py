"""Typer CLI — the single entry point for running the closed loop locally.

Usage:
    python -m cli.main generate-data
    python -m cli.main run --dataset data/synthetic/run_001_friction_increase.csv
    python -m cli.main run-all
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


@app.command()
def generate_data(out_dir: str = "data/synthetic", n_runs: int = 12):
    """Generate synthetic reaction-wheel telemetry (nominal + 3 fault types)."""
    generate_dataset(out_dir=out_dir, n_runs=n_runs)


@app.command()
def run(dataset: str, groq_key: Optional[str] = None):
    """Run the closed loop (detect -> hypothesize -> twin-test -> score) on one CSV."""
    df = pd.read_csv(dataset)
    report = run_closed_loop(df, groq_api_key=groq_key)
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command()
def run_all(data_dir: str = "data/synthetic", out_file: str = "data/reports.json", groq_key: Optional[str] = None):
    """Run the closed loop over every synthetic CSV and write a combined report."""
    reports = []
    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        report = run_closed_loop(df, groq_api_key=groq_key)
        report["source_file"] = csv_path.name
        reports.append(report)
        typer.echo(f"done: {csv_path.name} -> {report['n_events']} events")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(reports, indent=2, default=str))
    typer.echo(f"wrote {len(reports)} reports to {out_file}")


if __name__ == "__main__":
    app()
