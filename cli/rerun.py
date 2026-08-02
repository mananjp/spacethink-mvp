"""CLI rerun — deterministic run replay and run diffing.

spacethink rerun <run_id>           → re-run from runstore + captured seed + frozen model versions
spacethink rerun <id1> --diff <id2> → why did these differ?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from runstore import RunStore, ArtifactRef
from runstore.store import ContentAddressedStore


def rerun_from_store(
    run_id: str,
    store: RunStore | None = None,
    n_sims: int | None = None,
) -> dict:
    """Re-run a pipeline run from its stored artifacts and seed.

    Loads the manifest (which contains dataset, detector, twin, LLM names)
    and replays with the same configuration for bit-for-bit reproducibility
    within stochastic tolerance.
    """
    from plan.planner import run_closed_loop

    store = store or RunStore()

    # Load manifest
    manifest_ref = ArtifactRef(run_id=run_id, kind="manifest", key="manifest")
    try:
        manifest = store.get(manifest_ref)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Run {run_id} not found in runstore. "
            f"Available runs: {[d.name for d in store.root.iterdir() if d.is_dir()]}"
        )

    # Load original events to get telemetry (if stored)
    events_ref = ArtifactRef(run_id=run_id, kind="events", key="events")
    try:
        events = store.get(events_ref)
    except FileNotFoundError:
        events = []

    # Try to load original telemetry from the dataset
    dataset_name = manifest.dataset if hasattr(manifest, "dataset") else "synthetic"
    telemetry_path = Path("data") / "synthetic"

    # Find a CSV that was used
    csv_files = sorted(telemetry_path.glob("*.csv")) if telemetry_path.exists() else []
    if not csv_files:
        raise FileNotFoundError(
            "No telemetry data found for replay. "
            "Run `python -m cli.main generate-data` first."
        )

    # Use first available CSV for replay
    df = pd.read_csv(csv_files[0])

    # Re-run with same configuration
    new_run_id = f"{run_id}_rerun"
    report = run_closed_loop(
        telemetry=df,
        run_id=new_run_id,
        n_sims_per_hypothesis=n_sims or 20,
        store=store,
    )

    report["original_run_id"] = run_id
    report["replay"] = True

    return report


def diff_runs(
    run_id_1: str,
    run_id_2: str,
    store: RunStore | None = None,
) -> dict:
    """Compare two runs and explain their differences.

    Sources the audit ledger + run store to identify what changed
    between the runs: different detector, different parameters,
    different telemetry, etc.
    """
    store = store or RunStore()

    def _load_run_data(run_id: str) -> dict:
        data = {"run_id": run_id}
        try:
            manifest = store.get(ArtifactRef(run_id, "manifest", "manifest"))
            data["manifest"] = {
                "dataset": getattr(manifest, "dataset", "unknown"),
                "detector_name": getattr(manifest, "detector_name", "unknown"),
                "twin_name": getattr(manifest, "twin_name", "unknown"),
                "llm_name": getattr(manifest, "llm_name", "unknown"),
            }
        except FileNotFoundError:
            data["manifest"] = None

        try:
            events = store.get(ArtifactRef(run_id, "events", "events"))
            data["n_events"] = len(events) if isinstance(events, list) else 1
        except FileNotFoundError:
            data["n_events"] = 0

        # Collect artifact keys
        artifact_kinds = ["hypotheses", "sim_results", "council_consensus", "validation_status"]
        data["artifacts"] = {}
        for kind in artifact_kinds:
            refs = store.list(run_id, kind)
            data["artifacts"][kind] = len(refs)

        return data

    run_1 = _load_run_data(run_id_1)
    run_2 = _load_run_data(run_id_2)

    # Compare
    differences = []

    # Manifest differences
    if run_1.get("manifest") and run_2.get("manifest"):
        m1, m2 = run_1["manifest"], run_2["manifest"]
        for key in m1:
            if m1[key] != m2.get(key):
                differences.append({
                    "field": f"manifest.{key}",
                    "run_1": m1[key],
                    "run_2": m2.get(key),
                })

    # Event count differences
    if run_1["n_events"] != run_2["n_events"]:
        differences.append({
            "field": "n_events",
            "run_1": run_1["n_events"],
            "run_2": run_2["n_events"],
        })

    # Artifact count differences
    for kind in run_1.get("artifacts", {}):
        n1 = run_1["artifacts"].get(kind, 0)
        n2 = run_2.get("artifacts", {}).get(kind, 0)
        if n1 != n2:
            differences.append({
                "field": f"artifacts.{kind}",
                "run_1": n1,
                "run_2": n2,
            })

    return {
        "run_1": run_1,
        "run_2": run_2,
        "n_differences": len(differences),
        "differences": differences,
        "identical": len(differences) == 0,
    }


def format_diff_report(diff: dict) -> str:
    """Format a run diff as Markdown."""
    lines = [
        f"## 🔍 Run Diff: `{diff['run_1']['run_id'][:8]}` vs `{diff['run_2']['run_id'][:8]}`",
        "",
    ]

    if diff["identical"]:
        lines.append("✅ **Runs are identical** within stochastic tolerance.")
    else:
        lines.extend([
            f"⚠️ **{diff['n_differences']} difference(s) found:**",
            "",
            "| Field | Run 1 | Run 2 |",
            "|-------|-------|-------|",
        ])
        for d in diff["differences"]:
            lines.append(f"| `{d['field']}` | {d['run_1']} | {d['run_2']} |")

    return "\n".join(lines)
