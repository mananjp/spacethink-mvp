"""Triage dashboard mode — bulk triage and per-event deep-dive views.

Page 1 — Bulk triage: "1 day × 1 fleet → 412 alerts → 406 auto-explained,
                        6 escalated with verified cause."
Page 2 — Per-event deep-dive: refactored from existing dashboard/app.py.

Per-event cost ceiling: ≤ $0.20 LLM spend, ≤ 60s compute.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from domain import EventOfInterest, Telecommand, TriageResult
from hypothesize.telecommand_explainer import TelecommandExplainer


class TriageDashboardEngine:
    """Backend engine for the triage dashboard mode.

    Processes a batch of events through the telecommand auto-explain layer
    and prepares data structures for the Streamlit dashboard views.
    """

    def __init__(
        self,
        telecommands: list[Telecommand] | None = None,
        cost_ceiling_per_event: float = 0.20,
        compute_ceiling_s: float = 60.0,
    ):
        self.explainer = TelecommandExplainer(telecommands=telecommands)
        self.cost_ceiling = cost_ceiling_per_event
        self.compute_ceiling = compute_ceiling_s

    def run_bulk_triage(
        self,
        events: list[EventOfInterest],
    ) -> dict:
        """Run bulk triage on a list of events.

        Returns a summary dict suitable for dashboard rendering.
        """
        results = self.explainer.bulk_triage(events)
        summary = self.explainer.triage_summary(results)

        # Enrich with severity breakdown
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        escalated_by_severity = {"high": 0, "medium": 0, "low": 0}

        for event, result in zip(events, results):
            sev = event.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if not result.auto_explained:
                escalated_by_severity[sev] = escalated_by_severity.get(sev, 0) + 1

        summary["severity_breakdown"] = severity_counts
        summary["escalated_by_severity"] = escalated_by_severity
        summary["triage_results"] = results

        return summary

    def prepare_deep_dive(
        self,
        event: EventOfInterest,
        telemetry: pd.DataFrame,
        report_event: dict,
    ) -> dict:
        """Prepare data for per-event deep-dive view.

        Collates telemetry window, hypothesis rankings, council deliberation,
        and EIG planner recommendations.
        """
        # Extract event telemetry window
        event_duration = (event.end_ts - event.start_ts).total_seconds()
        window_start = max(0, int(event_duration) - 500)
        window_end = min(len(telemetry), int(event_duration) + 500)

        window_df = telemetry.iloc[window_start:window_end].copy()

        return {
            "event": {
                "id": event.id,
                "channel": event.channel,
                "severity": event.severity.value,
                "score": event.score,
                "detector": event.detector_name,
            },
            "telemetry_window": window_df,
            "diagnosis": {
                "top_hypothesis": report_event.get("top_hypothesis"),
                "posterior": report_event.get("posterior"),
                "ranked_mechanisms": report_event.get("ranked_mechanisms", []),
            },
            "council": report_event.get("council_consensus", {}),
            "validation_status": report_event.get("validation_status"),
            "timeline": report_event.get("ai_timeline", ""),
        }


def render_triage_page(summary: dict) -> str:
    """Render the bulk triage summary as Markdown for Streamlit display."""
    total = summary["total_events"]
    explained = summary["auto_explained"]
    escalated = summary["escalated"]
    rate = summary["auto_explain_rate"]

    lines = [
        "# 🛸 Bulk Triage Summary",
        "",
        f"**Total Alerts:** {total}",
        f"**Auto-Explained:** {explained} ({rate:.0%})",
        f"**Escalated:** {escalated}",
        "",
        "## Severity Breakdown",
        "",
        "| Severity | Total | Escalated |",
        "|----------|-------|-----------|",
    ]

    for sev in ["high", "medium", "low"]:
        total_sev = summary["severity_breakdown"].get(sev, 0)
        esc_sev = summary["escalated_by_severity"].get(sev, 0)
        icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "🟢"
        lines.append(f"| {icon} {sev.upper()} | {total_sev} | {esc_sev} |")

    if escalated > 0:
        lines.extend([
            "",
            "## ⚠️ Escalated Events",
            "",
        ])
        for result in summary.get("escalated_events", []):
            lines.append(f"- **{result.event_id[:8]}**: {result.explanation}")

    if explained > 0:
        lines.extend([
            "",
            "## ✅ Auto-Explained Events (sample)",
            "",
        ])
        for result in summary.get("explained_events", [])[:5]:
            tc_id = result.matching_telecommand_id or "N/A"
            lines.append(f"- **{result.event_id[:8]}** → TC: `{tc_id}` — {result.explanation[:100]}...")

    return "\n".join(lines)
