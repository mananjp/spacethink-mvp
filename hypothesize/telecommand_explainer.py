"""Telecommand Explainer — benign auto-explain pass.

Detected events that align with active telecommands get routed to "expected
nominal" with citation (telecommand record + retrieved doc). This is the
single most persuasive YC moment: "this is a commanded reaction-wheel bias
swap, not a fault."

For v1, a small static telecommand table covers the YC case; later,
knowledge/ feeds this.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from domain import EventOfInterest, Telecommand, TriageResult, new_id


# ────────────────────────────────────────────────────────────────────────────
#  Static telecommand table (v1 — covers the YC demo case)
# ────────────────────────────────────────────────────────────────────────────

STATIC_TC_TABLE = [
    {
        "name": "RW_BIAS_SWAP",
        "subsystem": "ADCS",
        "description": "Reaction wheel bias momentum swap — redistributes angular momentum "
                       "across wheel set. Expected speed transients on all 4 wheels.",
        "affected_channels": ["wheel_speed_rpm", "wheel_current_a"],
        "expected_duration_s": 120,
    },
    {
        "name": "RW_DESAT",
        "subsystem": "ADCS",
        "description": "Reaction wheel desaturation via magnetorquer. Speed ramp-down "
                       "followed by magnetic torque compensation.",
        "affected_channels": ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"],
        "expected_duration_s": 300,
    },
    {
        "name": "ATTITUDE_SLEW",
        "subsystem": "ADCS",
        "description": "Commanded attitude slew maneuver. Reaction wheels will show "
                       "acceleration/deceleration profiles during the maneuver.",
        "affected_channels": ["wheel_speed_rpm", "wheel_current_a"],
        "expected_duration_s": 180,
    },
    {
        "name": "SAFE_MODE_ENTRY",
        "subsystem": "ADCS",
        "description": "Autonomous safe mode entry — wheels ramped to sun-pointing hold. "
                       "Expected large speed change and current spike.",
        "affected_channels": ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"],
        "expected_duration_s": 60,
    },
    {
        "name": "THERMAL_CONTROL",
        "subsystem": "TCS",
        "description": "Heater activation/deactivation for thermal control. "
                       "Expected temperature transient in affected subsystem.",
        "affected_channels": ["wheel_temp_c"],
        "expected_duration_s": 600,
    },
]


@dataclass
class TelecommandMatch:
    """A match between an event and a telecommand."""
    telecommand: Telecommand | dict
    overlap_seconds: float
    confidence: float
    explanation: str


class TelecommandExplainer:
    """Auto-explain layer matching detected events to telecommand activity.

    Routes events aligned with known telecommands to "expected nominal" with
    full citation chain (telecommand record + description).
    """

    def __init__(
        self,
        telecommands: list[Telecommand] | None = None,
        tc_table: list[dict] | None = None,
        window_s: int = 120,
    ):
        """
        Parameters
        ----------
        telecommands : list of Telecommand records with timestamps
        tc_table : static telecommand lookup table (defaults to STATIC_TC_TABLE)
        window_s : time window (seconds) for matching events to TCs
        """
        self.telecommands = telecommands or []
        self.tc_table = tc_table or STATIC_TC_TABLE
        self.window_s = window_s

    def _find_matching_tc(self, event: EventOfInterest) -> TelecommandMatch | None:
        """Find the best-matching telecommand for an event."""
        best_match: TelecommandMatch | None = None
        best_confidence = 0.0

        for tc in self.telecommands:
            # Check time overlap
            tc_start = tc.timestamp
            # Look up expected duration from TC table
            tc_entry = next(
                (t for t in self.tc_table if tc.name.startswith(t["name"])),
                None,
            )
            expected_duration = tc_entry["expected_duration_s"] if tc_entry else self.window_s
            tc_end = tc_start + timedelta(seconds=expected_duration)

            # Check if event falls within TC window (± window_s buffer)
            buffer = timedelta(seconds=self.window_s)
            if event.start_ts > tc_end + buffer or event.end_ts < tc_start - buffer:
                continue

            # Check channel match
            if tc_entry and event.channel not in tc_entry.get("affected_channels", []):
                continue

            # Compute overlap
            overlap_start = max(event.start_ts, tc_start)
            overlap_end = min(event.end_ts, tc_end)
            overlap_s = max(0, (overlap_end - overlap_start).total_seconds())

            event_duration = max(1, (event.end_ts - event.start_ts).total_seconds())
            overlap_frac = overlap_s / event_duration

            # Confidence based on overlap and channel match
            confidence = min(1.0, overlap_frac * 1.2)
            if tc_entry:
                confidence = min(1.0, confidence + 0.2)  # bonus for known TC type

            if confidence > best_confidence:
                explanation = (
                    f"Event on channel '{event.channel}' aligns with telecommand "
                    f"'{tc.name}' (subsystem: {tc.subsystem}). "
                )
                if tc_entry:
                    explanation += tc_entry["description"]
                else:
                    explanation += f"Description: {tc.description}"

                best_match = TelecommandMatch(
                    telecommand=tc,
                    overlap_seconds=overlap_s,
                    confidence=confidence,
                    explanation=explanation,
                )
                best_confidence = confidence

        return best_match

    def _check_static_table(self, event: EventOfInterest) -> TelecommandMatch | None:
        """Fallback: check if event channel matches a known TC pattern.

        Used when no explicit telecommand records are available but the
        event pattern matches a known TC type signature.
        """
        for tc_entry in self.tc_table:
            if event.channel in tc_entry.get("affected_channels", []):
                # Low confidence match based on channel alone
                if event.severity.value == "low" and event.score < 2.0:
                    return TelecommandMatch(
                        telecommand=tc_entry,
                        overlap_seconds=0,
                        confidence=0.3,
                        explanation=(
                            f"Low-severity event on channel '{event.channel}' is consistent with "
                            f"'{tc_entry['name']}' operations. {tc_entry['description']}"
                        ),
                    )
        return None

    def explain(self, event: EventOfInterest) -> TriageResult:
        """Attempt to auto-explain an event as a telecommand-driven nominal operation.

        Returns a TriageResult indicating whether the event was auto-explained
        or should be escalated for further analysis.
        """
        # First, try explicit TC matching
        match = self._find_matching_tc(event)

        # Fallback to static table pattern matching
        if not match:
            match = self._check_static_table(event)

        if match and match.confidence >= 0.5:
            tc_id = (
                match.telecommand.id if isinstance(match.telecommand, Telecommand)
                else match.telecommand.get("name", "unknown")
            )
            return TriageResult(
                event_id=event.id,
                auto_explained=True,
                explanation=match.explanation,
                matching_telecommand_id=tc_id,
            )

        return TriageResult(
            event_id=event.id,
            auto_explained=False,
            explanation="No matching telecommand found. Escalating for hypothesis-driven analysis.",
            matching_telecommand_id=None,
        )

    def bulk_triage(self, events: list[EventOfInterest]) -> list[TriageResult]:
        """Triage a batch of events — auto-explain what we can, escalate the rest."""
        return [self.explain(event) for event in events]

    def triage_summary(self, results: list[TriageResult]) -> dict:
        """Produce a summary of bulk triage results for the dashboard."""
        n_total = len(results)
        n_explained = sum(1 for r in results if r.auto_explained)
        n_escalated = n_total - n_explained

        return {
            "total_events": n_total,
            "auto_explained": n_explained,
            "escalated": n_escalated,
            "auto_explain_rate": n_explained / max(n_total, 1),
            "explained_events": [r for r in results if r.auto_explained],
            "escalated_events": [r for r in results if not r.auto_explained],
        }
