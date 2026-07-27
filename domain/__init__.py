"""Domain entities — shared vocabulary across all modules (immutable dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


def new_id() -> str:
    return str(uuid.uuid4())


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class EventOfInterest:
    id: str
    run_id: str
    channel: str
    start_ts: datetime
    end_ts: datetime
    score: float
    severity: Severity
    detector_name: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FaultParameter:
    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True)
class SimMapping:
    subsystem: str
    fault_params: tuple[FaultParameter, ...]
    seed: int = 0


@dataclass(frozen=True)
class Hypothesis:
    id: str
    event_id: str
    text: str
    mechanism: str
    fault_params: tuple[FaultParameter, ...]
    prior: float
    generator: str  # "template" | "llm"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class SimResult:
    hypothesis_id: str
    distance: float          # lower = closer match between real and simulated telemetry
    posterior: float         # normalized belief after comparing against all candidate hypotheses
    n_sims: int
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: datetime
    dataset: str
    detector_name: str
    twin_name: str
    llm_name: str
    notes: str = ""


class CouncilRole(str, Enum):
    SYSTEMS_ENGINEER = "systems_engineer"
    DATA_QUALITY_ANALYST = "data_quality_analyst"
    TWIN_PHYSICS_VERIFIER = "twin_physics_verifier"
    RED_TEAM_SKEPTIC = "red_team_skeptic"


class CouncilVerdict(str, Enum):
    UNANIMOUS_AGREEMENT = "unanimous_agreement"
    STRONG_CONSENSUS = "strong_consensus"
    SPLIT_COUNCIL = "split_council"
    REJECTED_BY_COUNCIL = "rejected_by_council"


class ValidationStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    ESCALATED_PENDING_HUMAN = "escalated_pending_human"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    HUMAN_OVERRIDDEN = "human_overridden"


@dataclass(frozen=True)
class CouncilVote:
    role: CouncilRole
    agrees_with_top_hyp: bool
    confidence: float
    rationale: str


@dataclass(frozen=True)
class CouncilConsensus:
    consensus_score: float
    verdict: CouncilVerdict
    summary: str
    individual_votes: tuple[CouncilVote, ...]

