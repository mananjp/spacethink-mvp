"""Domain entities — shared vocabulary across all modules (immutable dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol, runtime_checkable
import uuid


def new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Timezone-aware UTC now (replaces deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)


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
    created_at: datetime = field(default_factory=_utcnow)


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


# ---------------------------------------------------------------------------
#  Phase 2 — Telecommand model for auto-explain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Telecommand:
    """A spacecraft telecommand record used for auto-explain matching."""
    id: str
    name: str
    subsystem: str
    timestamp: datetime
    parameters: dict = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class TriageResult:
    """Result of bulk event triage — auto-explained vs escalated."""
    event_id: str
    auto_explained: bool
    explanation: str
    matching_telecommand_id: Optional[str] = None


# ---------------------------------------------------------------------------
#  Phase 3 — Audit ledger, Claim-evidence graph, Adjudicator protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditEntry:
    """Single entry in the append-only hash-chained audit ledger."""
    prev_hash: str
    ts: datetime
    actor: str
    kind: str
    artifact_hash: str
    prompt_version: str = ""
    model_version: str = ""
    sim_params: dict = field(default_factory=dict)
    seed: int = 0


@dataclass(frozen=True)
class ClaimNode:
    """A single claim in a claim-evidence graph."""
    id: str
    claim_text: str
    mechanism_step: str
    uncertainty: float          # per-mechanistic-step uncertainty
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceEdge:
    """Links a claim to its supporting evidence artifact."""
    claim_id: str
    evidence_artifact_hash: str
    evidence_type: str          # "sim_result" | "telemetry" | "telecommand" | "ledger"
    strength: float = 1.0


@dataclass(frozen=True)
class EvidenceGraph:
    """Complete claim-evidence graph for an insurance/audit pack."""
    run_id: str
    claims: tuple[ClaimNode, ...]
    edges: tuple[EvidenceEdge, ...]


class Adjudicator(Protocol):
    """Protocol for per-mechanistic-step uncertainty + claim-evidence evaluation."""

    def adjudicate(
        self,
        event: EventOfInterest,
        hypothesis: Hypothesis,
        sim_result: SimResult,
        consensus: CouncilConsensus,
    ) -> EvidenceGraph:
        ...


# ---------------------------------------------------------------------------
#  Phase 4 — Calibrated twin parameters, Fleet clustering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibratedTwinParams:
    """Parameters auto-calibrated from customer telemetry for BasiliskTwin."""
    customer_id: str
    subsystem: str
    fidelity_tier: int          # 1 = parametric auto-calibrated, 2 = high-fidelity import
    parameters: dict = field(default_factory=dict)
    calibrated_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class FleetCluster:
    """A cluster of similar anomaly signatures across a fleet."""
    cluster_id: str
    event_ids: tuple[str, ...]
    representative_mechanism: str
    centroid_distance: float
    recommended_action: str = ""


# ---------------------------------------------------------------------------
#  Phase 3.5 — Domain-Agnostic, Calibration-Gated Verification Contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    """Container for heterogeneous domain evidence (telemetry DF, optical spectrum, catalog matches)."""
    domain: str
    raw_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """Result of a domain verification step."""
    hypothesis_id: str
    verified: bool
    fit_score: float
    posterior: float
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationStatus:
    """Standardized calibration boundary object crossing the domain boundary."""
    domain: str
    passed: bool
    confidence: float        # Calibrated confidence (e.g. from SBC/PPC or catalog benchmark), NOT self-reported
    method: str              # e.g., "SBC+PPC", "catalog cross-match"
    diagnostics: dict = field(default_factory=dict)


class OversightMode(str, Enum):
    ACTIVE = "active"     # Hold for human approval before taking action/claim
    PASSIVE = "passive"   # Proceed, log to evidence ledger, notify async


@dataclass(frozen=True)
class OversightPolicy:
    """Policy governing autonomy levels based strictly on CalibrationStatus."""
    min_confidence: float = 0.8
    require_calibration: bool = True


@runtime_checkable
class Verifier(Protocol):
    """Domain-specific hypothesis verification protocol. One implementation per domain.
    Downstream gating code never branches on domain.
    """

    def verify(self, hypothesis: Hypothesis, evidence: Evidence) -> VerificationResult:
        ...

    def calibration_status(self) -> CalibrationStatus:
        ...


