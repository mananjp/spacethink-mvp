"""Claim-pack export — regulator/underwriter-friendly artifact generation.

spacethink claim-pack <run_id> --format pdf,json → artifact bundle linking
every claim to its evidence (claim-evidence graph) + audit ledger digest.

Per-mechanistic-step uncertainty surfaced per the 2026 uncertainty-granularity
finding.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from domain import (
    ClaimNode,
    EvidenceEdge,
    EvidenceGraph,
    Hypothesis,
    SimResult,
    new_id,
)
from runstore import RunStore
from runstore.ledger import AuditLedger


def build_evidence_graph(
    run_id: str,
    hypotheses: list[Hypothesis],
    sim_results: list[SimResult],
    store: RunStore | None = None,
    ledger: AuditLedger | None = None,
) -> EvidenceGraph:
    """Build a claim-evidence graph from a run's hypotheses and results.

    Each hypothesis becomes a claim node with per-mechanistic-step uncertainty.
    Each SimResult and its diagnostics become evidence edges.
    """
    claims = []
    edges = []

    # Pair hypotheses with their sim results
    result_map = {r.hypothesis_id: r for r in sim_results}

    for hyp in hypotheses:
        result = result_map.get(hyp.id)
        if not result:
            continue

        # Per-mechanistic-step uncertainty from the posterior
        uncertainty = 1.0 - result.posterior

        claim_id = new_id()
        claim = ClaimNode(
            id=claim_id,
            claim_text=(
                f"Mechanism '{hyp.mechanism}' explains the observed telemetry anomaly "
                f"with posterior probability {result.posterior:.3f}."
            ),
            mechanism_step=hyp.mechanism,
            uncertainty=round(uncertainty, 4),
            evidence_ids=(hyp.id, result.hypothesis_id),
        )
        claims.append(claim)

        # Evidence edge: simulation result
        edges.append(EvidenceEdge(
            claim_id=claim_id,
            evidence_artifact_hash=f"sim_{result.hypothesis_id}",
            evidence_type="sim_result",
            strength=result.posterior,
        ))

        # Evidence edge: hypothesis parameters
        for fp in hyp.fault_params:
            edges.append(EvidenceEdge(
                claim_id=claim_id,
                evidence_artifact_hash=f"param_{fp.name}_{fp.value}",
                evidence_type="telemetry",
                strength=0.8,
            ))

    return EvidenceGraph(
        run_id=run_id,
        claims=tuple(claims),
        edges=tuple(edges),
    )


def export_claim_pack_json(
    run_id: str,
    evidence_graph: EvidenceGraph,
    ledger: AuditLedger | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Export claim pack as JSON.

    Includes: evidence graph, ledger digest, per-claim uncertainty,
    and full provenance chain.
    """
    output_dir = output_dir or Path("data") / "claim_packs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pack = {
        "version": "1.0",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims": [
            {
                "id": c.id,
                "claim_text": c.claim_text,
                "mechanism_step": c.mechanism_step,
                "uncertainty": c.uncertainty,
                "evidence_ids": list(c.evidence_ids),
            }
            for c in evidence_graph.claims
        ],
        "evidence_edges": [
            {
                "claim_id": e.claim_id,
                "evidence_artifact_hash": e.evidence_artifact_hash,
                "evidence_type": e.evidence_type,
                "strength": e.strength,
            }
            for e in evidence_graph.edges
        ],
        "audit_ledger_digest": ledger.digest() if ledger else "no_ledger",
        "audit_chain_valid": ledger.verify_chain()[0] if ledger else None,
    }

    output_path = output_dir / f"claim_pack_{run_id[:12]}.json"
    output_path.write_text(json.dumps(pack, indent=2, default=str), encoding="utf-8")

    return output_path


def export_claim_pack_text(
    run_id: str,
    evidence_graph: EvidenceGraph,
    ledger: AuditLedger | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Export claim pack as human-readable text (Markdown).

    Suitable for regulators and underwriters who need to verify the
    chain without technical context.
    """
    output_dir = output_dir or Path("data") / "claim_packs"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# spaceThink Claim Pack — Run `{run_id[:12]}`",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    # Audit chain status
    if ledger:
        valid, broken_at = ledger.verify_chain()
        status = "✅ VALID" if valid else f"❌ BROKEN at seq {broken_at}"
        lines.extend([
            "## Audit Chain",
            f"**Status:** {status}",
            f"**Digest:** `{ledger.digest()[:16]}...`",
            "",
        ])

    # Claims with uncertainty
    lines.extend([
        "## Claims & Evidence",
        "",
        "| # | Mechanism | Claim | Uncertainty | Evidence |",
        "|---|-----------|-------|-------------|----------|",
    ])

    for i, claim in enumerate(evidence_graph.claims, 1):
        n_evidence = sum(1 for e in evidence_graph.edges if e.claim_id == claim.id)
        uncertainty_bar = "█" * int(claim.uncertainty * 10) + "░" * (10 - int(claim.uncertainty * 10))
        lines.append(
            f"| {i} | `{claim.mechanism_step}` | {claim.claim_text[:60]}... | "
            f"`{uncertainty_bar}` {claim.uncertainty:.1%} | {n_evidence} artifacts |"
        )

    lines.extend([
        "",
        "## Per-Mechanistic-Step Uncertainty",
        "",
    ])

    for claim in evidence_graph.claims:
        confidence = 1.0 - claim.uncertainty
        color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        lines.append(
            f"- {color} **{claim.mechanism_step}**: "
            f"Confidence {confidence:.1%} (Uncertainty: {claim.uncertainty:.1%})"
        )

    output_path = output_dir / f"claim_pack_{run_id[:12]}.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return output_path

