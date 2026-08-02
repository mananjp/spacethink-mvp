"""Unit tests for Phases 1 through 6 implementation modules."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from domain import (
    AuditEntry,
    EventOfInterest,
    FaultParameter,
    Hypothesis,
    Severity,
    SimMapping,
    SimResult,
    Telecommand,
    new_id,
)


def test_content_addressed_store_and_ledger(tmp_path):
    from runstore.store import ContentAddressedStore
    from runstore.ledger import AuditLedger

    store = ContentAddressedStore(root=tmp_path / "runs", index_db=tmp_path / "index.db")
    ref = store.put("run-001", "manifest", {"key": "value"}, key="m1")

    assert ref.content_hash is not None
    assert store.verify("run-001", "manifest", "m1")

    item = store.get("run-001", "manifest", "m1")
    assert item["key"] == "value"

    # Ledger tests
    ledger = AuditLedger(db_path=tmp_path / "ledger.db")
    entry1 = ledger.append(actor="agent", kind="manifest", artifact_hash=ref.content_hash)
    entry2 = ledger.append(actor="agent", kind="sim_result", artifact_hash="hash2")

    valid, broken_seq = ledger.verify_chain()
    assert valid
    assert broken_seq == -1

    # Tamper test
    ledger.tamper(seq=1, field="actor", new_value="malicious")
    valid_after_tamper, broken_seq_after = ledger.verify_chain()
    assert not valid_after_tamper
    assert broken_seq_after == 1


def test_basilisk_twin():
    from twin.basilisk_twin import BasiliskTwin

    twin = BasiliskTwin().configure(SimMapping(
        subsystem="reaction_wheel",
        fault_params=(FaultParameter("friction", 1.0), FaultParameter("dropout_rate", 0.02)),
    ))
    df = twin.run(duration_s=500, seed=42)
    assert len(df) == 500
    assert "wheel_speed_rpm" in df.columns
    assert "wheel_current_a" in df.columns
    assert "wheel_temp_c" in df.columns


def test_sbi_scorer_and_calibration():
    from evaluate.sbi_scorer import SBIScorer
    from evaluate.calibration import run_sbc, run_ppc
    from twin.simulator import ToySimulator

    hyp = Hypothesis(
        id=new_id(), event_id="e", text="test", mechanism="bearing_friction_increase",
        fault_params=(FaultParameter("friction", 0.5),), prior=0.5, generator="template"
    )

    twin = ToySimulator().configure(SimMapping("reaction_wheel", hyp.fault_params))
    real = twin.run(duration_s=500, seed=0)
    sims = twin.run_ensemble(n_sims=5, duration_s=500, base_seed=1)

    scorer = SBIScorer()
    result = scorer.score(hyp, real, sims)
    assert result.distance >= 0.0

    # Calibration checks
    sbc_res = run_sbc(scorer, n_prior_samples=10, n_sims=5)
    assert isinstance(sbc_res.passed, bool)

    ppc_res = run_ppc(scorer, n_sims=10)
    assert isinstance(ppc_res.passed, bool)


def test_esa_metrics_scorer():
    from evaluate.esa_metrics import ESAMetricsScorer, compute_event_metrics

    rep = compute_event_metrics([(100, 200)], [(105, 195)])
    assert rep.event_f05 > 0.5
    assert rep.alarming_precision > 0.5

    hyp = Hypothesis(
        id=new_id(), event_id="e", text="test", mechanism="bearing_friction_increase",
        fault_params=(FaultParameter("friction", 0.5),), prior=0.5, generator="template"
    )

    real = pd.DataFrame({
        "wheel_speed_rpm": [4000] * 500,
        "wheel_current_a": [0.5] * 500,
        "wheel_temp_c": [25] * 500,
    })
    scorer = ESAMetricsScorer()
    res = scorer.score(hyp, real, [real.copy()])
    assert res.distance >= 0.0


def test_opssat_ad_ingest():
    from ingest.opssat_ad import generate_synthetic_opssat, opssat_to_pipeline_format

    dataset = generate_synthetic_opssat(n_channels=3, n_points=500)
    assert dataset.n_channels == 3
    df = opssat_to_pipeline_format(dataset)
    assert len(df) == 500
    assert "wheel_speed_rpm" in df.columns


def test_telemanom_lineage_detector():
    from explore.telemanom_lineage import TelemanomLineageDetector
    from ingest.synthetic_generator import generate_reaction_wheel_telemetry

    df = generate_reaction_wheel_telemetry(fault_type="stiction", n_points=1500, fault_start=600)
    detector = TelemanomLineageDetector()
    events = detector.detect(df, ["wheel_speed_rpm", "wheel_current_a"])
    assert isinstance(events, list)


def test_telecommand_explainer():
    from hypothesize.telecommand_explainer import TelecommandExplainer

    event = EventOfInterest(
        id=new_id(), run_id="r", channel="wheel_speed_rpm",
        start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc),
        score=1.8, severity=Severity.LOW, detector_name="test"
    )
    explainer = TelecommandExplainer()
    res = explainer.explain(event)
    assert isinstance(res.auto_explained, bool)


def test_eig_planner():
    from plan.eig_planner import EIGPlanner

    event = EventOfInterest(
        id=new_id(), run_id="r", channel="wheel_speed_rpm",
        start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc),
        score=4.0, severity=Severity.HIGH, detector_name="test"
    )
    hyp = Hypothesis(
        id=new_id(), event_id=event.id, text="test", mechanism="bearing_friction_increase",
        fault_params=(), prior=0.5, generator="template"
    )
    sim = SimResult(hypothesis_id=hyp.id, distance=1.0, posterior=0.8, n_sims=5)

    planner = EIGPlanner()
    proposals = planner.propose_actions(event, [hyp], [sim])
    assert len(proposals) > 0
    assert proposals[0].eig_score >= 0.0


def test_counterfactual_replay():
    from evaluate.counterfactual import run_counterfactual

    real = pd.DataFrame({
        "wheel_speed_rpm": [4000] * 200,
        "wheel_current_a": [0.5] * 200,
        "wheel_temp_c": [25] * 200,
    })
    orig_hyp = Hypothesis(id=new_id(), event_id="e", text="", mechanism="bearing_friction_increase", fault_params=(FaultParameter("friction", 0.5),), prior=0.5, generator="template")
    orig_sim = SimResult(hypothesis_id=orig_hyp.id, distance=0.2, posterior=0.8, n_sims=5)

    alt_hyp = Hypothesis(id=new_id(), event_id="e", text="", mechanism="encoder_dropout", fault_params=(FaultParameter("dropout_rate", 0.05),), prior=0.3, generator="template")

    res = run_counterfactual(real, orig_hyp, orig_sim, [alt_hyp])
    assert res.verdict in ("consistent", "inconsistent", "ambiguous")


def test_claim_pack_exporter(tmp_path):
    from cli.claim_pack import build_evidence_graph, export_claim_pack_json, export_claim_pack_text

    hyp = Hypothesis(id=new_id(), event_id="e", text="", mechanism="bearing_friction_increase", fault_params=(FaultParameter("friction", 0.5),), prior=0.5, generator="template")
    sim = SimResult(hypothesis_id=hyp.id, distance=0.2, posterior=0.8, n_sims=5)

    graph = build_evidence_graph("run-1", [hyp], [sim])
    json_path = export_claim_pack_json("run-1", graph, output_dir=tmp_path)
    text_path = export_claim_pack_text("run-1", graph, output_dir=tmp_path)

    assert json_path.exists()
    assert text_path.exists()


def test_twin_calibrator():
    from twin.calibrator import calibrate_twin

    real = pd.DataFrame({
        "wheel_speed_rpm": [4000] * 300,
        "wheel_current_a": [0.5] * 300,
        "wheel_temp_c": [25] * 300,
    })
    res = calibrate_twin("cust-1", real, n_iterations=5)
    assert res.customer_id == "cust-1"


def test_fleet_clustering():
    from evaluate.clustering import cluster_fleet_events

    ev1 = EventOfInterest(id=new_id(), run_id="r1", channel="wheel_speed_rpm", start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc), score=4.0, severity=Severity.HIGH, detector_name="d")
    ev2 = EventOfInterest(id=new_id(), run_id="r2", channel="wheel_speed_rpm", start_ts=datetime.now(timezone.utc), end_ts=datetime.now(timezone.utc), score=4.1, severity=Severity.HIGH, detector_name="d")

    clusters = cluster_fleet_events([ev1, ev2])
    assert isinstance(clusters, list)


def test_knowledge_rag(tmp_path):
    from knowledge.rag import KnowledgeBase, Document

    kb = KnowledgeBase(persist_dir=tmp_path / "chroma", collection_name="test_kb")
    doc = Document(id="doc1", text="Reaction wheel bearing friction increase causes thermal rise.", source="manual")
    kb.ingest(doc)

    res = kb.retrieve("bearing friction thermal", n_results=1)
    assert len(res.documents) == 1
    assert "Reaction wheel" in res.documents[0].text


def test_model_export(tmp_path):
    from explore.export import export_forecaster_numpy, export_forecaster_onnx

    rep_np = export_forecaster_numpy(output_dir=tmp_path)
    assert Path(rep_np.output_path).exists()

    rep_onnx = export_forecaster_onnx(output_dir=tmp_path)
    assert Path(rep_onnx.output_path).exists()


def test_onboard_evaluator():
    from explore.onboard import OnboardEvaluator

    evaluator = OnboardEvaluator()
    decisions = evaluator.evaluate_window("wheel_speed_rpm", np.array([4000.0] * 50))
    assert len(decisions) == 50
    assert evaluator.memory_usage_bytes() < 59 * 1024
