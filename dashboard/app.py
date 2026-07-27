"""Streamlit dashboard — closed-loop viewer with LLM Council Deliberation & Human Validation Gate.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from hypothesize.groq_explainer import generate_exhyte_timeline

load_dotenv()

st.set_page_config(page_title="spaceThink — EXHYTE Dashboard", layout="wide", page_icon="🛸")
st.title("🛸 spaceThink — Autonomous EXHYTE Closed-Loop Telemetry Agent")
st.caption("Explore (Detect) ➔ Hypothesize ➔ Test (Digital Twin) ➔ Score (SBI) ➔ LLM Council ➔ Human Gate")

# Initialize session state for human sign-offs if not present
if "human_decisions" not in st.session_state:
    st.session_state.human_decisions = {}

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration & Groq AI Setup")
user_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
    help="Enter your Groq API Key (starts with gsk_...) to generate live LLM timelines & scientific reasoning.",
)
selected_model = st.sidebar.selectbox(
    "Groq LLM Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    index=0,
)

if user_api_key:
    st.sidebar.success("⚡ Groq API Key Active")
else:
    st.sidebar.info("ℹ️ Running in Offline Mode (Offline Council & Timeline Templates)")

# --- Load Data Reports ---
reports_path = Path("data/reports.json")
if not reports_path.exists():
    st.warning("No reports found. Run: `python -m cli.main generate-data && python -m cli.main run-all`")
    st.stop()

reports = json.loads(reports_path.read_text())
file_names = [r["source_file"] for r in reports]
selected_file = st.selectbox("Select Telemetry Run", file_names)
report = next(r for r in reports if r["source_file"] == selected_file)

st.markdown(f"### 📊 Run: `{report['run_id'][:8]}` — Injected Fault: `{selected_file}` — `{report['n_events']}` Event(s) Detected")

# --- Plot Telemetry Channels ---
df = pd.read_csv(Path("data/synthetic") / selected_file)
fig = go.Figure()
for ch in ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]:
    fig.add_trace(go.Scatter(x=df["t"], y=df[ch], name=ch, mode="lines"))
fig.update_layout(
    title="Multichannel Telemetry Series (Wheel Speed, Current, Temperature)",
    xaxis_title="Time (s)",
    yaxis_title="Telemetry Value",
    height=360,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("## ⚡ Detected Events, LLM Council & Human Validation Gate")

# Filter or inspect events
for i, ev in enumerate(report["events"]):
    event_id = ev["event_id"]
    val_status = st.session_state.human_decisions.get(event_id, ev.get("validation_status", "escalated_pending_human"))

    severity_color = "🔴" if ev['severity'] == "high" else "🟡" if ev['severity'] == "medium" else "🟢"
    status_badge = (
        "🟢 AUTO APPROVED" if val_status == "auto_approved"
        else "🟢 HUMAN APPROVED" if val_status == "human_approved"
        else "🔴 HUMAN REJECTED" if val_status == "human_rejected"
        else "⚡ HUMAN OVERRIDDEN" if val_status == "human_overridden"
        else "⚠️ ESCALATED FOR HUMAN VALIDATION"
    )

    title = f"{severity_color} Event #{i+1}: `{ev['channel']}` — Severity: {ev['severity'].upper()} (Z-Score: {ev['score']:.2f}) | Status: [{status_badge}]"
    
    with st.expander(title, expanded=(i == 0)):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🎯 Diagnostic Summary & Posterior Belief")
            st.write(f"**Top Diagnosed Mechanism:** `{ev['top_hypothesis']}`")
            st.write(f"**Posterior Belief Score:** `{ev['posterior']:.3f}`" if ev['posterior'] else "N/A")
            st.info(ev['top_hypothesis_text'])

            st.markdown("**Ranked Candidate Mechanisms:**")
            st.table(pd.DataFrame(ev["ranked_mechanisms"], columns=["Mechanism", "Posterior Probability"]))

            # --- LLM Council Section ---
            council_data = ev.get("council_consensus")
            if council_data:
                st.markdown("#### 🏛️ LLM Council Deliberation Panel")
                verdict = council_data.get("verdict", "split_council")
                score = council_data.get("consensus_score", 0.0)

                v_color = "🟢" if "unanimous" in verdict or "strong" in verdict else "🟡" if "split" in verdict else "🔴"
                st.markdown(f"**Council Verdict:** {v_color} `{verdict.upper()}` (Consensus Index: `{score:.2f}`)")
                st.caption(f"Summary: {council_data.get('summary', '')}")

                votes = council_data.get("individual_votes", [])
                if votes:
                    vote_df = pd.DataFrame(votes)
                    vote_df.columns = ["Role", "Agrees", "Confidence", "Rationale"]
                    st.table(vote_df)

        with col2:
            # --- Human Validation Gate Section ---
            st.markdown("#### 🛡️ Human-in-the-Loop Validation Gate")
            st.write(f"**Current Gate Status:** `{val_status.upper()}`")

            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                if st.button(f"✅ Approve Discovery", key=f"app_{event_id}"):
                    st.session_state.human_decisions[event_id] = "human_approved"
                    st.rerun()
            with b_col2:
                if st.button(f"❌ Reject / False Alarm", key=f"rej_{event_id}"):
                    st.session_state.human_decisions[event_id] = "human_rejected"
                    st.rerun()
            with b_col3:
                if st.button(f"⚡ Override Diagnosis", key=f"ovr_{event_id}"):
                    st.session_state.human_decisions[event_id] = "human_overridden"
                    st.rerun()

            st.markdown("---")
            st.markdown("#### 🤖 AI EXHYTE Closed-Loop Timeline (Groq)")

            ai_text = ev.get("ai_timeline")
            if user_api_key:
                if st.button(f"🔄 Re-Generate Timeline with Groq for Event #{i+1}", key=f"btn_{ev['event_id']}"):
                    with st.spinner("Calling Groq API (llama-3.3-70b-versatile)..."):
                        ai_text = generate_exhyte_timeline(
                            event_id=ev['event_id'],
                            channel=ev['channel'],
                            severity=ev['severity'],
                            score=ev['score'],
                            top_hypothesis=ev['top_hypothesis'],
                            top_text=ev['top_hypothesis_text'],
                            posterior=ev['posterior'],
                            ranked_mechanisms=ev['ranked_mechanisms'],
                            api_key=user_api_key,
                            model_name=selected_model,
                        )

            if not ai_text:
                ai_text = generate_exhyte_timeline(
                    event_id=ev['event_id'],
                    channel=ev['channel'],
                    severity=ev['severity'],
                    score=ev['score'],
                    top_hypothesis=ev['top_hypothesis'],
                    top_text=ev['top_hypothesis_text'],
                    posterior=ev['posterior'],
                    ranked_mechanisms=ev['ranked_mechanisms'],
                    api_key=user_api_key,
                    model_name=selected_model,
                )

            st.markdown(ai_text)
