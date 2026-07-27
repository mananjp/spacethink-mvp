"""Streamlit dashboard — closed-loop viewer with Groq AI Timeline & Discovery Analysis.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from hypothesize.groq_explainer import generate_exhyte_timeline

load_dotenv()

st.set_page_config(page_title="spaceThink — EXHYTE Dashboard", layout="wide", page_icon="🛸")
st.title("🛸 spaceThink — Autonomous EXHYTE Closed-Loop Telemetry Agent")
st.caption("Explore (Detect) ➔ Hypothesize ➔ Test (Digital Twin) ➔ Refine (Posterior Scoring)")

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
    st.sidebar.info("ℹ️ Running in Offline Mode (Offline Timeline Template)")

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
    height=380,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("## ⚡ Detected Events & Closed-Loop EXHYTE Analysis")

# Filter or inspect events
for i, ev in enumerate(report["events"]):
    severity_color = "🔴" if ev['severity'] == "high" else "🟡" if ev['severity'] == "medium" else "🟢"
    title = f"{severity_color} Event #{i+1}: `{ev['channel']}` — Severity: {ev['severity'].upper()} (Z-Score: {ev['score']:.2f})"
    
    with st.expander(title, expanded=(i == 0)):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🎯 Diagnostic Summary")
            st.write(f"**Top Diagnosed Mechanism:** `{ev['top_hypothesis']}`")
            st.write(f"**Posterior Belief Score:** `{ev['posterior']:.3f}`" if ev['posterior'] else "N/A")
            st.info(ev['top_hypothesis_text'])

            st.markdown("**Ranked Candidate Mechanisms:**")
            st.table(pd.DataFrame(ev["ranked_mechanisms"], columns=["Mechanism", "Posterior Probability"]))

        with col2:
            st.markdown("#### 🤖 AI EXHYTE Closed-Loop Timeline (Powered by Groq)")

            # Check if live AI generation requested or present
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
