"""Streamlit dashboard — read-only viewer over run reports.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="spaceThink — MVP dashboard", layout="wide")
st.title("spaceThink — closed-loop diagnosis (synthetic data demo)")

reports_path = Path("data/reports.json")
if not reports_path.exists():
    st.warning("No reports found. Run: python -m cli.main generate-data && python -m cli.main run-all")
    st.stop()

reports = json.loads(reports_path.read_text())
file_names = [r["source_file"] for r in reports]
selected = st.selectbox("Select run", file_names)
report = next(r for r in reports if r["source_file"] == selected)

st.subheader(f"Run: {report['run_id'][:8]} — {report['n_events']} event(s) detected")

df = pd.read_csv(Path("data/synthetic") / selected)
fig = go.Figure()
for ch in ["wheel_speed_rpm", "wheel_current_a", "wheel_temp_c"]:
    fig.add_trace(go.Scatter(x=df["t"], y=df[ch], name=ch, mode="lines"))
fig.update_layout(title="Telemetry channels", xaxis_title="t (s)", height=400)
st.plotly_chart(fig, use_container_width=True)

for ev in report["events"]:
    with st.expander(f"Event on {ev['channel']} — severity {ev['severity']} (score {ev['score']:.2f})"):
        st.write(f"**Top hypothesis:** {ev['top_hypothesis']} (posterior {ev['posterior']:.2f})")
        st.write(ev["top_hypothesis_text"])
        st.write("Ranked mechanisms (mechanism, posterior):")
        st.table(pd.DataFrame(ev["ranked_mechanisms"], columns=["mechanism", "posterior"]))
