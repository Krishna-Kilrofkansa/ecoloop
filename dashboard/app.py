"""Streamlit dashboard — baseline vs Eco-Loop agent savings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config_loader import load_settings
from simulation.shared_state import STATE

ROOT = Path(__file__).resolve().parent.parent


def load_data() -> pd.DataFrame:
    cfg = load_settings()
    path = ROOT / cfg["dashboard"]["data_file"]
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def render_metrics(df: pd.DataFrame) -> None:
    baseline_kwh = STATE.total_kwh_baseline
    agent_kwh = STATE.total_kwh_agent
    if df.empty and baseline_kwh == 0:
        st.warning("No telemetry yet. Run: `python main.py --mode full --demo`")
        return

    if baseline_kwh == 0 and not df.empty:
        baseline_kwh = df[df["mode"] == "baseline"]["elec_power_kw"].sum() * 0.25
        agent_kwh = df[df["mode"] == "agent"]["elec_power_kw"].sum() * 0.25

    reduction = ((baseline_kwh - agent_kwh) / baseline_kwh * 100) if baseline_kwh > 0 else 0
    cost_baseline = baseline_kwh * 0.15
    cost_agent = agent_kwh * 0.12

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline kWh", f"{baseline_kwh:,.0f}")
    c2.metric("Agent kWh (Gemma 4 E2B)", f"{agent_kwh:,.0f}", f"-{reduction:.1f}%")
    c3.metric("Discomfort hrs (baseline)", STATE.discomfort_hours_baseline)
    c4.metric("Discomfort hrs (agent)", STATE.discomfort_hours_agent,
              f"-{max(0, STATE.discomfort_hours_baseline - STATE.discomfort_hours_agent)} hrs")


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    agent_df = df[df["mode"] == "agent"].copy()
    if agent_df.empty:
        agent_df = df.copy()

    agent_df["time_idx"] = range(len(agent_df))

    tab1, tab2, tab3 = st.tabs(["Energy Demand", "Thermal Comfort", "Setpoints & Grid"])

    with tab1:
        fig = px.line(agent_df, x="time_idx", y="elec_power_kw", title="Facility Demand (kW)")
        fig.add_hline(y=agent_df["elec_power_kw"].max(), line_dash="dash", annotation_text="Peak")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=agent_df["time_idx"], y=agent_df["pmv"], name="PMV"))
        fig2.add_hrect(y0=-0.5, y1=0.5, fillcolor="green", opacity=0.1, line_width=0)
        fig2.update_layout(title="Predicted Mean Vote (PMV)", yaxis_title="PMV")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.line(
            agent_df,
            x="time_idx",
            y=["cooling_setpoint_c", "heating_setpoint_c", "zone_temp_c"],
            title="Zone Temp vs Setpoints",
        )
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.line(agent_df, x="time_idx", y="grid_tariff_usd_kwh", title="Grid Tariff ($/kWh)")
        st.plotly_chart(fig4, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Eco-Loop Dashboard", layout="wide")
    st.title("Eco-Loop Building Agents")
    st.caption("Closed-loop control: EnergyPlus ↔ MCP ↔ Gemma 4 E2B")

    df = load_data()
    render_metrics(df)
    render_charts(df)

    with st.expander("Latest Agent Response (Gemma 4 E2B)"):
        st.code(STATE.last_agent_response or "No response yet")

    with st.expander("Diagnostics"):
        for d in STATE.diagnostics[-15:]:
            st.text(d)


if __name__ == "__main__":
    main()
