"""Compact prompt templates optimized for Gemma 4 E2B."""

from __future__ import annotations

import json

from agent.schemas import CompactTelemetry, compact_telemetry_from_latest

SYSTEM_PROMPT = """You are Eco-Loop HVAC supervisor. Output ONLY JSON:
{"heating_setpoint_c": float, "cooling_setpoint_c": float, "rationale": "short"}

Rules:
- heating: 18-22°C, cooling: 23-28°C, heating must be ≥1°C below cooling
- Keep |pmv_index| ≤ 0.5; if pmv_index > 0.45 tighten cooling
- grid_tier HIGH_PEAK (14-19h): allow cooling up to 25°C if pmv_index ≤ 0.45
- grid_tier OFF_PEAK (4-8h): pre-cool to ~21.5°C when afternoon peak expected
- co2_ppm > 900: do not aggressive setback"""

USER_PROMPT_TEMPLATE = """telemetry={telemetry}
last_error={last_error}
Return setpoints JSON only."""


def format_agent_prompt(summary: dict, last_error: str = "", mode: str = "agent") -> str:
    latest = summary.get("latest")
    if not latest:
        telemetry_json = "{}"
    else:
        compact = compact_telemetry_from_latest(latest)
        telemetry_json = compact.model_dump_json()

    return USER_PROMPT_TEMPLATE.format(
        telemetry=telemetry_json,
        last_error=last_error or "none",
    )
