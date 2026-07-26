"""FastMCP server exposing Eco-Loop building control tools (Gemma 4 E2B optimized)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field, ValidationError

from agent.safety import SafetyInterlock
from agent.schemas import CompactTelemetry, HVACSetpointArgs, compact_telemetry_from_latest
from simulation.log_parser import parse_eplus_errors
from simulation.shared_state import STATE, ControlCommand

mcp = FastMCP(
    name="ecoloop-building-agent",
    instructions=(
        "Eco-Loop MCP server for EnergyPlus. "
        "Read compact telemetry, apply bounded HVAC setpoints via apply_hvac_setpoints."
    ),
)

_safety = SafetyInterlock()


@mcp.tool()
def get_compact_telemetry() -> dict:
    """Return high-density JSON telemetry optimized for Gemma 4 E2B."""
    if STATE.latest is None:
        return {"status": "no_data"}
    from dataclasses import asdict
    compact = compact_telemetry_from_latest(asdict(STATE.latest))
    return compact.model_dump()


@mcp.tool()
def get_telemetry_summary() -> dict:
    """Return aggregated rolling window stats."""
    return STATE.get_summary()


@mcp.tool()
def parse_eplus_error_log(
    err_file: Annotated[str, Field(description="Path to EnergyPlus .err file")] = "output/simulation/eplusout.err",
) -> dict:
    """Parse EnergyPlus error log into fatal/severe/info categories."""
    return parse_eplus_errors(Path(err_file))


@mcp.tool()
def get_simulation_diagnostics() -> list[str]:
    """Return recent agent and safety diagnostics for self-correction."""
    return list(STATE.diagnostics[-10:])


@mcp.tool()
def apply_hvac_setpoints(
    heating_setpoint_c: Annotated[float, Field(ge=18.0, le=22.0, description="Heating setpoint °C")],
    cooling_setpoint_c: Annotated[float, Field(ge=21.0, le=28.0, description="Cooling setpoint °C")],
    rationale: Annotated[str, Field(max_length=1000, description="Brief reason")] = "",
) -> dict:
    """Single strict tool: inject validated heating/cooling setpoints into simulation."""
    try:
        args = HVACSetpointArgs(
            heating_setpoint_c=heating_setpoint_c,
            cooling_setpoint_c=cooling_setpoint_c,
            rationale=rationale,
        )
    except ValidationError as exc:
        current = STATE.latest
        return {
            "success": False,
            "error": str(exc),
            "held_setpoints": {
                "cooling_setpoint_c": current.cooling_setpoint_c if current else 23.0,
                "heating_setpoint_c": current.heating_setpoint_c if current else 21.0,
            },
        }

    command = ControlCommand(
        cooling_setpoint_c=args.cooling_setpoint_c,
        heating_setpoint_c=args.heating_setpoint_c,
        rationale=args.rationale,
        source="mcp_tool",
    )
    sim_minute = 0.0
    if STATE.latest:
        sim_minute = STATE.latest.sim_day * 1440 + STATE.latest.sim_hour * 60

    validated, err = _safety.validate(command, STATE.latest, sim_minute)
    if err or validated is None:
        current = STATE.latest
        STATE.append_diagnostic(f"MCP reject: {err}")
        return {
            "success": False,
            "error": err or "validation failed",
            "held_setpoints": {
                "cooling_setpoint_c": current.cooling_setpoint_c if current else 23.0,
                "heating_setpoint_c": current.heating_setpoint_c if current else 21.0,
            },
        }

    STATE.last_command = validated
    return {
        "success": True,
        "cooling_setpoint_c": validated.cooling_setpoint_c,
        "heating_setpoint_c": validated.heating_setpoint_c,
        "rationale": validated.rationale,
    }


@mcp.tool()
def export_telemetry_csv(
    output_path: Annotated[str, Field(description="CSV output path")] = "output/telemetry_log.csv",
) -> dict:
    """Export telemetry history to CSV."""
    path = Path(output_path)
    STATE.export_csv(path)
    return {"success": True, "path": str(path), "rows": len(STATE.history)}


@mcp.tool()
def get_energy_savings_metrics() -> dict:
    """Compare baseline vs agent cumulative kWh."""
    baseline = STATE.total_kwh_baseline or 1.0
    agent = STATE.total_kwh_agent or 0.0
    reduction_pct = ((baseline - agent) / baseline * 100) if baseline > 0 and agent > 0 else 0.0
    return {
        "total_kwh_baseline": round(STATE.total_kwh_baseline, 2),
        "total_kwh_agent": round(STATE.total_kwh_agent, 2),
        "kwh_reduction_pct": round(reduction_pct, 2),
        "discomfort_hours_baseline": STATE.discomfort_hours_baseline,
        "discomfort_hours_agent": STATE.discomfort_hours_agent,
    }


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    run_server()
