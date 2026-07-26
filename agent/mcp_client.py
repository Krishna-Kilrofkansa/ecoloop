"""MCP-aware agent that invokes Eco-Loop tools then Gemma 4 E2B."""

from __future__ import annotations

from agent.controller import AgentConfig, EcoLoopAgent
from simulation.shared_state import STATE


def run_supervisory_cycle(agent: EcoLoopAgent) -> None:
    """Single LLM decision cycle with MCP tool context."""
    summary = STATE.get_summary()
    if summary.get("status") == "no_data":
        return

    # Mirror MCP tool flow: diagnostics + summary already in STATE
    cmd = agent.decide(STATE.latest)
    if cmd:
        STATE.append_diagnostic(
            f"MCP cycle: cool={cmd.cooling_setpoint_c} heat={cmd.heating_setpoint_c}"
        )
    return cmd
