"""Eco-Loop closed-loop orchestrator."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import sys
import json
from dataclasses import asdict
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from agent.controller import EcoLoopAgent
from agent.safety import SafetyInterlock
from config_loader import agent_config_from_settings, load_settings, safety_config_from_settings
from simulation.demo_engine import DemoBuildingSimulator
from simulation.shared_state import STATE, ControlCommand, TelemetrySnapshot


def run_baseline_pass(cfg: dict) -> None:
    """Run baseline RBC simulation to populate comparison metrics."""
    sim = cfg["simulation"]
    engine = DemoBuildingSimulator(
        mode="baseline",
        timestep_minutes=sim["timestep_minutes"],
        run_days=sim["run_period_days"],
    )
    print("[Eco-Loop] Running baseline RBC simulation...")
    engine.run(realtime=False)
    print(f"[Eco-Loop] Baseline total kWh: {STATE.total_kwh_baseline:.1f}")


def save_checkpoint(checkpoint_file: Path, step: int, engine: Any, safety: Any) -> None:
    try:
        with STATE.lock:
            data = {
                "step": step,
                "engine": {
                    "cooling_sp": getattr(engine, "cooling_sp", 23.0),
                    "heating_sp": getattr(engine, "heating_sp", 21.0),
                    "zone_temp": getattr(engine, "zone_temp", 23.5),
                    "rh": getattr(engine, "rh", 52.0),
                    "co2": getattr(engine, "co2", 780.0),
                    "last_supervisory_step": getattr(engine, "_last_supervisory_step", -999),
                } if isinstance(engine, DemoBuildingSimulator) else None,
                "safety": {
                    "last_command": asdict(safety._last_command) if safety._last_command else None,
                    "last_change_minute": safety._last_change_minute,
                },
                "state": {
                    "total_kwh_baseline": STATE.total_kwh_baseline,
                    "total_kwh_agent": STATE.total_kwh_agent,
                    "discomfort_hours_baseline": STATE.discomfort_hours_baseline,
                    "discomfort_hours_agent": STATE.discomfort_hours_agent,
                    "last_agent_response": STATE.last_agent_response,
                    "diagnostics": STATE.diagnostics,
                    "history": [asdict(h) for h in STATE.history],
                }
            }
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Eco-Loop] Warning: failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_file: Path, engine: Any, safety: Any) -> bool:
    if not checkpoint_file.exists():
        return False
    try:
        with checkpoint_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Restore engine attributes if demo simulator
        if data["engine"] and isinstance(engine, DemoBuildingSimulator):
            eng_data = data["engine"]
            engine.cooling_sp = eng_data["cooling_sp"]
            engine.heating_sp = eng_data["heating_sp"]
            engine.zone_temp = eng_data["zone_temp"]
            engine.rh = eng_data["rh"]
            engine.co2 = eng_data["co2"]
            engine._last_supervisory_step = eng_data["last_supervisory_step"]
        
        # Set engine step
        engine.step = data["step"]
        
        # Restore safety attributes
        saf_data = data["safety"]
        if saf_data["last_command"]:
            cmd = saf_data["last_command"]
            safety._last_command = ControlCommand(
                cooling_setpoint_c=cmd["cooling_setpoint_c"],
                heating_setpoint_c=cmd["heating_setpoint_c"],
                rationale=cmd["rationale"],
                source=cmd["source"]
            )
        safety._last_change_minute = saf_data["last_change_minute"]
        
        # Restore SharedState fields
        st_data = data["state"]
        with STATE.lock:
            STATE.total_kwh_baseline = st_data["total_kwh_baseline"]
            STATE.total_kwh_agent = st_data["total_kwh_agent"]
            STATE.discomfort_hours_baseline = st_data["discomfort_hours_baseline"]
            STATE.discomfort_hours_agent = st_data["discomfort_hours_agent"]
            STATE.last_agent_response = st_data["last_agent_response"]
            STATE.diagnostics = st_data["diagnostics"]
            
            STATE.history.clear()
            STATE.rolling_window.clear()
            for h_dict in st_data["history"]:
                snap = TelemetrySnapshot(**h_dict)
                STATE.history.append(snap)
                STATE.rolling_window.append(snap)
            if STATE.history:
                STATE.latest = STATE.history[-1]
                
        print(f"[Eco-Loop] Checkpoint loaded successfully. Resuming from step {engine.step} (Baseline: {STATE.total_kwh_baseline:.1f} kWh, Agent: {STATE.total_kwh_agent:.1f} kWh)")
        return True
    except Exception as e:
        print(f"[Eco-Loop] Warning: failed to load checkpoint: {e}")
        return False


def run_closed_loop(cfg: dict, demo: bool = True, checkpoint_file: Path | None = None) -> None:
    sim_cfg = cfg["simulation"]
    safety = SafetyInterlock(safety_config_from_settings(cfg))
    agent = EcoLoopAgent(agent_config_from_settings(cfg), safety)

    if agent.check_ollama_available():
        print("[Eco-Loop] Gemma 4 E2B detected via Ollama")
    else:
        print("[Eco-Loop] Gemma 4 E2B not found — using rule-based fallback")

    if demo or sim_cfg.get("demo_mode", True):
        engine = DemoBuildingSimulator(
            mode="agent",
            timestep_minutes=sim_cfg["timestep_minutes"],
            run_days=sim_cfg["run_period_days"],
            supervisory_interval_minutes=sim_cfg["supervisory_interval_minutes"],
        )
    else:
        from simulation.eplus_bridge import EnergyPlusBridge
        root = Path(__file__).resolve().parent
        engine = EnergyPlusBridge(
            idf_path=root / sim_cfg["idf_baseline"],
            epw_path=root / sim_cfg["weather"],
            output_dir=root / sim_cfg["output_dir"],
            mode="agent",
        )

    # Reconstruct state from checkpoint or start fresh
    loaded_checkpoint = False
    if checkpoint_file and checkpoint_file.exists():
        loaded_checkpoint = load_checkpoint(checkpoint_file, engine, safety)
    
    if not loaded_checkpoint:
        with STATE.lock:
            STATE.history.clear()
            STATE.rolling_window.clear()

    out = Path(cfg["dashboard"]["data_file"])

    def on_step(snap, step):
        if not engine.needs_supervisory():
            return
        cmd = agent.decide(snap)
        if cmd:
            engine.apply_command(cmd.cooling_setpoint_c, cmd.heating_setpoint_c)
            msg = f"Step {step}: cool={cmd.cooling_setpoint_c}°C heat={cmd.heating_setpoint_c}°C — {cmd.rationale} (Source: {cmd.source})"
            STATE.append_diagnostic(msg)
            print(f"[Eco-Loop] {msg}")
        else:
            # Self-correction hold: keep active setpoints, never crash the loop
            engine.apply_command(snap.cooling_setpoint_c, snap.heating_setpoint_c)
            print(f"[Eco-Loop] Step {step}: Held setpoints (Self-correction fallback)")
        
        # Export telemetry log in real-time for live dashboard updates
        STATE.export_csv(out)
        
        # Save checkpoint in real-time
        if checkpoint_file:
            save_checkpoint(checkpoint_file, step, engine, safety)

    print("[Eco-Loop] Starting agent closed-loop simulation...")
    if hasattr(engine, "run"):
        if isinstance(engine, DemoBuildingSimulator):
            engine.run(on_step=on_step, realtime=False)
        else:
            engine.set_supervisory_callback(lambda s: on_step(s, 0))
            engine.run()

    # Clean up checkpoint on successful completion
    if checkpoint_file and checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            print("[Eco-Loop] Simulation complete. Checkpoint cleaned up.")
        except Exception:
            pass

    out = Path(cfg["dashboard"]["data_file"])
    STATE.export_csv(out)
    print(f"[Eco-Loop] Telemetry exported to {out}")

    baseline = STATE.total_kwh_baseline or 1.0
    agent_kwh = STATE.total_kwh_agent
    reduction = (baseline - agent_kwh) / baseline * 100 if agent_kwh > 0 else 0
    print(f"[Eco-Loop] Agent kWh: {agent_kwh:.1f} | Reduction vs baseline: {reduction:.1f}%")


def run_mcp_server(cfg: dict) -> None:
    from mcp_server.server import run_server
    mcp_cfg = cfg["mcp"]
    run_server(host=mcp_cfg["host"], port=mcp_cfg["port"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Eco-Loop Building Agents")
    parser.add_argument("--mode", choices=["loop", "baseline", "mcp", "full"], default="full")
    parser.add_argument("--demo", action="store_true", help="Force demo mode (no EnergyPlus)")
    args = parser.parse_args()

    cfg = load_settings()
    demo = args.demo or cfg["simulation"].get("demo_mode", True)

    if args.mode == "mcp":
        run_mcp_server(cfg)
        return

    if args.mode in ("baseline", "full"):
        run_baseline_pass(cfg)

    if args.mode in ("loop", "full"):
        run_closed_loop(cfg, demo=demo)


if __name__ == "__main__":
    main()
