"""Shared runtime state between simulation, MCP server, and agent."""

from __future__ import annotations

import csv
import json
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TelemetrySnapshot:
    """Single timestep telemetry from EnergyPlus or demo engine."""

    timestamp: str
    sim_day: int
    sim_hour: float
    zone_temp_c: float
    outdoor_temp_c: float
    zone_rh_percent: float
    zone_co2_ppm: float
    pmv: float
    elec_power_kw: float
    cooling_setpoint_c: float
    heating_setpoint_c: float
    grid_tariff_usd_kwh: float
    grid_carbon_gco2_kwh: float
    mode: str = "baseline"  # baseline | agent
    agent_action: str = ""


@dataclass
class ControlCommand:
    """Validated setpoint override from agent."""

    cooling_setpoint_c: float
    heating_setpoint_c: float
    rationale: str = ""
    source: str = "gemma4:e2b"


@dataclass
class SharedState:
    """Thread-safe bridge for closed-loop control."""

    latest: TelemetrySnapshot | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=500))
    rolling_window: deque = field(default_factory=lambda: deque(maxlen=12))
    last_command: ControlCommand | None = None
    last_agent_response: str = ""
    diagnostics: list[str] = field(default_factory=list)
    simulation_running: bool = False
    agent_enabled: bool = True
    total_kwh_baseline: float = 0.0
    total_kwh_agent: float = 0.0
    discomfort_hours_baseline: int = 0
    discomfort_hours_agent: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def push_telemetry(self, snap: TelemetrySnapshot) -> None:
        with self.lock:
            self.latest = snap
            self.history.append(snap)
            self.rolling_window.append(snap)
            if abs(snap.pmv) > 0.5 and 7 <= snap.sim_hour <= 22:
                if snap.mode == "baseline":
                    self.discomfort_hours_baseline += 1
                else:
                    self.discomfort_hours_agent += 1
            dt_hours = 0.25  # 15-min timestep
            kwh = snap.elec_power_kw * dt_hours
            if snap.mode == "baseline":
                self.total_kwh_baseline += kwh
            else:
                self.total_kwh_agent += kwh

    def get_summary(self) -> dict[str, Any]:
        with self.lock:
            items = list(self.rolling_window)
            if not items:
                return {"status": "no_data"}
            temps = [i.zone_temp_c for i in items]
            pmvs = [i.pmv for i in items]
            powers = [i.elec_power_kw for i in items]
            return {
                "count": len(items),
                "zone_temp_mean_c": sum(temps) / len(temps),
                "zone_temp_max_c": max(temps),
                "pmv_mean": sum(pmvs) / len(pmvs),
                "pmv_max_abs": max(abs(p) for p in pmvs),
                "power_mean_kw": sum(powers) / len(powers),
                "power_peak_kw": max(powers),
                "latest": asdict(self.latest) if self.latest else None,
            }

    def append_diagnostic(self, message: str) -> None:
        with self.lock:
            self.diagnostics.append(f"[{datetime.now().isoformat()}] {message}")
            if len(self.diagnostics) > 200:
                self.diagnostics = self.diagnostics[-200:]

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            rows = [asdict(h) for h in self.history]
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def to_json(self) -> str:
        return json.dumps(self.get_summary(), indent=2)


# Global singleton used across modules
STATE = SharedState()
