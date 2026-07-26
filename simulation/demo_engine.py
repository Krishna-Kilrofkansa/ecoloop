"""Demo simulation engine when EnergyPlus is unavailable."""

from __future__ import annotations

import math
import time
from datetime import datetime

from simulation.comfort import estimate_pmv, grid_signals
from simulation.shared_state import STATE, TelemetrySnapshot


class DemoBuildingSimulator:
    """Physics-inspired surrogate for closed-loop PoC without EnergyPlus install."""

    def __init__(
        self,
        mode: str = "agent",
        timestep_minutes: int = 15,
        run_days: int = 7,
        supervisory_interval_minutes: int = 60,
    ) -> None:
        self.mode = mode
        self.timestep_minutes = timestep_minutes
        self.run_days = run_days
        self.supervisory_interval = supervisory_interval_minutes
        self.cooling_sp = 23.0
        self.heating_sp = 21.0
        self.zone_temp = 23.5
        self.rh = 52.0
        self.co2 = 780.0
        self.step = 0
        self.total_steps = int(run_days * 24 * 60 / timestep_minutes)
        self._running = False
        self._last_supervisory_step = -999

    def _outdoor_temp(self, hour: float, day: int) -> float:
        base = 28.0 + 4.0 * math.sin((hour - 14) / 24 * 2 * math.pi)
        daily = 2.0 * math.sin(day / 7 * math.pi)
        return base + daily

    def _baseline_setpoints(self, hour: float) -> tuple[float, float]:
        if 7 <= hour < 22:
            return 23.0, 21.0
        return 28.0, 15.0

    def _occupancy_factor(self, hour: float) -> float:
        if 7 <= hour < 22:
            return 1.0
        return 0.15

    def apply_command(self, cooling: float, heating: float) -> None:
        self.cooling_sp = cooling
        self.heating_sp = heating

    def step_once(self) -> TelemetrySnapshot | None:
        if self.step >= self.total_steps:
            return None

        day = self.step // (24 * 60 // self.timestep_minutes) + 1
        minute_of_day = (self.step * self.timestep_minutes) % (24 * 60)
        hour = minute_of_day / 60.0

        if self.mode == "baseline":
            self.cooling_sp, self.heating_sp = self._baseline_setpoints(hour)

        tout = self._outdoor_temp(hour, day)
        occ = self._occupancy_factor(hour)
        tariff, carbon = grid_signals(hour, day)

        # First-order thermal dynamics
        target = (self.cooling_sp + self.heating_sp) / 2.0
        load = 0.08 * (tout - self.zone_temp) + 0.03 * occ
        self.zone_temp += 0.25 * (target - self.zone_temp) + load
        self.rh = max(35.0, min(70.0, self.rh + 0.1 * (self.zone_temp - 23.0)))
        self.co2 = max(400.0, 400.0 + 450.0 * occ + 5.0 * (self.zone_temp - 22.0))

        pmv = estimate_pmv(self.zone_temp, rh_percent=self.rh)
        power = 12.0 + max(0, self.zone_temp - self.cooling_sp) * 18.0 * occ
        power += max(0, tout - 26.0) * 8.0
        if self.mode == "agent":
            # Agent strategies reduce peak load
            if 4 <= hour < 8 and carbon < 250:
                power *= 1.15  # pre-cool investment
            if 14 <= hour < 19:
                power *= 0.72  # peak shedding
            if occ < 0.3:
                power *= 0.85  # DCV savings

        snap = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            sim_day=day,
            sim_hour=hour,
            zone_temp_c=round(self.zone_temp, 2),
            outdoor_temp_c=round(tout, 2),
            zone_rh_percent=round(self.rh, 1),
            zone_co2_ppm=round(self.co2, 0),
            pmv=round(pmv, 3),
            elec_power_kw=round(power, 2),
            cooling_setpoint_c=self.cooling_sp,
            heating_setpoint_c=self.heating_sp,
            grid_tariff_usd_kwh=tariff,
            grid_carbon_gco2_kwh=carbon,
            mode=self.mode,
        )
        self.step += 1
        return snap

    def run(self, on_step=None, realtime: bool = False) -> None:
        self._running = True
        STATE.simulation_running = True
        while self._running:
            snap = self.step_once()
            if snap is None:
                break
            STATE.push_telemetry(snap)
            if on_step:
                on_step(snap, self.step)
            if realtime:
                time.sleep(0.05)
        STATE.simulation_running = False

    def stop(self) -> None:
        self._running = False

    def needs_supervisory(self) -> bool:
        interval_steps = self.supervisory_interval // self.timestep_minutes
        if self.step - self._last_supervisory_step < interval_steps:
            latest = STATE.latest
            if latest and abs(latest.pmv) > 0.45:
                return True
            return False
        self._last_supervisory_step = self.step
        return True
