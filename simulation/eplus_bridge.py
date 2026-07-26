"""PyEnergyPlus runtime bridge for live closed-loop control."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Callable

from simulation.comfort import estimate_pmv, grid_signals
from simulation.shared_state import STATE, TelemetrySnapshot


class EnergyPlusBridge:
    """Wraps pyenergyplus API with EMS actuator injection."""

    def __init__(
        self,
        idf_path: Path,
        epw_path: Path,
        output_dir: Path,
        mode: str = "agent",
    ) -> None:
        self.idf_path = idf_path
        self.epw_path = epw_path
        self.output_dir = output_dir
        self.mode = mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cooling_sp = 23.0
        self.heating_sp = 21.0
        self._api = None
        self._state = None
        self._handles: dict[str, int] = {}
        self._on_supervisory: Callable | None = None
        self._step = 0

    def set_supervisory_callback(self, callback: Callable) -> None:
        self._on_supervisory = callback

    def apply_command(self, cooling: float, heating: float) -> None:
        self.cooling_sp = cooling
        self.heating_sp = heating
        if self._api and self._state:
            cool_h = self._handles.get("cooling_schedule")
            heat_h = self._handles.get("heating_schedule")
            if cool_h and cool_h > 0:
                self._api.exchange.set_actuator_value(self._state, cool_h, cooling)
            if heat_h and heat_h > 0:
                self._api.exchange.set_actuator_value(self._state, heat_h, heating)

    def _init_handles(self) -> None:
        api = self._api
        state = self._state
        ex = api.exchange

        self._handles["zone_temp"] = ex.get_variable_handle(
            state, "Zone Mean Air Temperature", "Perimeter_ZN_1_ZN"
        )
        self._handles["outdoor_temp"] = ex.get_variable_handle(
            state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        self._handles["zone_rh"] = ex.get_variable_handle(
            state, "Zone Air Relative Humidity", "Perimeter_ZN_1_ZN"
        )
        self._handles["co2"] = ex.get_variable_handle(
            state, "Zone Air CO2 Concentration", "Perimeter_ZN_1_ZN"
        )
        self._handles["facility_power"] = ex.get_variable_handle(
            state, "Facility Total Electricity Demand Rate", "Whole Building"
        )
        self._handles["cooling_schedule"] = ex.get_actuator_handle(
            state, "Schedule:Compact", "Schedule Value", "ECOLOOP_COOLING_SP"
        )
        self._handles["heating_schedule"] = ex.get_actuator_handle(
            state, "Schedule:Compact", "Schedule Value", "ECOLOOP_HEATING_SP"
        )

    def _callback(self, state) -> None:
        self._state = state
        api = self._api
        ex = api.exchange

        if not self._handles:
            self._init_handles()

        day = ex.day_of_simulation(state)
        hour = ex.hour_of_simulation(state)
        minute = ex.minutes(state)
        sim_hour = hour + minute / 60.0

        zone_temp = ex.get_variable_value(state, self._handles["zone_temp"]) if self._handles["zone_temp"] > 0 else 23.0
        outdoor = ex.get_variable_value(state, self._handles["outdoor_temp"]) if self._handles["outdoor_temp"] > 0 else 30.0
        rh = ex.get_variable_value(state, self._handles["zone_rh"]) if self._handles["zone_rh"] > 0 else 50.0
        co2 = ex.get_variable_value(state, self._handles["co2"]) if self._handles["co2"] > 0 else 800.0
        power_w = ex.get_variable_value(state, self._handles["facility_power"]) if self._handles["facility_power"] > 0 else 0.0

        pmv = estimate_pmv(zone_temp, rh_percent=rh)
        tariff, carbon = grid_signals(sim_hour, day)

        snap = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            sim_day=day,
            sim_hour=sim_hour,
            zone_temp_c=round(zone_temp, 2),
            outdoor_temp_c=round(outdoor, 2),
            zone_rh_percent=round(rh, 1),
            zone_co2_ppm=round(co2, 0),
            pmv=round(pmv, 3),
            elec_power_kw=round(power_w / 1000.0, 3),
            cooling_setpoint_c=self.cooling_sp,
            heating_setpoint_c=self.heating_sp,
            grid_tariff_usd_kwh=tariff,
            grid_carbon_gco2_kwh=carbon,
            mode=self.mode,
        )
        STATE.push_telemetry(snap)
        self._step += 1

        if self._on_supervisory and self._step % 4 == 0:
            self._on_supervisory(snap)

        self.apply_command(self.cooling_sp, self.heating_sp)

    def run(self) -> None:
        from pyenergyplus.api import EnergyPlusAPI

        eplus_dir = os.environ.get("ENERGYPLUS_DIR", "")
        self._api = EnergyPlusAPI(eplus_dir if eplus_dir else None)
        api = self._api
        STATE.simulation_running = True

        state = api.state_manager.new_state()

        def handler(st):
            self._callback(st)

        api.runtime.callback_begin_zone_timestep_before_set_current_weather(
            state, handler
        )
        api.runtime.run_energyplus(
            state,
            [
                "-w", str(self.epw_path),
                "-d", str(self.output_dir),
                str(self.idf_path),
            ],
        )
        api.state_manager.delete_state(state)
        STATE.simulation_running = False
