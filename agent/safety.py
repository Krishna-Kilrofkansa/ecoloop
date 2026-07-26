"""Deterministic safety interlock for setpoint commands."""

from __future__ import annotations

from dataclasses import dataclass

from simulation.shared_state import ControlCommand, TelemetrySnapshot


@dataclass
class SafetyConfig:
    cooling_min: float = 18.0
    cooling_max: float = 28.0
    heating_min: float = 12.0
    heating_max: float = 22.0
    min_deadband: float = 1.5
    max_ramp_c_per_hour: float = 1.0
    min_dwell_minutes: float = 30.0
    zone_temp_min: float = 18.0
    zone_temp_max: float = 27.0
    rh_max: float = 65.0
    co2_max: float = 1000.0


class SafetyInterlock:
    """Validates agent commands before actuator injection."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self._last_command: ControlCommand | None = None
        self._last_change_minute: float = -999.0

    def validate(
        self,
        command: ControlCommand,
        telemetry: TelemetrySnapshot | None,
        sim_minute: float,
    ) -> tuple[ControlCommand | None, str | None]:
        cfg = self.config
        cool = command.cooling_setpoint_c
        heat = command.heating_setpoint_c

        if not (cfg.cooling_min <= cool <= cfg.cooling_max):
            return None, f"Cooling setpoint {cool}°C outside [{cfg.cooling_min}, {cfg.cooling_max}]"
        if not (cfg.heating_min <= heat <= cfg.heating_max):
            return None, f"Heating setpoint {heat}°C outside [{cfg.heating_min}, {cfg.heating_max}]"
        if heat + cfg.min_deadband > cool:
            adjusted_heat = cool - cfg.min_deadband
            command = ControlCommand(
                cooling_setpoint_c=cool,
                heating_setpoint_c=adjusted_heat,
                rationale=command.rationale + " [deadband auto-fix]",
                source=command.source,
            )
            heat = adjusted_heat

        if self._last_command is not None:
            elapsed = sim_minute - self._last_change_minute
            max_delta = cfg.max_ramp_c_per_hour * (elapsed / 60.0) if elapsed > 0 else cfg.max_ramp_c_per_hour
            dc = abs(cool - self._last_command.cooling_setpoint_c)
            dh = abs(heat - self._last_command.heating_setpoint_c)
            if elapsed < cfg.min_dwell_minutes and (dc > 0.01 or dh > 0.01):
                return None, f"Dwell time not met ({elapsed:.0f} < {cfg.min_dwell_minutes} min)"
            if dc > max_delta + 0.01:
                cool = self._last_command.cooling_setpoint_c + max_delta * (1 if cool > self._last_command.cooling_setpoint_c else -1)
            if dh > max_delta + 0.01:
                heat = self._last_command.heating_setpoint_c + max_delta * (1 if heat > self._last_command.heating_setpoint_c else -1)
            command = ControlCommand(
                cooling_setpoint_c=round(cool, 2),
                heating_setpoint_c=round(heat, 2),
                rationale=command.rationale,
                source=command.source,
            )

        if telemetry:
            if not (cfg.zone_temp_min <= telemetry.zone_temp_c <= cfg.zone_temp_max):
                return None, f"Zone temp {telemetry.zone_temp_c}°C outside safety envelope"
            if telemetry.zone_rh_percent > cfg.rh_max:
                return None, f"RH {telemetry.zone_rh_percent}% exceeds {cfg.rh_max}%"
            if telemetry.zone_co2_ppm > cfg.co2_max:
                command = ControlCommand(
                    cooling_setpoint_c=min(command.cooling_setpoint_c, telemetry.zone_temp_c),
                    heating_setpoint_c=command.heating_setpoint_c,
                    rationale=command.rationale + " [CO2 mitigation]",
                    source=command.source,
                )

        self._last_command = command
        self._last_change_minute = sim_minute
        return command, None
