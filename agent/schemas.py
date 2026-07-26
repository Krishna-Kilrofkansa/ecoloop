"""Pydantic schemas for Gemma 4 E2B — compact payloads and strict tool bounds."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


GridTier = Literal["OFF_PEAK", "MID_PEAK", "HIGH_PEAK"]


class CompactTelemetry(BaseModel):
    """High-density telemetry JSON for E2B (no prose)."""

    zone_temp_c: float = Field(description="Zone mean air temperature °C")
    outdoor_temp_c: float = Field(description="Outdoor drybulb °C")
    pmv_index: float = Field(description="Predicted Mean Vote")
    co2_ppm: float = Field(description="Zone CO2 concentration")
    power_kw: float = Field(description="Facility electricity demand kW")
    rh_percent: float = Field(description="Zone relative humidity %")
    grid_tier: GridTier = Field(description="Tariff/carbon tier")
    sim_hour: float = Field(description="Simulation hour of day")
    cooling_sp_c: float = Field(description="Active cooling setpoint °C")
    heating_sp_c: float = Field(description="Active heating setpoint °C")


class HVACSetpointArgs(BaseModel):
    """Strict single-tool schema — bounded floats prevent E2B hallucinations."""

    heating_setpoint_c: float = Field(..., ge=18.0, le=22.0, description="Heating setpoint in Celsius")
    cooling_setpoint_c: float = Field(..., ge=21.0, le=28.0, description="Cooling setpoint in Celsius")
    rationale: str = Field(default="", max_length=1000, description="Brief reason")

    @model_validator(mode="after")
    def enforce_deadband(self) -> HVACSetpointArgs:
        if self.heating_setpoint_c + 1.0 > self.cooling_setpoint_c:
            raise ValueError(
                f"Deadband violation: heating {self.heating_setpoint_c}°C must be ≥1°C below cooling {self.cooling_setpoint_c}°C"
            )
        return self


def tariff_to_grid_tier(tariff: float) -> GridTier:
    if tariff >= 0.30:
        return "HIGH_PEAK"
    if tariff >= 0.12:
        return "MID_PEAK"
    return "OFF_PEAK"


def compact_telemetry_from_latest(latest: dict) -> CompactTelemetry:
    """Flatten a telemetry snapshot dict into E2B-friendly compact JSON."""
    return CompactTelemetry(
        zone_temp_c=latest["zone_temp_c"],
        outdoor_temp_c=latest["outdoor_temp_c"],
        pmv_index=latest["pmv"],
        co2_ppm=latest["zone_co2_ppm"],
        power_kw=latest["elec_power_kw"],
        rh_percent=latest["zone_rh_percent"],
        grid_tier=tariff_to_grid_tier(latest["grid_tariff_usd_kwh"]),
        sim_hour=latest["sim_hour"],
        cooling_sp_c=latest["cooling_setpoint_c"],
        heating_sp_c=latest["heating_setpoint_c"],
    )
