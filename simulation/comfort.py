"""Fanger PMV approximation for thermal comfort telemetry."""

from __future__ import annotations

import math


def estimate_pmv(
    air_temp_c: float,
    mrt_c: float | None = None,
    rh_percent: float = 50.0,
    air_velocity: float = 0.1,
    metabolic_rate: float = 1.2,
    clothing_insulation: float = 0.5,
) -> float:
    """Simplified PMV estimate (ASHRAE-style) for demo and live loops."""
    ta = air_temp_c
    tr = mrt_c if mrt_c is not None else air_temp_c
    v = max(air_velocity, 0.05)
    rh = max(min(rh_percent, 100.0), 1.0)

    pa = rh / 100.0 * 611.0 * math.exp(17.27 * ta / (237.7 + ta))

    m = metabolic_rate * 58.15
    icl = clothing_insulation * 0.155
    fcl = 1.0 + 0.31 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(v)
    tcl = ta + (35.5 - ta) / (3.5 * (6.45 * icl + 0.1))
    hc = max(hcf, 2.38 * abs(tcl - ta) ** 0.25)
    radiative = fcl * 4.0 * 5.67e-8 * ((tr + 273.15) ** 4 - (tcl + 273.15) ** 4)
    convective = fcl * hc * (tcl - ta)
    evap = 2.2 * (5733 - 6.99 * m - pa) * (tcl - ta) if m > 58.15 else 0.0
    pmv = (0.303 * math.exp(-0.036 * m) + 0.028) * (m - 3.05e-3 * (5733 - 6.99 * m - pa) - evap - radiative - convective)
    return max(-3.0, min(3.0, pmv / 50.0))


def grid_signals(hour: float, day: int = 1) -> tuple[float, float]:
    """Time-of-use tariff ($/kWh) and carbon intensity (gCO2/kWh)."""
    if 14 <= hour < 19:
        tariff = 0.35
        carbon = 480.0
    elif 10 <= hour < 14 or 19 <= hour < 22:
        tariff = 0.18
        carbon = 320.0
    else:
        tariff = 0.08
        carbon = 180.0
    if day % 7 in (6, 0):
        tariff *= 0.9
        carbon *= 0.85
    return tariff, carbon
