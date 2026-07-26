"""Generate reference comparison dataset for dashboard when no live run exists."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "telemetry_log.csv"


def generate():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(42)
    for mode in ("baseline", "agent"):
        zone_temp = 23.0
        for step in range(672):  # 7 days × 96 steps
            hour = (step * 0.25) % 24
            day = step // 96 + 1
            tout = 28 + 4 * np.sin((hour - 14) / 24 * 2 * np.pi)
            if mode == "baseline":
                cool, heat = (23, 21) if 7 <= hour < 22 else (28, 15)
            else:
                if 4 <= hour < 8:
                    cool, heat = 21.5, 20.0
                elif 14 <= hour < 19:
                    cool, heat = 25.0, 21.0
                elif 7 <= hour < 22:
                    cool, heat = 23.0, 21.0
                else:
                    cool, heat = 26.0, 18.0
            zone_temp += 0.1 * (cool - zone_temp) + 0.05 * (tout - zone_temp)
            pmv = 0.15 * (zone_temp - 23.0) + rng.normal(0, 0.05)
            power = 12 + max(0, zone_temp - cool) * 15
            if mode == "agent" and 14 <= hour < 19:
                power *= 0.72
            tariff = 0.35 if 14 <= hour < 19 else (0.18 if 10 <= hour < 22 else 0.08)
            carbon = 480 if tariff > 0.3 else 200
            rows.append({
                "timestamp": datetime.now().isoformat(),
                "sim_day": day,
                "sim_hour": round(hour, 2),
                "zone_temp_c": round(zone_temp, 2),
                "outdoor_temp_c": round(tout, 2),
                "zone_rh_percent": round(50 + rng.normal(0, 2), 1),
                "zone_co2_ppm": round(750 + 100 * (hour > 7 and hour < 22), 0),
                "pmv": round(pmv, 3),
                "elec_power_kw": round(power, 2),
                "cooling_setpoint_c": cool,
                "heating_setpoint_c": heat,
                "grid_tariff_usd_kwh": tariff,
                "grid_carbon_gco2_kwh": carbon,
                "mode": mode,
                "agent_action": "Gemma 4 peak shed" if mode == "agent" and 14 <= hour < 19 else "",
            })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    generate()
