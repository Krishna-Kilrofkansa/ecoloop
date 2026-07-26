"""Gemma 4 E2B agent orchestration via Ollama/LiteLLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from litellm import completion
from pydantic import ValidationError

from agent.prompts import SYSTEM_PROMPT, format_agent_prompt
from agent.safety import SafetyInterlock
from agent.schemas import HVACSetpointArgs
from simulation.shared_state import STATE, ControlCommand, TelemetrySnapshot


@dataclass
class AgentConfig:
    model: str = "ollama/gemma4:e2b"
    api_base: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 256
    timeout: int = 60


class EcoLoopAgent:
    """Supervisory LLM controller using Gemma 4 E2B (~2 GB quantized)."""

    def __init__(self, config: AgentConfig | None = None, safety: SafetyInterlock | None = None) -> None:
        self.config = config or AgentConfig()
        self.safety = safety or SafetyInterlock()
        self._fallback_enabled = True
        self._last_error: str = ""
        self._hold_setpoints: tuple[float, float] = (23.0, 21.0)

    def check_ollama_available(self) -> bool:
        try:
            r = httpx.get(f"{self.config.api_base.rstrip('/')}/api/tags", timeout=10.0)
            if r.status_code != 200:
                return False
            models = [m.get("name", "") for m in r.json().get("models", [])]
            return any("gemma4" in m and "e2b" in m for m in models)
        except Exception:
            return False

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        text = text.strip()
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            raise ValueError(f"No JSON object in response: {text[:200]}")
        return json.loads(match.group())

    def _parse_and_validate(self, text: str) -> HVACSetpointArgs:
        parsed = self._parse_json_response(text)
        return HVACSetpointArgs.model_validate(parsed)

    def _hold_current(self, reason: str, telemetry: TelemetrySnapshot | None) -> ControlCommand:
        """Self-correction: keep active setpoints, log error for next E2B cycle."""
        if telemetry:
            cool, heat = telemetry.cooling_setpoint_c, telemetry.heating_setpoint_c
        elif STATE.latest:
            cool, heat = STATE.latest.cooling_setpoint_c, STATE.latest.heating_setpoint_c
        else:
            cool, heat = self._hold_setpoints

        self._last_error = reason
        STATE.append_diagnostic(f"E2B hold: {reason}")
        return ControlCommand(
            cooling_setpoint_c=cool,
            heating_setpoint_c=heat,
            rationale=f"Hold setpoints — {reason}",
            source="gemma4:e2b_hold",
        )

    def _rule_based_fallback(self, summary: dict) -> ControlCommand:
        latest = summary.get("latest") or {}
        hour = latest.get("sim_hour", 12.0)
        pmv = latest.get("pmv", 0.0)
        carbon = latest.get("grid_carbon_gco2_kwh", 300.0)
        tariff = latest.get("grid_tariff_usd_kwh", 0.18)
        cool, heat = 23.0, 21.0

        if 4 <= hour < 8 and (carbon > 400 or tariff > 0.3):
            cool, heat = 24.0, 20.0  # cooling 24, heating 20 — valid E2B bounds
        elif 14 <= hour < 19 and (tariff >= 0.35 or carbon >= 450):
            cool, heat = 25.0, 21.0
        elif 22 <= hour or hour < 7:
            cool, heat = 26.0, 18.0

        if abs(pmv) > 0.45:
            cool = max(23.0, cool - 0.5)
            heat = min(21.0, heat + 0.5)

        return ControlCommand(
            cooling_setpoint_c=cool,
            heating_setpoint_c=heat,
            rationale="Rule-based fallback (Gemma 4 E2B unavailable)",
            source="fallback_rbc",
        )

    def decide(self, telemetry: TelemetrySnapshot | None = None) -> ControlCommand | None:
        summary = STATE.get_summary()
        if summary.get("status") == "no_data":
            return None

        if telemetry:
            self._hold_setpoints = (telemetry.cooling_setpoint_c, telemetry.heating_setpoint_c)

        user_prompt = format_agent_prompt(summary, self._last_error)
        self._last_error = ""

        command: ControlCommand | None = None
        try:
            response = completion(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_base=self.config.api_base,
                timeout=self.config.timeout,
            )
            raw_response = response.choices[0].message.content or ""
            STATE.last_agent_response = raw_response

            validated_args = self._parse_and_validate(raw_response)
            command = ControlCommand(
                cooling_setpoint_c=validated_args.cooling_setpoint_c,
                heating_setpoint_c=validated_args.heating_setpoint_c,
                rationale=validated_args.rationale or "E2B setpoint update",
                source="gemma4:e2b",
            )
        except (ValidationError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return self._hold_current(f"invalid output: {exc}", telemetry)
        except Exception as exc:
            STATE.append_diagnostic(f"LLM error: {exc}")
            if not self._fallback_enabled:
                return self._hold_current(str(exc), telemetry)
            command = self._rule_based_fallback(summary)

        sim_minute = 0.0
        if telemetry:
            sim_minute = telemetry.sim_day * 1440 + telemetry.sim_hour * 60
        elif STATE.latest:
            sim_minute = STATE.latest.sim_day * 1440 + STATE.latest.sim_hour * 60

        validated, err = self.safety.validate(command, telemetry or STATE.latest, sim_minute)
        if err:
            if "deadband" in err.lower():
                fixed = ControlCommand(
                    cooling_setpoint_c=command.cooling_setpoint_c,
                    heating_setpoint_c=min(command.cooling_setpoint_c - 1.5, 22.0),
                    rationale=command.rationale + " [deadband fix]",
                    source=command.source,
                )
                validated, err = self.safety.validate(fixed, telemetry or STATE.latest, sim_minute)
            if err:
                return self._hold_current(f"safety rejected: {err}", telemetry)

        if validated:
            STATE.last_command = validated
            self._hold_setpoints = (validated.cooling_setpoint_c, validated.heating_setpoint_c)
            snap = telemetry or STATE.latest
            if snap:
                snap.agent_action = validated.rationale
        return validated
