"""Configuration loader."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agent.safety import SafetyConfig
from agent.controller import AgentConfig

ROOT = Path(__file__).resolve().parent


def load_settings() -> dict:
    load_dotenv(ROOT / ".env")
    path = ROOT / "config" / "settings.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["llm"]["model"] = os.getenv("ECOLOOP_LLM_MODEL", cfg["llm"]["model"])
    cfg["llm"]["api_base"] = os.getenv("OLLAMA_API_BASE", cfg["llm"]["api_base"])
    demo = os.getenv("ECOLOOP_DEMO_MODE", str(cfg["simulation"].get("demo_mode", False)))
    cfg["simulation"]["demo_mode"] = demo.lower() in ("1", "true", "yes")
    return cfg


def agent_config_from_settings(cfg: dict) -> AgentConfig:
    llm = cfg["llm"]
    model = llm["model"]
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    return AgentConfig(
        model=model,
        api_base=llm["api_base"],
        temperature=llm.get("temperature", 0.2),
        max_tokens=llm.get("max_tokens", 1024),
        timeout=llm.get("timeout_seconds", 120),
    )


def safety_config_from_settings(cfg: dict) -> SafetyConfig:
    sp = cfg["setpoints"]
    comfort = cfg["comfort"]
    return SafetyConfig(
        cooling_min=sp["cooling_min"],
        cooling_max=sp["cooling_max"],
        heating_min=sp["heating_min"],
        heating_max=sp["heating_max"],
        min_deadband=sp["min_deadband"],
        max_ramp_c_per_hour=sp["max_ramp_c_per_hour"],
        min_dwell_minutes=sp["min_dwell_minutes"],
        co2_max=comfort["co2_max_ppm"],
        rh_max=comfort["rh_max_percent"],
    )
