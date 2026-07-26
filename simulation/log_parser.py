"""EnergyPlus error log parser for LLM self-correction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedError:
    severity: str
    message: str
    line_hint: str = ""


FATAL = re.compile(r"\*\*\s*Fatal\s*\*\*", re.I)
SEVERE = re.compile(r"\*\*\s*Severe\s*\*\*", re.I)
WARMUP = re.compile(r"Warming up|Initializing Simulation|Performing Zone Sizing", re.I)


def parse_eplus_errors(err_path: Path, max_entries: int = 20) -> dict[str, list[str]]:
    """Parse .err file into fatal, severe, and filtered info buckets."""
    result: dict[str, list[str]] = {"fatal": [], "severe": [], "info": []}
    if not err_path.exists():
        return result

    lines = err_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or WARMUP.search(stripped):
            continue
        if FATAL.search(stripped):
            result["fatal"].append(stripped)
        elif SEVERE.search(stripped):
            if "deadband" in stripped.lower() or "setpoint" in stripped.lower():
                result["severe"].insert(0, stripped)
            else:
                result["severe"].append(stripped)
        elif "warning" in stripped.lower() and len(result["info"]) < max_entries:
            result["info"].append(stripped)

    for key in result:
        result[key] = result[key][:max_entries]
    return result


def summarize_for_agent(err_path: Path) -> str:
    """Compact diagnostic string for LLM context."""
    parsed = parse_eplus_errors(err_path)
    if not any(parsed.values()):
        return "No simulation errors detected."
    parts = []
    if parsed["fatal"]:
        parts.append("FATAL: " + "; ".join(parsed["fatal"][:3]))
    if parsed["severe"]:
        parts.append("SEVERE: " + "; ".join(parsed["severe"][:5]))
    if not parts:
        parts.append("INFO warnings only (warmup filtered).")
    return "\n".join(parts)
