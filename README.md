# Eco-Loop Building Agents

> **Autonomous closed-loop building energy optimization** combining EnergyPlus simulation, Gemma 4 E2B (open-source LLM), and Model Context Protocol (MCP).

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)]()
[![Gemma 4 E2B](https://img.shields.io/badge/LLM-Gemma_4_E2B-4285F4?logo=google&logoColor=white)]()
[![Ollama](https://img.shields.io/badge/Runtime-Ollama-000000?logo=ollama)]()
[![EnergyPlus](https://img.shields.io/badge/Simulation-EnergyPlus-FF6F00)]()
[![MCP](https://img.shields.io/badge/Protocol-MCP-8B5CF6)]()

---

## The Problem

Buildings consume **~40% of global energy**. Traditional Building Management Systems use rigid, rule-based schedules that cannot adapt to real-time weather, occupancy, or grid conditions — wasting energy and missing optimization opportunities.

## Our Solution

Eco-Loop transforms a building into an **autonomous, self-correcting agent**:

- **Gemma 4 E2B** (5.1B params, ~3 GB) reasons about energy cost, carbon intensity, and occupant comfort
- **Physics-based simulation** (EnergyPlus or built-in surrogate) provides real-time building telemetry
- **4-layer safety interlock** guarantees setpoints stay within safe bounds (no LLM hallucination risk)
- **Model Context Protocol** enables any MCP-compatible agent to query/control the building
- **Self-correcting loop**: invalid LLM outputs trigger fallback, errors feed back into next decision cycle

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│   Building   │────▶│  Shared State    │────▶│  Gemma 4 E2B  │
│  Simulation  │     │  (Telemetry)     │     │  (via Ollama)  │
└──────────────┘     └──────────────────┘     └───────┬───────┘
       ▲                                              │
       │              ┌──────────────────┐            ▼
       └──────────────│ Safety Interlock │◀───── Setpoint Cmd
                      │ (4-Layer Check)  │
                      └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              ┌──────────┐      ┌────────────┐
              │Dashboard │      │ MCP Server │
              │(Streamlit)│      │ (FastMCP)  │
              └──────────┘      └────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full system design with Mermaid diagrams.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Gemma 4 E2B (Q4_K_M, 5.1B) | Cognitive HVAC decision-making |
| **LLM Runtime** | Ollama | Local inference, no cloud dependency |
| **Simulation** | EnergyPlus / Python Surrogate | Physics-based energy modeling |
| **Protocol** | Model Context Protocol (MCP) | Standardized agent-tool API |
| **Backend** | Python 3.11, LiteLLM, Pydantic | Orchestration & validation |
| **Dashboard** | Streamlit + Plotly | Real-time visualization |
| **Safety** | SafetyInterlock (custom) | Bounds, deadband, ramp-rate, dwell |

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Set Up Gemma 4 E2B

```bash
ollama pull gemma4:e2b
ollama list                   # Verify model appears
```

### 3. Configure Environment

```bash
copy .env.example .env
```

### 4. Run Simulation (Demo Mode — No EnergyPlus Required)

```bash
# Full run: baseline comparison + agent closed-loop
python main.py --mode full --demo

# Agent loop only
python main.py --mode loop --demo

# Baseline only
python main.py --mode baseline --demo
```

### 5. Launch Dashboard

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) to view real-time energy savings, thermal comfort, and grid response.

### 6. Start MCP Server (Optional)

```bash
python main.py --mode mcp
```

---

## How It Works

1. **Baseline Pass** — Runs the building with fixed rule-based schedules (23/21°C day, 28/15°C night), records total energy consumption
2. **Agent Loop** — Every supervisory interval (default: 2 hours), the agent:
   - Reads current telemetry (temp, humidity, CO2, PMV, grid pricing, carbon intensity)
   - Queries Gemma 4 E2B for optimal setpoints
   - Validates through 4-layer safety interlock
   - Applies to HVAC (or holds current setpoints on failure)
3. **Comparison** — Dashboard shows % energy savings, comfort deviation, and cost reduction vs baseline

---

## Safety Interlock (4 Layers)

Every command from the LLM passes through deterministic safety checks:

| Layer | Check | Parameter |
|-------|-------|-----------|
| 1 | **Absolute Bounds** | Cooling: 18–28°C, Heating: 12–22°C |
| 2 | **Deadband** | `cooling − heating ≥ 1.5°C` (auto-fixed if violated) |
| 3 | **Ramp Rate** | `≤ 1.0°C/hour` (clamped if exceeded) |
| 4 | **Dwell Time** | `≥ 30 min` between setpoint changes |

Additional environmental guards: zone temp 18–27°C, RH ≤ 65%, CO2 ≤ 1000 ppm.

---

## LLM Configuration

```yaml
# config/settings.yaml
llm:
  provider: ollama
  model: gemma4:e2b
  temperature: 0.1          # Low for deterministic outputs
  max_tokens: 2048
  timeout_seconds: 300
```

The agent prompt is intentionally compact for E2B's small context — sends a flat JSON telemetry snapshot and expects a JSON setpoint response.

---

## Live EnergyPlus Mode

For production-grade physics simulation:

1. Install [EnergyPlus 25.1+](https://energyplus.net/downloads)
2. Download Atlanta TMY3 weather file → save as `config/weather_Atlanta.epw`
3. Set `ENERGYPLUS_DIR` in `.env`
4. In `config/settings.yaml`, set `demo_mode: false`
5. Run: `python main.py --mode full`

---

## Project Structure

```
honeywell_track1/
├── main.py                    # Orchestrator (baseline + agent loop + checkpointing)
├── config_loader.py           # YAML → dataclass configuration
├── config/
│   ├── settings.yaml          # All tunable parameters
│   ├── baseline_medium_office.idf
│   └── WEATHER_README.md
├── agent/
│   ├── controller.py          # EcoLoopAgent — LLM calls + fallback
│   ├── safety.py              # SafetyInterlock — 4-layer validation
│   ├── prompts.py             # Compact prompt templates
│   ├── schemas.py             # Pydantic: CompactTelemetry, HVACSetpointArgs
│   └── mcp_client.py          # MCP client helper
├── simulation/
│   ├── demo_engine.py         # Physics surrogate (no EnergyPlus needed)
│   ├── eplus_bridge.py        # Full EnergyPlus co-simulation
│   ├── shared_state.py        # Thread-safe telemetry store (singleton)
│   └── comfort.py             # PMV comfort + grid signal calculations
├── mcp_server/
│   └── server.py              # FastMCP server (tools + resources)
├── dashboard/
│   └── app.py                 # Streamlit real-time dashboard
├── output/                    # Simulation outputs
├── docs/
│   └── ARCHITECTURE.md        # Detailed architecture with diagrams
└── requirements.txt
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Gemma 4 E2B** over larger models | Runs on 8 GB RAM, edge-deployable, no cloud latency |
| **Deterministic safety layer** | LLM outputs are never trusted directly — always validated |
| **Self-correction via error feedback** | Invalid LLM responses are logged and injected into the next prompt |
| **Rule-based fallback** | System never stops — degrades gracefully to RBC when LLM is unavailable |
| **MCP protocol** | Future-proof: any MCP-compatible agent can plug into the building |
| **Demo surrogate** | Enables full PoC without EnergyPlus installation |

---

## References

- [EnergyPlus](https://energyplus.net/) — DOE building energy simulation
- [Gemma 4](https://ai.google.dev/gemma) — Google's open-source LLM family
- [Ollama](https://ollama.com/) — Local LLM runtime
- [Model Context Protocol](https://modelcontextprotocol.io/) — Standardized agent protocol
- [ASHRAE 90.1-2022](https://www.ashrae.org/) — Energy Standard for Buildings
- [ISO 7730:2005](https://www.iso.org/) — PMV/PPD Thermal Comfort

---

## License

MIT — Honeywell Hackathon 2026
