# Eco-Loop Architecture

> Autonomous closed-loop building energy optimization using Gemma 4 E2B, EnergyPlus, and Model Context Protocol.

---

## System Overview

Eco-Loop transforms a building from a passive energy consumer into an **autonomous, self-correcting agent**. The system pairs a physics-based simulation engine with an open-source LLM to continuously optimize HVAC setpoints across three objectives: **energy cost**, **carbon emissions**, and **occupant comfort**.

```mermaid
graph TB
    subgraph Building["🏢 Building Environment"]
        SENSORS["📡 Sensors<br/>Temperature | Humidity | CO2 | Occupancy"]
        HVAC["❄️ HVAC Plant<br/>Cooling & Heating Actuators"]
    end

    subgraph Core["🧠 Eco-Loop Core"]
        SIM["⚙️ Simulation Engine<br/>EnergyPlus / Physics Surrogate"]
        LLM["🤖 Gemma 4 E2B<br/>via Ollama (5.1B, Q4_K_M)"]
        SAFETY["🛡️ Safety Interlock<br/>4-Layer Validation"]
        STATE["📊 Shared State<br/>Thread-safe Telemetry Store"]
    end

    subgraph Signals["🌐 External Signals"]
        GRID["⚡ Grid<br/>Tariff & Carbon Intensity"]
        WEATHER["🌤️ Weather<br/>Outdoor Temperature"]
    end

    subgraph Interfaces["📈 Interface Layer"]
        DASH["🖥️ Streamlit Dashboard<br/>Real-time Monitoring"]
        MCP["🔌 MCP Server<br/>Standardized Agent API"]
    end

    SENSORS -->|telemetry| STATE
    WEATHER -->|outdoor conditions| SIM
    GRID -->|pricing & carbon| LLM
    STATE -->|context window| LLM
    LLM -->|setpoint commands| SAFETY
    SAFETY -->|validated commands| HVAC
    SIM -->|simulated snapshots| STATE
    STATE -->|live data| DASH
    STATE -->|tool/resource API| MCP
    HVAC -->|thermal response| SENSORS

    style Core fill:#1a1a2e,stroke:#00d4ff,stroke-width:2px,color:#fff
    style Building fill:#0d2137,stroke:#00ff88,stroke-width:2px,color:#fff
    style Signals fill:#1a0d37,stroke:#ff6b9d,stroke-width:2px,color:#fff
    style Interfaces fill:#0d3721,stroke:#ffd700,stroke-width:2px,color:#fff
```

---

## Project Structure

```
honeywell_track1/
├── main.py                          # Orchestrator — baseline + agent loop + checkpointing
├── config_loader.py                 # YAML config → dataclass mapping
│
├── config/
│   ├── settings.yaml                # All tunable parameters
│   ├── baseline_medium_office.idf   # EnergyPlus building model (stub)
│   └── WEATHER_README.md            # Instructions for EPW download
│
├── agent/
│   ├── __init__.py
│   ├── controller.py                # EcoLoopAgent — LLM inference + fallback logic
│   ├── safety.py                    # SafetyInterlock — 4-layer command validation
│   ├── prompts.py                   # System & user prompt templates for Gemma 4 E2B
│   ├── schemas.py                   # Pydantic schemas (CompactTelemetry, HVACSetpointArgs)
│   └── mcp_client.py                # MCP client helper
│
├── simulation/
│   ├── demo_engine.py               # DemoBuildingSimulator — physics surrogate (no EnergyPlus)
│   ├── eplus_bridge.py              # EnergyPlusBridge — full EnergyPlus co-simulation
│   ├── shared_state.py              # SharedState singleton — thread-safe telemetry store
│   └── comfort.py                   # PMV calculation + grid signal functions
│
├── mcp_server/
│   ├── __init__.py
│   └── server.py                    # FastMCP server with tools & resources
│
├── dashboard/
│   └── app.py                       # Streamlit real-time dashboard
│
├── output/                          # Simulation outputs (telemetry CSV, checkpoints)
├── docs/                            # Documentation
│   └── ARCHITECTURE.md              # This file
├── requirements.txt
└── README.md
```

---

## Component Deep Dive

### 1. Simulation Engine

Two interchangeable backends behind a common interface:

```mermaid
graph LR
    subgraph Engines["Simulation Backends"]
        DEMO["DemoBuildingSimulator<br/>Pure Python<br/>1st-order thermal dynamics<br/>Instant execution"]
        EPLUS["EnergyPlusBridge<br/>Full EnergyPlus co-sim<br/>DOE-2 validated physics<br/>IDF + EPW required"]
    end

    MAIN["main.py<br/>--demo flag"]
    MAIN -->|"demo=True"| DEMO
    MAIN -->|"demo=False"| EPLUS

    DEMO --> STATE["SharedState"]
    EPLUS --> STATE

    style DEMO fill:#1a3a5c,stroke:#60a5fa,color:#fff
    style EPLUS fill:#3a1a5c,stroke:#a78bfa,color:#fff
```

**Demo Engine Physics Model:**
```
zone_temp += 0.25 × (target − zone_temp) + 0.08 × (outdoor − zone_temp) + 0.03 × occupancy
humidity  = clamp(RH + 0.1 × (zone_temp − 23.0), 35%, 70%)
CO2       = 400 + 450 × occupancy + 5 × (zone_temp − 22.0)
power     = base_load + cooling_load × occupancy + outdoor_heat_gain
```

**Timestep:** Configurable (default 15 minutes)
**Run Period:** Configurable (default 1–7 days)

---

### 2. Agent Controller (EcoLoopAgent)

The cognitive engine that decides HVAC setpoints every supervisory interval.

```mermaid
flowchart TD
    START["Supervisory Trigger<br/>(every N steps or comfort alert)"] --> SUMMARY["Build Context<br/>STATE.get_summary()"]
    SUMMARY --> PROMPT["Format Prompt<br/>CompactTelemetry JSON + last_error"]
    PROMPT --> LLM["Call Gemma 4 E2B<br/>via LiteLLM → Ollama<br/>temp=0.1, max_tokens=256"]

    LLM --> PARSE{"Parse JSON<br/>from Response?"}
    PARSE -->|Valid| VALIDATE["Pydantic Validation<br/>HVACSetpointArgs"]
    PARSE -->|No JSON found| HOLD["Hold Current<br/>Setpoints"]
    PARSE -->|LLM Timeout| FALLBACK["Rule-Based<br/>Fallback"]

    VALIDATE --> SAFETY["Safety Interlock<br/>4-Layer Check"]
    FALLBACK --> SAFETY

    SAFETY --> SAFE{"Passes<br/>Safety?"}
    SAFE -->|Yes| APPLY["Apply Command<br/>to HVAC"]
    SAFE -->|Deadband Issue| FIX["Auto-fix Deadband<br/>Retry Validation"]
    SAFE -->|Rejected| HOLD
    FIX --> SAFETY

    HOLD --> LOG["Log Diagnostic<br/>Pass Error to Next Cycle"]
    APPLY --> LOG

    style LLM fill:#ff6b9d,stroke:#fff,color:#000
    style SAFETY fill:#ffd700,stroke:#fff,color:#000
    style APPLY fill:#00ff88,stroke:#fff,color:#000
    style HOLD fill:#ff4444,stroke:#fff,color:#fff
```

**Self-Correction Mechanism:**
- If the LLM produces invalid JSON → setpoints are held, error is injected into the *next* prompt via `last_error`
- The LLM sees its own mistake and corrects on the next cycle
- If the LLM is completely unavailable → rule-based fallback takes over (no interruption)

**LLM Prompt Design (Compact):**
```
System: "You are Eco-Loop HVAC supervisor. Output ONLY JSON:
         {heating_setpoint_c, cooling_setpoint_c, rationale}"

User:   "telemetry={compact_json} last_error={error_or_none}
         Return setpoints JSON only."
```

---

### 3. Safety Interlock

A **deterministic, non-bypassable** validation layer between the LLM and the HVAC actuators.

```mermaid
flowchart TD
    CMD["Incoming Command<br/>from Agent"] --> L1

    subgraph INTERLOCK["🛡️ Safety Interlock — 4 Layers"]
        L1{"Layer 1: Absolute Bounds<br/>Cooling: 18–28°C<br/>Heating: 12–22°C"}
        L1 -->|Pass| L2
        L1 -->|Fail| REJECT

        L2{"Layer 2: Deadband<br/>cooling − heating ≥ 1.5°C"}
        L2 -->|Pass| L3
        L2 -->|Violation| AUTOFIX["Auto-fix:<br/>heat = cool − 1.5"]
        AUTOFIX --> L3

        L3{"Layer 3: Ramp Rate<br/>≤ 1.0°C per hour"}
        L3 -->|Pass| L4
        L3 -->|Exceeds| CLAMP["Clamp to<br/>max allowed delta"]
        CLAMP --> L4

        L4{"Layer 4: Dwell Time<br/>≥ 30 min since<br/>last change"}
        L4 -->|Pass| ENV
        L4 -->|Too soon| REJECT
    end

    ENV{"Environmental Check<br/>Zone temp in 18–27°C?<br/>RH ≤ 65%? CO2 ≤ 1000?"}
    ENV -->|Pass| ACCEPT["✅ Apply to HVAC"]
    ENV -->|CO2 High| MITIGATE["Tighten cooling<br/>for ventilation"]
    ENV -->|Temp/RH Fail| REJECT

    MITIGATE --> ACCEPT
    REJECT["❌ Reject → Hold Setpoints"]

    style INTERLOCK fill:#1a1a2e,stroke:#ffd700,stroke-width:2px,color:#fff
    style ACCEPT fill:#0d3721,stroke:#00ff88,stroke-width:2px,color:#fff
    style REJECT fill:#370d0d,stroke:#ff4444,stroke-width:2px,color:#fff
```

**Key Safety Parameters (from `config/settings.yaml`):**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `cooling_min / max` | 18°C / 28°C | Absolute cooling bounds |
| `heating_min / max` | 12°C / 22°C | Absolute heating bounds |
| `min_deadband` | 1.5°C | Prevents cool/heat fighting |
| `max_ramp_c_per_hour` | 1.0°C | Prevents thermal shock |
| `min_dwell_minutes` | 30 min | Prevents oscillation |

---

### 4. Shared State (Thread-Safe Singleton)

Central data store accessed by simulation, agent, dashboard, and MCP server concurrently.

```mermaid
graph TB
    subgraph SharedState["📊 SharedState Singleton"]
        LATEST["latest: TelemetrySnapshot"]
        HISTORY["history: deque (maxlen=10000)"]
        ROLLING["rolling_window: deque (maxlen=100)"]
        METRICS["Metrics:<br/>total_kwh_baseline<br/>total_kwh_agent<br/>discomfort_hours_*"]
        DIAG["diagnostics: list[str]"]
        LOCK["threading.Lock()"]
    end

    SIM["Simulation Engine"] -->|"push_telemetry()"| SharedState
    AGENT["Agent Controller"] -->|"get_summary()"| SharedState
    DASH["Dashboard"] -->|"read latest/history"| SharedState
    MCP["MCP Server"] -->|"read via tools"| SharedState

    style SharedState fill:#1f2d2d,stroke:#34d399,stroke-width:2px,color:#fff
```

**TelemetrySnapshot Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `zone_temp_c` | float | Current zone temperature |
| `outdoor_temp_c` | float | Outdoor temperature |
| `zone_rh_percent` | float | Relative humidity |
| `zone_co2_ppm` | float | CO2 concentration |
| `pmv` | float | Predicted Mean Vote comfort index |
| `elec_power_kw` | float | Electrical power consumption |
| `cooling_setpoint_c` | float | Active cooling setpoint |
| `heating_setpoint_c` | float | Active heating setpoint |
| `grid_tariff_usd_kwh` | float | Current electricity price |
| `grid_carbon_gco2_kwh` | float | Current grid carbon intensity |
| `sim_day` / `sim_hour` | int/float | Simulation time |

---

### 5. MCP Server

Exposes building telemetry and control via the **Model Context Protocol**, enabling any MCP-compatible AI agent to interact with the building.

**Tools:**
| Tool | Description |
|------|-------------|
| `apply_hvac_setpoints` | Set cooling/heating setpoints (validated through SafetyInterlock) |
| `get_telemetry` | Current building telemetry snapshot |
| `get_diagnostics` | Agent decision log |

**Resources:**
| URI | Description |
|-----|-------------|
| `eco-loop://telemetry/latest` | Latest telemetry as JSON |
| `eco-loop://metrics/summary` | Energy savings & comfort metrics |

---

### 6. Dashboard

Real-time Streamlit dashboard with auto-refresh.

**Tabs:**
| Tab | Content |
|-----|---------|
| Overview | KPI cards (savings %, kWh, comfort hours) |
| Thermal Comfort | Zone temp, PMV, setpoint timeseries |
| Setpoints & Grid | Setpoint history, tariff overlay, carbon intensity |

---

## Data Flow — Complete Closed Loop

```mermaid
sequenceDiagram
    participant MAIN as main.py
    participant BASE as Baseline Sim
    participant AGENT_SIM as Agent Sim
    participant STATE as SharedState
    participant AGENT as EcoLoopAgent
    participant LLM as Gemma 4 E2B
    participant SAFETY as SafetyInterlock
    participant DASH as Dashboard
    participant CSV as telemetry_log.csv

    Note over MAIN: Phase 1 — Baseline
    MAIN->>BASE: run(mode="baseline")
    loop Every 15-min step
        BASE->>STATE: push_telemetry(snap)
    end
    BASE-->>MAIN: total_kwh_baseline recorded

    Note over MAIN: Phase 2 — Agent Loop
    MAIN->>AGENT_SIM: run(mode="agent", on_step=callback)
    loop Every 15-min step
        AGENT_SIM->>STATE: push_telemetry(snap)
        AGENT_SIM->>AGENT_SIM: needs_supervisory()?

        alt Supervisory step (every 2 hours or comfort alert)
            AGENT_SIM->>AGENT: decide(snap)
            AGENT->>STATE: get_summary()
            STATE-->>AGENT: context JSON
            AGENT->>LLM: prompt(telemetry + last_error)
            LLM-->>AGENT: JSON setpoints
            AGENT->>SAFETY: validate(command)
            SAFETY-->>AGENT: validated command
            AGENT->>AGENT_SIM: apply_command(cool, heat)
            AGENT->>STATE: append_diagnostic(msg)
            AGENT->>CSV: export_csv()
        end
    end

    Note over DASH: Continuous (every 5s)
    DASH->>CSV: read telemetry_log.csv
    DASH->>DASH: render charts + KPIs
```

---

## Configuration Reference

All parameters in `config/settings.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `llm` | `model` | `gemma4:e2b` | Ollama model name |
| `llm` | `temperature` | `0.1` | Low for deterministic outputs |
| `llm` | `max_tokens` | `2048` | Max response length |
| `llm` | `timeout_seconds` | `300` | LLM call timeout |
| `simulation` | `timestep_minutes` | `15` | Simulation granularity |
| `simulation` | `run_period_days` | `1` | Simulation duration |
| `simulation` | `supervisory_interval_minutes` | `120` | LLM decision frequency |
| `simulation` | `demo_mode` | `false` | Use surrogate vs EnergyPlus |
| `setpoints` | `min_deadband` | `1.5` | Min cool−heat gap |
| `setpoints` | `max_ramp_c_per_hour` | `1.0` | Max temperature change rate |
| `setpoints` | `min_dwell_minutes` | `30` | Min time between changes |
| `objective_weights` | `alpha_cost` | `1.0` | Energy cost weight |
| `objective_weights` | `beta_carbon` | `0.8` | Carbon emission weight |
| `objective_weights` | `gamma_comfort` | `2.5` | Occupant comfort weight |

---

## Deployment Modes

```mermaid
graph LR
    subgraph Demo["🔬 Demo Mode (Current)"]
        D1["DemoBuildingSimulator"]
        D2["Gemma 4 E2B via Ollama"]
        D3["Streamlit Dashboard"]
        D1 --> D2 --> D3
    end

    subgraph Pilot["🏢 Pilot Mode"]
        P1["EnergyPlus + Real EPW"]
        P2["Gemma 4 E2B"]
        P3["Dashboard + MCP"]
        P1 --> P2 --> P3
    end

    subgraph Prod["🏙️ Production Mode"]
        PR1["BACnet/Modbus Gateway"]
        PR2["Gemma 4 / Larger LLM"]
        PR3["MCP Fleet Management"]
        PR1 --> PR2 --> PR3
    end

    Demo -->|"Add EPW + IDF"| Pilot
    Pilot -->|"Add BACnet bridge"| Prod

    style Demo fill:#1a3a5c,stroke:#60a5fa,stroke-width:2px,color:#fff
    style Pilot fill:#3a3a1a,stroke:#fbbf24,stroke-width:2px,color:#fff
    style Prod fill:#1a3a1a,stroke:#34d399,stroke-width:2px,color:#fff
```
