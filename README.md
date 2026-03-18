# ADK Project — Multi-Agent System with Google's Agent Development Kit

A hands-on learning project built with [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/), progressing from a single agent through to a full multi-agent system. Supports **Google Gemini**, **Anthropic Claude**, and **Ollama (local)** as the LLM backend.

---

## Project Structure

```
ADK-project/
├── phase1_single_agent/     # Single agent: weather + time tools
├── phase2_stateful/         # Agent with session memory (ToolContext)
├── phase3_multi_agent/      # Coordinator + 3 specialist sub-agents
├── model_config.py          # Switch LLM provider in one place
├── main.py                  # Interactive terminal runner
├── server.py                # FastAPI REST API server
├── Dockerfile               # Container image
├── docker-compose.yml       # Run all 3 phases on separate ports
└── requirements.txt
```

---

## The Three Phases

### Phase 1 — Single Agent
A `WeatherBot` agent with two tools:
- `get_weather(city)` — returns condition, temperature, humidity
- `get_current_time(city)` — returns local time and timezone

**What you learn:** How ADK auto-wraps plain Python functions as tools using docstrings and type hints. How the Runner + Session pattern works.

---

### Phase 2 — Stateful Agent
Extends Phase 1 with session memory using `ToolContext`:
- `set_home_city` / `get_home_weather` — remembers your preferred city across turns
- `add_to_watchlist` / `get_watchlist_weather` — tracks a list of cities

**What you learn:** How `tool_context.state` persists data across conversation turns within a session. The difference between in-memory and persistent sessions.

---

### Phase 3 — Multi-Agent System

```
         ┌────────────────────────────┐
         │   travel_assistant         │
         │   (coordinator)            │
         └──────────┬─────────────────┘
                    │ delegates via sub_agents
       ┌────────────┼────────────┐
       ▼            ▼            ▼
┌────────────┐ ┌──────────┐ ┌──────────────┐
│  weather   │ │  time    │ │    travel    │
│ specialist │ │specialist│ │   advisor   │
└────────────┘ └──────────┘ └──────────────┘
```

Three specialist agents, each with their own tools, coordinated by a root agent:
- **Weather Specialist** — condition, temperature, UV index
- **Time Specialist** — local time, UTC offset, time difference between cities
- **Travel Advisor** — attractions, best seasons, packing advice

**What you learn:** How `sub_agents` delegation works. How the coordinator's LLM uses each specialist's `description` to route requests automatically. The `SequentialAgent` pipeline pattern.

---

## Supported Cities
New York · London · Tokyo · Mumbai · New Delhi · Milwaukee · Chicago 

---

## Setup

### 1. Clone and install
```bash
git clone https://github.com/DK1541/ADK-project.git
cd ADK-project
pip install -r requirements.txt
```

### 2. Configure your LLM provider

Copy `.env` and set your provider:

```bash
# Choose one: anthropic | ollama | google
MODEL_PROVIDER=anthropic

# Anthropic / Claude (https://console.anthropic.com/)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# Ollama — local, no API key needed (https://ollama.com)
# OLLAMA_API_BASE=http://localhost:11434
# OLLAMA_MODEL=llama3.2

# Google Gemini (https://aistudio.google.com/app/apikey)
# GOOGLE_API_KEY=...
```

---

## Running

### Terminal chat
```bash
python main.py --phase 1   # single agent
python main.py --phase 2   # stateful agent
python main.py --phase 3   # multi-agent
```

### Browser Dev UI (visualises tool calls and event stream)
```bash
adk web phase3_multi_agent
```

### REST API server
```bash
python server.py --phase 3 --port 8000
```
Swagger docs at `http://localhost:8000/docs`

**Example request:**
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "appName": "phase3_multi_agent",
    "userId": "user_001",
    "sessionId": "sess_001",
    "newMessage": {
      "role": "user",
      "parts": [{ "text": "What should I pack for a trip to Tokyo?" }]
    }
  }'
```

---

## Deployment

### Docker
```bash
# Build
docker build -t adk-agent .

# Run phase 3 with your .env
docker run -p 8000:8000 --env-file .env -e PHASE=3 adk-agent

# Run all phases simultaneously
docker compose up
# Phase 1 → :8001  |  Phase 2 → :8002  |  Phase 3 → :8003
```

### Cloud
Push the Docker image to any container platform:

| Platform | Notes |
|---|---|
| Railway / Render | Connect GitHub repo, add env vars in dashboard |
| AWS ECS | Push to ECR, create ECS service |
| GCP Cloud Run | `gcloud run deploy` or `adk deploy cloud_run` |
| Azure Container Apps | `az containerapp create` |

> Vertex AI Agent Engine requires Gemini models. For Claude or Ollama, use the Docker path above.

---

## Key ADK Concepts

| Concept | Where to see it |
|---|---|
| Functions as tools (docstring + type hints) | All phases |
| `Agent(model=, instruction=, tools=)` | `phase1_single_agent/agent.py` |
| Runner + InMemorySessionService | `main.py` |
| `ToolContext.state` for session memory | `phase2_stateful/agent.py` |
| `sub_agents` delegation | `phase3_multi_agent/agent.py` |
| LiteLLM bridge (Claude / Ollama) | `model_config.py` |
| FastAPI REST server | `server.py` |
| `SequentialAgent` pipeline pattern | `phase3_multi_agent/agent.py` (commented example) |

---

## Switching Models

Edit `MODEL_PROVIDER` in your `.env` — no code changes needed:

```
MODEL_PROVIDER=anthropic   →  Claude via Anthropic API
MODEL_PROVIDER=ollama      →  Local model via Ollama
MODEL_PROVIDER=google      →  Gemini via Google AI Studio
```
