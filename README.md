# ADK Assistant — Multi-Agent System with Google's Agent Development Kit

A production-style multi-agent assistant built with [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/). One coordinator routes requests across six specialist agents, each focused on a different domain. Supports **Google Gemini**, **Anthropic Claude**, **Ollama (local)**, and **HuggingFace** as LLM backends — switchable via a single environment variable.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │       assistant              │
                    │      (coordinator)           │
                    └──────────────┬───────────────┘
                                   │ delegates via sub_agents
        ┌──────────┬───────────────┼──────────────┬──────────┬──────────────┐
        ▼          ▼               ▼              ▼          ▼              ▼
  ┌──────────┐ ┌────────┐ ┌──────────────┐ ┌──────────┐ ┌──────┐ ┌──────────────┐
  │ weather  │ │ travel │ │ math/science │ │ language │ │ code │ │  knowledge   │
  │  & time  │ │ advisor│ │  calculator  │ │ writing  │ │ agent│ │  & research  │
  └──────────┘ └────────┘ └──────────────┘ └──────────┘ └──────┘ └──────────────┘
```

**Specialists and their tools:**

| Agent | Capabilities |
|---|---|
| Weather & Time | Real-time weather, local time, all-city comparisons, best month recommendations |
| Travel Advisor | City info, attractions, travel tips, packing advice |
| Math & Science | Calculator (safe AST eval), unit conversions (length/weight/temp/speed/data) |
| Language & Writing | Translation, grammar, writing, summarisation, text analysis |
| Code | Code explanation, debugging, generation, review across languages |
| Knowledge | General Q&A, history, science, facts, reasoning |

---

## Project Structure

```
ADK-project/
├── agents/
│   ├── agent.py           # All 6 specialists + coordinator root_agent
│   ├── assistant/
│   │   └── agent.py       # Re-exports root_agent (ADK loader entry point)
│   └── __init__.py
├── shared_tools.py        # Weather/time data and tools (single source of truth)
├── model_config.py        # LLM provider switcher
├── main.py                # Interactive terminal chat
├── server.py              # FastAPI REST API server + chat frontend
├── frontend/
│   └── index.html         # Dark-themed chat UI with history & file upload
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

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

Create a `.env` file (copy from below — never commit it):

```bash
# Choose one: anthropic | ollama | huggingface | google
MODEL_PROVIDER=ollama

# Anthropic / Claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# Ollama — local, no API key needed (https://ollama.com)
OLLAMA_MODEL=llama3.2
# Per-specialist overrides (optional):
# OLLAMA_MODEL_WEATHER=qwen2.5:14b
# OLLAMA_MODEL_TRAVEL=llama3.1
# OLLAMA_MODEL_MATH=llama3.2
# OLLAMA_MODEL_LANGUAGE=llama3.2
# OLLAMA_MODEL_CODE=qwen2.5:14b
# OLLAMA_MODEL_KNOWLEDGE=llama3.1

# HuggingFace (via Sambanova router — best for tool use)
# HF_TOKEN=hf_...
# HF_MODEL=huggingface/sambanova/Qwen/Qwen2.5-72B-Instruct

# Google Gemini
# GOOGLE_API_KEY=...
```

---

## Running

### Browser chat UI
```bash
python server.py
```
Open `http://localhost:8000` — full chat interface with history, file/image upload.

### Terminal chat
```bash
python main.py
```

### ADK Dev UI (visualises tool calls and event stream)
```bash
adk web agents
```

---

## API

Swagger docs at `http://localhost:8000/docs`

**Example request:**
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "appName": "assistant",
    "userId": "user_001",
    "sessionId": "sess_001",
    "newMessage": {
      "role": "user",
      "parts": [{ "text": "What is the weather in Tokyo right now?" }]
    }
  }'
```

---

## Switching Models

Edit `MODEL_PROVIDER` in your `.env` — no code changes needed:

```
MODEL_PROVIDER=anthropic    →  Claude via Anthropic API
MODEL_PROVIDER=ollama       →  Local model via Ollama
MODEL_PROVIDER=huggingface  →  HuggingFace Inference API
MODEL_PROVIDER=google       →  Gemini via Google AI Studio
```

For Ollama you can assign a different model to each specialist via `OLLAMA_MODEL_<SPECIALIST>` vars, letting you mix and match models based on task type.

---

## Deployment

### Docker
```bash
docker build -t adk-assistant .
docker run -p 8000:8000 --env-file .env adk-assistant
```

### Cloud

| Platform | Notes |
|---|---|
| Railway / Render | Connect GitHub repo, add env vars in dashboard |
| AWS ECS | Push to ECR, create ECS service |
| GCP Cloud Run | `gcloud run deploy` or `adk deploy cloud_run` |
| Azure Container Apps | `az containerapp create` |

> Vertex AI Agent Engine requires Gemini models. For Claude or Ollama, use the Docker path.

---

## Key ADK Concepts

| Concept | Where to see it |
|---|---|
| Functions as tools (docstring + type hints) | `shared_tools.py`, `agents/agent.py` |
| `Agent(model=, instruction=, tools=)` | `agents/agent.py` |
| `sub_agents` delegation | `agents/agent.py` — coordinator |
| LiteLLM bridge (Claude / Ollama / HuggingFace) | `model_config.py` |
| Per-specialist model routing via env vars | `agents/agent.py` — `_model()` helper |
| FastAPI REST server | `server.py` |
| SQLite persistent sessions | `server.py` — `sqlite+aiosqlite:///sessions.db` |
| Chat history + file upload frontend | `frontend/index.html` |
