# ADK Assistant — Multi-Agent System with Google's Agent Development Kit

A production-style multi-agent assistant built with [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/). One coordinator routes requests across seven specialist agents, each focused on a different domain. Supports **Google Gemini**, **Anthropic Claude**, **Ollama (local)**, and **HuggingFace** as LLM backends — switchable via a single environment variable.

---

## Architecture

```
                         ┌──────────────────────┐
                         │      assistant       │
                         │     (coordinator)    │
                         └──────────┬───────────┘
                                    │ delegates via sub_agents
     ┌──────────┬──────────┬────────┼──────────┬──────────┬──────────┐
     ▼          ▼          ▼        ▼          ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌───────┐ ┌────────┐ ┌──────┐ ┌─────────┐ ┌───────┐
│ weather │ │ travel │ │ math/ │ │  lang  │ │ code │ │knowledge│ │ media │
│  & time │ │advisor │ │science│ │writing │ │      │ │research │ │editor │
└─────────┘ └────────┘ └───────┘ └────────┘ └──────┘ └─────────┘ └───────┘
```

**Specialists and their tools:**

| Agent | Capabilities |
|---|---|
| Weather & Time | Live weather, UV index, local time, world clock, time-zone differences, city comparisons |
| Travel Advisor | Attractions, best visiting seasons, packing lists, city facts (currency, language, population) |
| Math & Science | Safe AST-evaluated calculator, unit conversions (length/weight/temp/speed/volume/data/area) |
| Language & Writing | Translation, grammar correction, tone rewriting, summarisation, creative writing, emails |
| Code | Write, explain, debug, and review code in any language; SQL, regex, shell, architecture |
| Knowledge | History, science, geography, culture, philosophy, how-things-work, research synthesis |
| Media | Convert photos to video slideshows, edit images (resize/crop/rotate/brightness/text overlay), edit videos (trim/speed/reverse/merge), extract frames, read EXIF/video metadata, **YOLOv8 object detection** (detect/label/count objects in images and videos) |

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.11+** — check with `python --version`
  - Download: https://www.python.org/downloads/
- **Git** — to clone the repo
- **One of the following LLM backends** (pick whichever fits you):

| Provider | What you need |
|---|---|
| **Ollama** (local, free) | Install Ollama from https://ollama.com, then pull models (see below) |
| **Google Gemini** | A free API key from https://aistudio.google.com/app/apikey |
| **Anthropic Claude** | An API key from https://console.anthropic.com |
| **HuggingFace** | A free token from https://huggingface.co/settings/tokens |

### Ollama — pull models before first run

If you're using Ollama, start the Ollama app, then pull the models you plan to use:

```bash
# Minimum — one model for everything
ollama pull llama3.2

# Recommended — best model per task (requires ~40 GB RAM total)
ollama pull llama3.2          # weather, time
ollama pull llama3.1          # time specialist
ollama pull qwen2.5:14b       # coordinator, travel, math
ollama pull qwen2.5-coder:14b # code specialist
ollama pull phi4              # knowledge / research
ollama pull minicpm-v         # media specialist (vision + tool calling) — code default
ollama pull aya:8b            # language / translation (23+ languages) — code default
ollama pull aya-expanse:32b   # language upgrade — best translation quality (20 GB, optional)
ollama pull llava:13b         # media upgrade — better image understanding (optional)
```

> **Smart defaults:** Even without per-specialist env vars, the language agent automatically
> uses `aya:8b` (not the generic fallback) and the media agent uses `minicpm-v`.
> Set `OLLAMA_MODEL_LANGUAGE` or `OLLAMA_MODEL_MEDIA` in `.env` to override.

> **Tip:** If you have very limited RAM, set every specialist to the same small model:
> set `OLLAMA_MODEL=llama3.2` in `.env` and remove all per-specialist overrides.
> Note: translation quality will drop significantly with `llama3.2`.

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/DK1541/ADK-project.git
cd ADK-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your environment

```bash
cp .env.example .env
```

Then open `.env` and:
- Set `MODEL_PROVIDER` to your chosen backend (`ollama`, `google`, `anthropic`, or `huggingface`)
- Fill in the API key for that provider (or leave blank if using Ollama)
- Optionally assign different models per specialist via `OLLAMA_MODEL_<SPECIALIST>`

**Minimal `.env` for Ollama (local, no API key needed):**
```env
MODEL_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

**Minimal `.env` for Google Gemini:**
```env
MODEL_PROVIDER=google
GOOGLE_API_KEY=your_key_here
```

**Minimal `.env` for Anthropic Claude:**
```env
MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

### 5. Run

```bash
python server.py
```

Open **http://localhost:8000** — the chat UI loads in your browser.

---

## Project Structure

```
ADK-project/
├── agents/
│   ├── __init__.py            # Marks agents/ as a Python package
│   ├── agent.py               # All 7 specialists + coordinator (root_agent)
│   └── assistant/
│       ├── __init__.py        # Marks assistant/ as a Python package
│       └── agent.py           # ADK loader entry point — re-exports root_agent
│
├── shared_tools.py            # Weather/time/city data + shared tool functions (37 cities)
├── media_tools.py             # Image & video processing tools (Pillow, moviepy, OpenCV, YOLOv8)
├── model_config.py            # LLM provider switcher (Gemini / Claude / Ollama / HuggingFace)
├── server.py                  # FastAPI REST server + chat frontend mount
├── main.py                    # Interactive terminal chat loop
│
├── frontend/
│   └── index.html             # Dark-themed chat UI (history, file upload, drag-and-drop)
│
├── Dockerfile                 # Container image — python:3.11-slim, runs server.py
├── docker-compose.yml         # Single-service Compose config, port 8000
├── requirements.txt           # Python dependencies with minimum version pins
├── .env.example               # Template for all environment variables
├── .python-version            # Specifies Python 3.11 (used by pyenv / mise)
├── pyrightconfig.json         # Pylance / Pyright config for VS Code
└── .gitignore                 # Excludes .env, __pycache__, *.db, .venv, postman/
```

### File descriptions

**`agents/agent.py`**
The heart of the system. Defines all seven specialist `Agent` objects and the `root_agent` coordinator. Each specialist has its own model (configurable per-specialist via env vars using the `_model(env_key)` helper), a focused `instruction` prompt, and a precise `description` the coordinator uses for routing. Specialist tools defined here: `get_weather_detailed`, `get_time_detailed`, `get_time_difference`, `get_travel_tips`, `get_packing_advice`, `calculate`, `convert_units`. Media tools are imported from `media_tools.py`, shared tools from `shared_tools.py`.

**`agents/assistant/agent.py`**
A one-line re-export: `from agents.agent import root_agent`. Required because ADK's loader expects the structure `<agents_dir>/<app_name>/agent.py`. The app name is `assistant`, so this file is the loader's entry point.

**`shared_tools.py`**
Single source of truth for all city data and shared tool functions. Contains `WEATHER_DATA`, `CITY_TIMEZONES`, and `CITY_INFO` for 37 cities across 9 regions (North America, South America, Europe, Africa, Middle East, South Asia, East Asia, Southeast Asia, Oceania). Exports tools used by multiple specialists: `get_weather`, `get_current_time`, `get_all_times`, `get_all_weather`, `compare_weather`, `get_city_info`, `get_best_cities_for_month`.

**`media_tools.py`**
All image and video processing tools for the media specialist. Image tools: `get_image_info` (EXIF, dimensions, format), `edit_image` (resize, crop, rotate, brightness, contrast, saturation, grayscale, flip, format conversion), `add_text_to_image` (captions/watermarks). Video tools: `photos_to_video` (slideshow creator), `get_video_info` (duration, fps, resolution, codec), `edit_video` (trim, speed, reverse, mute), `extract_video_frames` (save frames at intervals), `merge_videos` (concatenate clips). YOLOv8 tools: `detect_objects_in_image`, `count_objects`, `detect_objects_in_video`.

**`model_config.py`**
Reads `MODEL_PROVIDER` from `.env` and returns the correct model object for ADK. Google Gemini returns a plain model name string (`"gemini-2.0-flash"`); Claude, Ollama, and HuggingFace return a `LiteLlm(model=...)` wrapper. Validates that API keys are present and raises a clear `ValueError` with a signup link if they are missing. `agents/agent.py` extends this with a `_model(env_key)` helper for per-specialist model overrides.

**`server.py`**
Builds the FastAPI app using ADK's `get_fast_api_app()`, pointing it at the `agents/` directory with a SQLite session database (`sessions.db`) for persistence across restarts. Mounts `frontend/index.html` at `/` so the chat UI and API are served from the same port. Accepts `--port` (default 8000), `--host`, and `--ui` (enables ADK Dev UI at `/dev-ui`) flags.

**`main.py`**
Minimal terminal chat loop using ADK's `Runner` + `InMemorySessionService`. Imports `root_agent`, creates a single in-memory session, and prints streamed responses to stdout. Useful for quick testing without starting the web server. Sessions are not persisted between runs.

**`frontend/index.html`**
Self-contained single-page chat UI with no build step required. Features: dark theme, collapsible sidebar with persistent chat history (localStorage + server session resume on click), suggestion chips, animated typing indicator, file and image upload (base64-encoded, sent as `inlineData` parts), and drag-and-drop onto the input bar.

**`Dockerfile`**
Builds a `python:3.11-slim` image. Installs dependencies in a separate layer (cached unless `requirements.txt` changes), copies source files, and runs `server.py`. Includes a Docker health check hitting `/list-apps`. All secrets injected at runtime via `--env-file .env` — nothing sensitive is baked in.

**`docker-compose.yml`**
Single-service Compose config that builds the image and maps port 8000. Mounts a `./data` volume so the SQLite session database (`sessions.db`) survives container restarts. Set to `restart: unless-stopped`.

**`.env.example`**
Template showing every supported environment variable with comments. Copy to `.env` and fill in your values. Covers all four providers (Ollama, Google, Anthropic, HuggingFace) and all per-specialist model override keys.

**`requirements.txt`**
Direct dependencies only:
- `google-adk` — Agent Development Kit (runner, sessions, FastAPI wrapper, LiteLLM bridge)
- `litellm` — unified API layer for Claude, Ollama, and HuggingFace
- `fastapi` — web framework used by ADK's server
- `uvicorn` — ASGI server
- `python-dotenv` — `.env` file loader
- `aiosqlite` — async SQLite driver for persistent sessions
- `Pillow` — image reading, editing, and format conversion
- `moviepy` — video creation, editing, and concatenation
- `opencv-python` — video frame extraction and metadata
- `ultralytics` — YOLOv8 object detection for images and videos

**`pyrightconfig.json`**
Tells Pylance/Pyright (VS Code) to use Python 3.11, look in `.venv` for the interpreter, and add the project root to `extraPaths` so imports like `from shared_tools import ...` resolve without errors.

---

## Supported Cities

**North America:** New York · Los Angeles · Chicago · Toronto · Mexico City · Milwaukee

**South America:** São Paulo · Buenos Aires

**Europe:** London · Paris · Berlin · Rome · Barcelona · Amsterdam · Vienna · Zurich · Stockholm · Istanbul · Moscow

**Africa:** Cairo · Lagos · Nairobi · Johannesburg

**Middle East:** Dubai

**South Asia:** Mumbai · New Delhi

**East Asia:** Tokyo · Beijing · Shanghai · Seoul · Hong Kong

**Southeast Asia:** Bangkok · Singapore · Kuala Lumpur · Jakarta · Manila

**Oceania:** Sydney

---

## Other Ways to Run

### Terminal chat (no browser needed)
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

Edit `MODEL_PROVIDER` in `.env` — no code changes needed:

```
MODEL_PROVIDER=anthropic    →  Claude via Anthropic API
MODEL_PROVIDER=ollama       →  Local model via Ollama
MODEL_PROVIDER=huggingface  →  HuggingFace Inference API
MODEL_PROVIDER=google       →  Gemini via Google AI Studio
```

For Ollama, assign a different model to each specialist via `OLLAMA_MODEL_<SPECIALIST>` to mix and match based on task type.

---

## Deployment

### Docker
```bash
docker build -t adk-assistant .
docker run -p 8000:8000 --env-file .env adk-assistant
```

### Docker Compose
```bash
docker compose up
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

## Troubleshooting

**Agent responds in the wrong language**
The coordinator and all specialists are instructed to match the user's language. If a model keeps drifting (common with Qwen-based models), switch `OLLAMA_MODEL_COORDINATOR` to a more English-dominant model like `llama3.2` or `phi4`.

**"Cannot reach server" in the chat UI**
Make sure `server.py` is running (`python server.py`) and the browser is pointing at `http://localhost:8000`.

**Ollama errors / empty responses**
- Confirm Ollama is running: `ollama list`
- Confirm the model you set in `.env` is pulled: `ollama pull <model-name>`
- Check the terminal running `server.py` for Python tracebacks

**`[No text response — check server logs]` in the UI**
The agent pipeline ran but produced no text. Check the `server.py` terminal for model errors. Usually caused by a model that doesn't support tool calling (required by ADK) — try switching to `llama3.2` or `qwen2.5:14b`.

---

## Key ADK Concepts

| Concept | Where to see it |
|---|---|
| Functions as tools (docstring + type hints required) | `shared_tools.py`, `agents/agent.py`, `media_tools.py` |
| `Agent(model=, instruction=, tools=)` | `agents/agent.py` — each specialist |
| `sub_agents` delegation and routing | `agents/agent.py` — `root_agent` |
| LiteLLM bridge (Claude / Ollama / HuggingFace) | `model_config.py` |
| Per-specialist model routing via env vars | `agents/agent.py` — `_model()` helper |
| ADK loader directory convention | `agents/assistant/agent.py` |
| FastAPI REST server via `get_fast_api_app()` | `server.py` |
| SQLite persistent sessions | `server.py` — `sqlite+aiosqlite:///sessions.db` |
| Chat history + file upload frontend | `frontend/index.html` |
| Image/video processing (Pillow, moviepy, OpenCV) | `media_tools.py` |
| YOLOv8 object detection (Ultralytics) | `media_tools.py` — `detect_objects_in_image`, `count_objects`, `detect_objects_in_video` |
