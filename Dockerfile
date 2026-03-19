# ============================================================
# Dockerfile — ADK Assistant Server
# ============================================================
# Build:  docker build -t adk-assistant .
# Run:    docker run -p 8000:8000 --env-file .env adk-assistant
#
# ENV vars injected at runtime via --env-file .env (never bake keys into the image):
#   MODEL_PROVIDER    = anthropic | ollama | huggingface | google
#   ANTHROPIC_API_KEY = sk-ant-...   (if using anthropic)
#   GOOGLE_API_KEY    = ...          (if using google)
#   HF_TOKEN          = hf_...       (if using huggingface)
#   OLLAMA_MODEL      = llama3.2     (if using ollama — also needs Ollama running separately)
# ============================================================

FROM python:3.11-slim

# Keeps Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (separate layer — cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY model_config.py .
COPY shared_tools.py .
COPY server.py .
COPY agents/ ./agents/
COPY frontend/ ./frontend/

ENV PORT=8000

# Health check — Docker marks container unhealthy if /list-apps fails
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/list-apps')" || exit 1

CMD python server.py --port $PORT --host 0.0.0.0
