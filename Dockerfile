# ============================================================
# Dockerfile — ADK Agent Server
# ============================================================
# Build:  docker build -t adk-agent .
# Run:    docker run -p 8000:8000 --env-file .env adk-agent
#
# ENV vars injected at runtime via --env-file .env (never bake keys into the image):
#   MODEL_PROVIDER    = anthropic | ollama | google
#   ANTHROPIC_API_KEY = sk-ant-...
#   PHASE             = 1 | 2 | 3  (default: 3)
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
COPY server.py .
COPY phase1_single_agent/ ./phase1_single_agent/
COPY phase2_stateful/     ./phase2_stateful/
COPY phase3_multi_agent/  ./phase3_multi_agent/

# Default phase to run (override with -e PHASE=1 at runtime)
ENV PHASE=3
ENV PORT=8000

# Health check — Docker marks container unhealthy if /list-apps fails
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/list-apps')" || exit 1

# Use shell form so $PHASE and $PORT are expanded at runtime
CMD python server.py --phase $PHASE --port $PORT --host 0.0.0.0
