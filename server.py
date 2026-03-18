"""
server.py — Production REST API server for all ADK phases
==========================================================
Wraps any agent directory into a FastAPI app with full HTTP endpoints,
and serves the chat frontend at http://localhost:8000/

Run:
  python server.py --phase 1        # single agent
  python server.py --phase 3        # multi-agent
  python server.py --phase 3 --ui   # also serve ADK Dev UI at /dev-ui

Endpoints (Swagger docs at http://localhost:8000/docs):
  GET  /                                                   → chat frontend
  GET  /list-apps                                          → list agents
  POST /apps/{app}/users/{uid}/sessions/{sid}              → create session
  POST /run                                                → single response
  POST /run_sse                                            → streaming (SSE)
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

# Map phase numbers to their directory names
PHASE_DIRS = {
    1: "phase1_single_agent",
    2: "phase2_stateful",
    3: "phase3_multi_agent",
}


def create_app(phase: int, serve_ui: bool = False):
    """
    Builds the FastAPI app for the given phase.

    get_fast_api_app() scans the agent directory for a module that exposes
    a `root_agent` variable — the same convention used by `adk web`.

    Parameters:
      agents_dir         — path to the folder containing the agent module
      session_service_uri— where sessions are stored:
                             None / "" → in-memory (lost on restart)
                             "sqlite+aiosqlite:///sessions.db" → SQLite file
      allow_origins      — CORS origins (use specific domains in production)
      web                — True to also serve the Dev UI at /
    """
    from google.adk.cli.fast_api import get_fast_api_app

    project_root = Path(__file__).parent
    agent_dir = project_root / PHASE_DIRS[phase]

    if not agent_dir.exists():
        print(f"ERROR: {agent_dir} does not exist.")
        sys.exit(1)

    # Add project root to path so agent modules can import model_config
    sys.path.insert(0, str(project_root))

    # SQLite for persistent sessions across restarts.
    # Change to "" for in-memory (simpler, sessions lost on restart).
    session_db = f"sqlite+aiosqlite:///sessions_phase{phase}.db"

    app = get_fast_api_app(
        agents_dir=str(agent_dir),
        session_service_uri=session_db,
        allow_origins=["*"],   # restrict to your domain in production
        web=serve_ui,
    )

    # Serve the chat frontend at /chat and its static assets
    frontend_dir = project_root / "frontend"
    if frontend_dir.exists():
        app.mount("/chat", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

        @app.get("/", include_in_schema=False)
        async def root():
            return FileResponse(str(frontend_dir / "index.html"))

    return app


def main():
    parser = argparse.ArgumentParser(description="ADK REST API Server")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--ui", action="store_true", help="Also serve Dev UI")
    args = parser.parse_args()

    app = create_app(phase=args.phase, serve_ui=args.ui)

    phase_dir = PHASE_DIRS[args.phase]
    print(f"\n{'='*55}")
    print(f"  ADK Agent Server — {phase_dir}")
    print(f"  Provider  : {os.getenv('MODEL_PROVIDER', 'google')}")
    print(f"  Chat UI   : http://localhost:{args.port}/")
    print(f"  API docs  : http://localhost:{args.port}/docs")
    if args.ui:
        print(f"  ADK UI    : http://localhost:{args.port}/dev-ui")
    print(f"{'='*55}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
