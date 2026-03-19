"""
server.py — REST API server for the ADK multi-agent assistant
=============================================================
Run:
  python server.py
  python server.py --port 8080
  python server.py --ui       # also serve ADK Dev UI at /dev-ui

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


def create_app(serve_ui: bool = False):
    """
    Builds the FastAPI app for the agents/ directory.

    get_fast_api_app() scans the agent directory for a module that exposes
    a `root_agent` variable — the same convention used by `adk web`.
    """
    from google.adk.cli.fast_api import get_fast_api_app

    project_root = Path(__file__).parent
    agent_dir = project_root / "agents"

    if not agent_dir.exists():
        print(f"ERROR: {agent_dir} does not exist.")
        sys.exit(1)

    # Add project root to path so agent modules can import model_config / shared_tools
    sys.path.insert(0, str(project_root))

    # SQLite for persistent sessions across restarts.
    session_db = "sqlite+aiosqlite:///sessions.db"

    app = get_fast_api_app(
        agents_dir=str(agent_dir),
        session_service_uri=session_db,
        allow_origins=["*"],   # restrict to your domain in production
        web=serve_ui,
    )

    # Serve the chat frontend at /
    frontend_dir = project_root / "frontend"
    if frontend_dir.exists():
        app.mount("/chat", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

        @app.get("/", include_in_schema=False)
        async def root():
            return FileResponse(str(frontend_dir / "index.html"))

    return app


def main():
    parser = argparse.ArgumentParser(description="ADK Multi-Agent REST API Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--ui", action="store_true", help="Also serve ADK Dev UI at /dev-ui")
    args = parser.parse_args()

    app = create_app(serve_ui=args.ui)

    print(f"\n{'='*55}")
    print(f"  ADK Assistant — 6 Specialist Agents")
    print(f"  Provider  : {os.getenv('MODEL_PROVIDER', 'google')}")
    print(f"  Chat UI   : http://localhost:{args.port}/")
    print(f"  API docs  : http://localhost:{args.port}/docs")
    if args.ui:
        print(f"  ADK UI    : http://localhost:{args.port}/dev-ui")
    print(f"{'='*55}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
