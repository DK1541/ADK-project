"""
ADK Project — Interactive Runner
==================================
Usage:
  python main.py --phase 1    # Phase 1: single agent (weather + time)
  python main.py --phase 2    # Phase 2: stateful agent (coming soon)
  python main.py --phase 3    # Phase 3: multi-agent system (coming soon)

The runner keeps a session alive so the agent remembers your conversation
history within the same run.
"""

import asyncio
import argparse
import os
from dotenv import load_dotenv

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load .env before importing agents (they need GOOGLE_API_KEY at import time)
load_dotenv()


# ---------------------------------------------------------------------------
# Core chat loop
# ---------------------------------------------------------------------------

async def chat_loop(root_agent, app_name: str):
    """
    Keeps a single session alive and lets you have a multi-turn conversation.

    HOW SESSIONS WORK:
    - app_name  : logical name for your application (any string)
    - user_id   : identifies the user (could be a DB id in production)
    - session_id: identifies one conversation thread

    InMemorySessionService stores all of this in RAM. When you restart the
    script the history is gone. For persistent history you'd swap in
    DatabaseSessionService or VertexAiSessionService.
    """
    session_service = InMemorySessionService()

    # Create a session (one conversation thread)
    session = await session_service.create_session(
        app_name=app_name,
        user_id="learner",
        session_id="demo-session",
    )

    # Runner wires the agent to the session service
    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
    )

    print(f"\n{'='*60}")
    print(f"  Agent: {root_agent.name}")
    print(f"  Type 'exit' or 'quit' to stop.")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Wrap the user's text in the format ADK expects
        message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        # run_async is a streaming generator.
        # Events flow in this order:
        #   1. LLM starts generating (text chunks)
        #   2. Tool call event (if the agent decides to use a tool)
        #   3. Tool response event (result comes back)
        #   4. LLM generates final text
        #   5. is_final_response() == True  ← this is what we print
        #
        # You could process intermediate events for a streaming UI,
        # but for the terminal we just wait for the final answer.

        print("Agent: ", end="", flush=True)
        async for event in runner.run_async(
            user_id="learner",
            session_id="demo-session",
            new_message=message,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    print(event.content.parts[0].text)
                else:
                    print("[no text response]")
        print()  # blank line between turns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ADK Learning Project")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Which phase to run (1=single agent, 2=stateful, 3=multi-agent)",
    )
    args = parser.parse_args()

    if args.phase == 1:
        from phase1_single_agent.agent import root_agent
        asyncio.run(chat_loop(root_agent, app_name="phase1"))

    elif args.phase == 2:
        try:
            from phase2_stateful.agent import root_agent
            asyncio.run(chat_loop(root_agent, app_name="phase2"))
        except ImportError:
            print("Phase 2 not built yet — coming soon!")

    elif args.phase == 3:
        try:
            from phase3_multi_agent.agent import root_agent
            asyncio.run(chat_loop(root_agent, app_name="phase3"))
        except ImportError:
            print("Phase 3 not built yet — coming soon!")


if __name__ == "__main__":
    main()
