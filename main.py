"""
main.py — Interactive terminal runner
======================================
Usage:
  python main.py

Starts a conversation with the multi-agent assistant.
"""

import asyncio
import os
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()


async def chat_loop(root_agent):
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="assistant", user_id="user", session_id="session"
    )
    runner = Runner(agent=root_agent, app_name="assistant", session_service=session_service)

    provider = os.getenv("MODEL_PROVIDER", "google")
    print(f"\n{'='*55}")
    print(f"  ADK Assistant  |  Provider: {provider}")
    print(f"  7 specialists: weather·time·travel·math·language·code·knowledge·media")
    print(f"  Type 'exit' to quit.")
    print(f"{'='*55}\n")

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

        message = types.Content(role="user", parts=[types.Part(text=user_input)])
        print("Assistant: ", end="", flush=True)
        async for event in runner.run_async(
            user_id="user", session_id="session", new_message=message
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    print(event.content.parts[0].text)
                else:
                    print("[no response]")
        print()


def main():
    from agents.agent import root_agent
    asyncio.run(chat_loop(root_agent))


if __name__ == "__main__":
    main()
