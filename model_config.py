"""
model_config.py — Central model provider switch
================================================
All agent files import `get_model()` from here.
To change providers, just set MODEL_PROVIDER in your .env file.

Supported values:
  MODEL_PROVIDER=anthropic   → Claude via Anthropic API (LiteLLM)
  MODEL_PROVIDER=ollama      → Local model via Ollama (LiteLLM)
  MODEL_PROVIDER=google      → Gemini via Google AI Studio (native ADK)

HOW LITELLM WORKS IN ADK:
  Instead of passing a model string like "gemini-2.0-flash", you pass a
  LiteLlm(model="provider/model-name") object. ADK treats it identically —
  same tools, same sessions, same runner. LiteLLM translates the API calls.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Read .env
PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower().strip()


def get_model():
    """
    Returns the right model object for the configured provider.
    Pass the return value directly to Agent(model=get_model()).
    """
    if PROVIDER == "anthropic":
        return _anthropic_model()
    elif PROVIDER == "ollama":
        return _ollama_model()
    else:
        return _google_model()


def get_provider_name() -> str:
    return PROVIDER


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _anthropic_model():
    """
    Claude via Anthropic API using LiteLLM.
    Requires: ANTHROPIC_API_KEY in .env
    """
    from google.adk.models.lite_llm import LiteLlm

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Get yours at https://console.anthropic.com/ and add it to .env"
        )

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # LiteLLM needs ANTHROPIC_API_KEY in the environment (it reads it directly)
    os.environ["ANTHROPIC_API_KEY"] = api_key

    return LiteLlm(model=f"anthropic/{model_name}")


def _ollama_model():
    """
    Local model via Ollama using LiteLLM.
    Requires: Ollama running at OLLAMA_API_BASE (default: http://localhost:11434)
    Use 'ollama_chat/' prefix — NOT 'ollama/' — to avoid tool call loop bugs.
    """
    from google.adk.models.lite_llm import LiteLlm

    base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")

    os.environ["OLLAMA_API_BASE"] = base_url

    # IMPORTANT: use ollama_chat/ not ollama/ — the plain ollama/ provider
    # ignores chat history and causes infinite tool call loops.
    return LiteLlm(model=f"ollama_chat/{model_name}")


def _google_model():
    """
    Gemini via Google AI Studio — ADK's native model, no LiteLLM needed.
    Requires: GOOGLE_API_KEY in .env
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Get yours at https://aistudio.google.com/app/apikey and add it to .env"
        )
    # For Google, ADK accepts a plain model name string (no LiteLLM wrapper)
    return "gemini-2.0-flash"
