"""
PHASE 1: Single Agent with Tools
=================================
Concepts covered:
  - Creating an Agent with `google.adk.agents.Agent`
  - Writing tools as plain Python functions (auto-wrapped by ADK)
  - The Runner + InMemorySessionService pattern
  - Streaming events and getting the final response

The agent here can tell you the weather and current time for a few cities.
In a real app you'd call a live API; here we use mock data so you don't
need extra API keys to get started.
"""

import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from model_config import get_model

# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------
# Rules ADK enforces:
#   1. Docstring is REQUIRED — the LLM reads it to know when/how to call the tool.
#   2. All parameters must have type hints.
#   3. Return a dict. ADK wraps non-dict returns as {"result": value}.
# ---------------------------------------------------------------------------

WEATHER_DATA = {
    "new york":   {"condition": "Sunny",        "temp_c": 25, "humidity": "45%"},
    "london":     {"condition": "Cloudy",        "temp_c": 14, "humidity": "72%"},
    "tokyo":      {"condition": "Partly Cloudy", "temp_c": 18, "humidity": "60%"},
    "mumbai":     {"condition": "Hot & Humid",   "temp_c": 34, "humidity": "85%"},
    "new delhi":  {"condition": "Hazy",          "temp_c": 32, "humidity": "55%"},
}

CITY_TIMEZONES = {
    "new york":  "America/New_York",
    "london":    "Europe/London",
    "tokyo":     "Asia/Tokyo",
    "mumbai":    "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
}


def get_weather(city: str) -> dict:
    """Returns the current weather report for a given city.

    Use this tool when the user asks about weather, temperature, or
    conditions in a specific city.

    Args:
        city: The name of the city (e.g. 'New York', 'London', 'Tokyo').

    Returns:
        A dict with 'status' and either 'report' (on success) or
        'error_message' (on failure).
    """
    key = city.lower().strip()
    if key in WEATHER_DATA:
        d = WEATHER_DATA[key]
        temp_f = round(int(d["temp_c"]) * 9 / 5 + 32)
        report = (
            f"{d['condition']}, {d['temp_c']}°C / {temp_f}°F, "
            f"Humidity: {d['humidity']}"
        )
        return {"status": "success", "report": report}
    supported = ", ".join(WEATHER_DATA.keys())
    return {
        "status": "error",
        "error_message": (
            f"No weather data for '{city}'. "
            f"Supported cities: {supported}."
        ),
    }


def get_current_time(city: str) -> dict:
    """Returns the current local time for a given city.

    Use this tool when the user asks what time it is in a specific city.

    Args:
        city: The name of the city (e.g. 'New York', 'London', 'Tokyo').

    Returns:
        A dict with 'status' and either 'report' (on success) or
        'error_message' (on failure).
    """
    key = city.lower().strip()
    if key in CITY_TIMEZONES:
        tz = ZoneInfo(CITY_TIMEZONES[key])
        now = datetime.datetime.now(tz)
        report = now.strftime("%A, %B %d %Y  %I:%M %p %Z")
        return {"status": "success", "report": report}
    supported = ", ".join(CITY_TIMEZONES.keys())
    return {
        "status": "error",
        "error_message": (
            f"No timezone info for '{city}'. "
            f"Supported cities: {supported}."
        ),
    }


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------
# `model`       — which Gemini model to use. "gemini-2.0-flash" is fast & cheap.
# `instruction` — the system prompt. Tells the agent its persona and rules.
# `tools`       — list of Python functions. ADK wraps them automatically.
# `description` — used by parent agents (in multi-agent setups) to decide
#                 when to delegate here. Not needed for a standalone agent,
#                 but good habit to always set it.
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="weather_time_agent",
    model=get_model(),
    description="Answers questions about weather and local time for major cities.",
    instruction=(
        "You are WeatherBot, a friendly assistant specialising in weather "
        "and time information.\n\n"
        "Guidelines:\n"
        "- Always use get_weather or get_current_time tools when asked; "
        "  never make up data.\n"
        "- If the user asks for both weather and time, call both tools.\n"
        "- If a city is not supported, tell the user which cities are available.\n"
        "- Keep responses concise and conversational.\n"
        "- Supported cities: New York, London, Tokyo, Mumbai, New Delhi."
    ),
    tools=[get_weather, get_current_time],
)
