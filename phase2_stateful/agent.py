"""
PHASE 2: Stateful Agent — Session Memory
==========================================
Concepts covered:
  - ToolContext: how tools read and write session state
  - Remembering user preferences across conversation turns
  - output_key: storing an agent's final reply into session state
    so other agents (or the same agent later) can reference it

New tools added here:
  - set_home_city      : saves the user's preferred city into session state
  - get_home_weather   : reads that preference and looks up weather for it
  - add_to_watchlist   : keeps a list of cities the user cares about
  - get_watchlist      : reads back that list

The agent from Phase 1 knew nothing about the user.
This agent remembers preferences for the whole session.
"""

import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from model_config import get_model

# ---------------------------------------------------------------------------
# Reuse Phase 1 weather/time data
# ---------------------------------------------------------------------------

WEATHER_DATA = {
    "new york":   {"condition": "Sunny",        "temp_c": 25, "humidity": "45%"},
    "london":     {"condition": "Cloudy",        "temp_c": 14, "humidity": "72%"},
    "tokyo":      {"condition": "Partly Cloudy", "temp_c": 18, "humidity": "60%"},
    "mumbai":     {"condition": "Hot & Humid",   "temp_c": 34, "humidity": "85%"},
    "new delhi":  {"condition": "Hazy",          "temp_c": 32, "humidity": "55%"},
    "chicago":    {"condition": "Windy",         "temp_c": 10, "humidity": "58%"},
    "milwaukee":  {"condition": "Overcast",      "temp_c": 8,  "humidity": "63%"},
}

CITY_TIMEZONES = {
    "new york":  "America/New_York",
    "london":    "Europe/London",
    "tokyo":     "Asia/Tokyo",
    "mumbai":    "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
    "chicago":   "America/Chicago",
    "milwaukee": "America/Chicago",
}


def get_weather(city: str) -> dict:
    """Returns the current weather report for a given city.

    Args:
        city: The name of the city (e.g. 'New York', 'London', 'Tokyo').
    """
    key = city.lower().strip()
    if key in WEATHER_DATA:
        d = WEATHER_DATA[key]
        temp_f = round(int(d["temp_c"]) * 9 / 5 + 32)
        return {
            "status": "success",
            "report": f"{d['condition']}, {d['temp_c']}°C / {temp_f}°F, Humidity: {d['humidity']}",
        }
    return {
        "status": "error",
        "error_message": f"No data for '{city}'. Supported: {', '.join(WEATHER_DATA)}.",
    }


def get_current_time(city: str) -> dict:
    """Returns the current local time for a given city.

    Args:
        city: The name of the city.
    """
    key = city.lower().strip()
    if key in CITY_TIMEZONES:
        tz = ZoneInfo(CITY_TIMEZONES[key])
        now = datetime.datetime.now(tz)
        return {"status": "success", "report": now.strftime("%A, %B %d %Y  %I:%M %p %Z")}
    return {
        "status": "error",
        "error_message": f"No timezone for '{city}'. Supported: {', '.join(CITY_TIMEZONES)}.",
    }


# ---------------------------------------------------------------------------
# STATEFUL TOOLS — these use ToolContext to read/write session state
# ---------------------------------------------------------------------------
# KEY CONCEPT:
#   When ADK sees a parameter named `tool_context: ToolContext` it injects
#   the context automatically — the LLM never sees this parameter.
#   tool_context.state is a dict that lives for the entire session.
# ---------------------------------------------------------------------------

def set_home_city(city: str, tool_context: ToolContext) -> dict:
    """Saves the user's home city so it can be used in future requests.

    Call this when the user tells you their city, home location,
    or default city for weather/time lookups.

    Args:
        city: The city name to save as the user's home city.
    """
    key = city.lower().strip()
    supported = list(WEATHER_DATA.keys())

    if key not in supported:
        return {
            "status": "error",
            "error_message": f"'{city}' is not supported. Choose from: {', '.join(supported)}.",
        }

    # Write to session state — persists for ALL future turns in this session
    tool_context.state["home_city"] = key
    return {"status": "success", "message": f"Home city set to '{city}'."}


def get_home_weather(tool_context: ToolContext) -> dict:
    """Returns the weather for the user's saved home city.

    Use this when the user asks for 'my weather', 'weather at home',
    or 'weather here' without specifying a city name.
    """
    home = tool_context.state.get("home_city")
    if not home:
        return {
            "status": "error",
            "error_message": (
                "No home city set. Ask the user to tell you their city first, "
                "then use set_home_city."
            ),
        }
    return get_weather(home)


def add_to_watchlist(city: str, tool_context: ToolContext) -> dict:
    """Adds a city to the user's weather watchlist.

    Use when the user says they want to track or monitor a city.

    Args:
        city: The city name to add to the watchlist.
    """
    key = city.lower().strip()
    if key not in WEATHER_DATA:
        return {
            "status": "error",
            "error_message": f"'{city}' not supported. Choose from: {', '.join(WEATHER_DATA)}.",
        }

    # Read existing watchlist (default to empty list if not set yet)
    watchlist: list = tool_context.state.get("watchlist", [])
    if key in watchlist:
        return {"status": "info", "message": f"'{city}' is already in your watchlist."}

    watchlist.append(key)
    tool_context.state["watchlist"] = watchlist  # write back
    return {"status": "success", "message": f"Added '{city}' to your watchlist."}


def get_watchlist_weather(tool_context: ToolContext) -> dict:
    """Returns weather for all cities in the user's watchlist.

    Use when the user asks about their watchlist or all tracked cities.
    """
    watchlist: list = tool_context.state.get("watchlist", [])
    if not watchlist:
        return {
            "status": "info",
            "message": "Your watchlist is empty. Add cities with add_to_watchlist.",
        }

    reports = {}
    for city in watchlist:
        result = get_weather(city)
        reports[city.title()] = result.get("report", result.get("error_message"))

    return {"status": "success", "reports": reports}


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="stateful_weather_agent",
    model=get_model(),
    description=(
        "A weather and time assistant that remembers the user's home city "
        "and maintains a city watchlist across the conversation."
    ),
    instruction=(
        "You are WeatherBot Pro, a smart weather assistant with memory.\n\n"
        "You can:\n"
        "  1. Look up weather and time for specific cities.\n"
        "  2. Remember the user's home city (set_home_city / get_home_weather).\n"
        "  3. Maintain a watchlist of cities the user cares about.\n\n"
        "Guidelines:\n"
        "  - When the user mentions their city or home, call set_home_city.\n"
        "  - When they ask about 'my weather' or 'weather here', call get_home_weather.\n"
        "  - When they ask about their watchlist, call get_watchlist_weather.\n"
        "  - Always use tools for data; never fabricate weather or time.\n"
        "  - Supported cities: New York, London, Tokyo, Mumbai, New Delhi, Chicago, Milwaukee.\n"
        "  - Be concise and friendly."
    ),
    tools=[
        get_weather,
        get_current_time,
        set_home_city,
        get_home_weather,
        add_to_watchlist,
        get_watchlist_weather,
    ],
)
