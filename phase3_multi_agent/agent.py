"""
PHASE 3: Multi-Agent System
==============================
Architecture:
                    ┌─────────────────────────┐
                    │   coordinator_agent      │
                    │  (routes all requests)   │
                    └──────────┬──────────────┘
                               │ delegates to sub_agents
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
    │ weather_agent│  │  time_agent  │  │  travel_advisor  │
    │ (weather data│  │ (time zones) │  │ (packing tips,   │
    │  + humidity) │  │              │  │  travel advice)  │
    └──────────────┘  └──────────────┘  └──────────────────┘

Concepts covered:
  - sub_agents: how the coordinator delegates to specialists
  - description: the key field that tells the coordinator WHEN to delegate
  - Each specialist is a full Agent with its own tools and instruction
  - AgentTool: alternative pattern (wrap an agent as a tool — shown at bottom)
  - SequentialAgent: run agents in a fixed pipeline order

HOW DELEGATION WORKS:
  The coordinator agent's LLM sees the name + description of each sub-agent.
  When a user message matches a sub-agent's description, the coordinator
  passes the request to that agent. The sub-agent runs its own tools,
  produces a response, and the coordinator returns it to the user.
  You don't write any routing logic — the LLM handles it.
"""

import datetime
import os
from zoneinfo import ZoneInfo

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from model_config import get_model

# ---------------------------------------------------------------------------
# Per-agent model assignment
# ---------------------------------------------------------------------------
# When MODEL_PROVIDER=ollama each specialist runs on a different model.
# You can tune this — swap any model name to whichever you prefer.
# When MODEL_PROVIDER=anthropic or google, all agents fall back to get_model().
# ---------------------------------------------------------------------------

def _ollama(model_name: str):
    """Returns a LiteLlm instance for the given local Ollama model."""
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    os.environ["OLLAMA_API_BASE"] = base
    return LiteLlm(model=f"ollama_chat/{model_name}")

_provider = os.getenv("MODEL_PROVIDER", "google").lower()

if _provider == "ollama":
    # Each specialist gets its own model — change these to suit your hardware
    WEATHER_MODEL  = _ollama("llama3.2:latest")   # lightest — fast lookups
    TIME_MODEL     = _ollama("llama3.1:latest")   # solid reasoning for timezone math
    TRAVEL_MODEL   = _ollama("qwen2.5:14b")       # strongest — rich travel advice
    COORD_MODEL    = _ollama("qwen2.5:14b")       # coordinator needs best routing ability
else:
    # Anthropic / Google — single model for all agents
    WEATHER_MODEL = TIME_MODEL = TRAVEL_MODEL = COORD_MODEL = get_model()

# ---------------------------------------------------------------------------
# Data (same mock data as previous phases)
# ---------------------------------------------------------------------------

WEATHER_DATA = {
    "new york":   {"condition": "Sunny",        "temp_c": 25, "humidity": "45%", "uv_index": 6},
    "london":     {"condition": "Cloudy",        "temp_c": 14, "humidity": "72%", "uv_index": 2},
    "tokyo":      {"condition": "Partly Cloudy", "temp_c": 18, "humidity": "60%", "uv_index": 4},
    "mumbai":     {"condition": "Hot & Humid",   "temp_c": 34, "humidity": "85%", "uv_index": 8},
    "new delhi":  {"condition": "Hazy",          "temp_c": 32, "humidity": "55%", "uv_index": 7},
    "chicago":    {"condition": "Windy",         "temp_c": 10, "humidity": "58%", "uv_index": 3},
    "milwaukee":  {"condition": "Overcast",      "temp_c": 8,  "humidity": "63%", "uv_index": 2},
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

TRAVEL_TIPS = {
    "new york": {
        "best_season": "Spring (Apr–Jun) or Fall (Sep–Nov)",
        "must_see": ["Central Park", "Times Square", "Brooklyn Bridge", "MoMA"],
        "local_tips": "Get a MetroCard for the subway. Book restaurants in advance.",
    },
    "london": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Tower of London", "British Museum", "Hyde Park", "Borough Market"],
        "local_tips": "Tap your card on the Tube. Weather changes fast — carry a jacket.",
    },
    "tokyo": {
        "best_season": "Cherry blossom (Mar–Apr) or Autumn (Oct–Nov)",
        "must_see": ["Shinjuku Gyoen", "Senso-ji", "Shibuya Crossing", "Tsukiji Market"],
        "local_tips": "Get a Suica card. Most places are cash-friendly. Shoes easy to remove.",
    },
    "mumbai": {
        "best_season": "Winter (Nov–Feb)",
        "must_see": ["Gateway of India", "Elephanta Caves", "Marine Drive", "Dharavi"],
        "local_tips": "Avoid monsoon (Jun–Sep). Local trains are fast but very crowded.",
    },
    "new delhi": {
        "best_season": "Winter (Oct–Mar)",
        "must_see": ["Red Fort", "Qutub Minar", "India Gate", "Humayun's Tomb"],
        "local_tips": "Use the Delhi Metro. Bargain at markets. Air quality varies — check AQI.",
    },
    "chicago": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Millennium Park", "Art Institute of Chicago", "Navy Pier", "The 606 Trail"],
        "local_tips": "Wind off Lake Michigan makes it feel colder — layer up. The 'L' train covers most attractions.",
    },
    "milwaukee": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Milwaukee Art Museum", "Harley-Davidson Museum", "Third Ward", "Lakefront Brewery"],
        "local_tips": "Very walkable downtown. Known for Friday fish fries and craft beer. Free Summerfest in summer.",
    },
}


# ---------------------------------------------------------------------------
# SPECIALIST 1: Weather Agent
# ---------------------------------------------------------------------------

def get_weather_detailed(city: str) -> dict:
    """Returns detailed weather including temperature, humidity, and UV index.

    Args:
        city: The city name.
    """
    key = city.lower().strip()
    if key in WEATHER_DATA:
        d = WEATHER_DATA[key]
        temp_f = round(int(d["temp_c"]) * 9 / 5 + 32)
        return {
            "status": "success",
            "city": city.title(),
            "condition": d["condition"],
            "temperature": f"{d['temp_c']}°C / {temp_f}°F",
            "humidity": d["humidity"],
            "uv_index": d["uv_index"],
            "uv_advice": (
                "High — wear sunscreen" if d["uv_index"] >= 6
                else "Moderate — some protection advised" if d["uv_index"] >= 3
                else "Low"
            ),
        }
    return {
        "status": "error",
        "error_message": f"No data for '{city}'. Supported: {', '.join(WEATHER_DATA)}.",
    }


weather_agent = Agent(
    name="weather_specialist",
    model=WEATHER_MODEL,
    # IMPORTANT: description is what the coordinator reads to decide delegation.
    # Be specific — vague descriptions cause mis-routing.
    description=(
        "Handles all weather-related questions: temperature, humidity, "
        "conditions, UV index for cities."
    ),
    instruction=(
        "You are the Weather Specialist. Answer weather questions accurately.\n"
        "Always use get_weather_detailed for any city weather request.\n"
        "Include UV advice when UV index is high.\n"
        "Supported cities: New York, London, Tokyo, Mumbai, New Delhi, Chicago, Milwaukee."
    ),
    tools=[get_weather_detailed],
)


# ---------------------------------------------------------------------------
# SPECIALIST 2: Time Agent
# ---------------------------------------------------------------------------

def get_time_with_offset(city: str) -> dict:
    """Returns current local time and UTC offset for a city.

    Args:
        city: The city name.
    """
    key = city.lower().strip()
    if key in CITY_TIMEZONES:
        tz = ZoneInfo(CITY_TIMEZONES[key])
        now = datetime.datetime.now(tz)
        offset = now.strftime("%z")
        formatted_offset = f"UTC{offset[:3]}:{offset[3:]}"
        return {
            "status": "success",
            "city": city.title(),
            "local_time": now.strftime("%A, %B %d %Y  %I:%M %p"),
            "timezone": CITY_TIMEZONES[key],
            "utc_offset": formatted_offset,
        }
    return {
        "status": "error",
        "error_message": f"No timezone for '{city}'. Supported: {', '.join(CITY_TIMEZONES)}.",
    }


def get_time_difference(city1: str, city2: str) -> dict:
    """Returns the time difference in hours between two cities.

    Use this when the user asks things like 'if it's 3pm in London,
    what time is it in Tokyo?'

    Args:
        city1: First city name.
        city2: Second city name.
    """
    k1, k2 = city1.lower().strip(), city2.lower().strip()
    missing = [c for c, k in [(city1, k1), (city2, k2)] if k not in CITY_TIMEZONES]
    if missing:
        return {
            "status": "error",
            "error_message": f"No timezone info for: {', '.join(missing)}.",
        }

    tz1 = ZoneInfo(CITY_TIMEZONES[k1])
    tz2 = ZoneInfo(CITY_TIMEZONES[k2])
    now = datetime.datetime.now(datetime.timezone.utc)
    offset1 = now.astimezone(tz1).utcoffset().total_seconds() / 3600
    offset2 = now.astimezone(tz2).utcoffset().total_seconds() / 3600
    diff = offset2 - offset1

    return {
        "status": "success",
        "city1": city1.title(),
        "city2": city2.title(),
        "difference_hours": diff,
        "description": (
            f"{city2.title()} is {abs(diff):.0f} hour(s) "
            f"{'ahead of' if diff > 0 else 'behind'} {city1.title()}."
        ),
    }


time_agent = Agent(
    name="time_specialist",
    model=TIME_MODEL,
    description=(
        "Handles time-related questions: current local time, timezone info, "
        "and time differences between cities."
    ),
    instruction=(
        "You are the Time Specialist. Answer all time and timezone questions.\n"
        "Use get_time_with_offset for 'what time is it in X' questions.\n"
        "Use get_time_difference for 'time difference between X and Y' questions.\n"
        "Supported cities: New York, London, Tokyo, Mumbai, New Delhi, Chicago, Milwaukee."
    ),
    tools=[get_time_with_offset, get_time_difference],
)


# ---------------------------------------------------------------------------
# SPECIALIST 3: Travel Advisor
# ---------------------------------------------------------------------------

def get_travel_tips(city: str) -> dict:
    """Returns travel tips, best visiting seasons, and must-see attractions.

    Use for travel advice, trip planning, packing suggestions, or
    questions about what to see and do in a city.

    Args:
        city: The city name.
    """
    key = city.lower().strip()
    if key in TRAVEL_TIPS:
        tips = TRAVEL_TIPS[key]
        return {
            "status": "success",
            "city": city.title(),
            "best_season": tips["best_season"],
            "must_see": tips["must_see"],
            "local_tips": tips["local_tips"],
        }
    return {
        "status": "error",
        "error_message": f"No travel data for '{city}'. Supported: {', '.join(TRAVEL_TIPS)}.",
    }


def get_packing_advice(city: str) -> dict:
    """Returns packing advice based on current weather conditions in a city.

    Use when the user asks what to pack or wear for a trip.

    Args:
        city: The destination city.
    """
    key = city.lower().strip()
    if key not in WEATHER_DATA:
        return {
            "status": "error",
            "error_message": f"No data for '{city}'.",
        }

    d = WEATHER_DATA[key]
    items = []

    if d["temp_c"] < 15:
        items += ["Warm jacket", "Layers", "Scarf and gloves"]
    elif d["temp_c"] > 28:
        items += ["Light breathable clothing", "Sun hat", "Sunscreen SPF 50+"]
    else:
        items += ["Light jacket or cardigan", "Comfortable walking shoes"]

    if int(d["humidity"].strip("%")) > 70:
        items.append("Moisture-wicking fabrics")

    if d["uv_index"] >= 6:
        items.append("Sunglasses (UV protection)")

    return {
        "status": "success",
        "city": city.title(),
        "recommended_packing": items,
        "note": f"Based on current conditions: {d['condition']}, {d['temp_c']}°C.",
    }


travel_agent = Agent(
    name="travel_advisor",
    model=TRAVEL_MODEL,
    description=(
        "Handles travel planning: attractions, best travel seasons, packing advice, "
        "and local tips for cities."
    ),
    instruction=(
        "You are the Travel Advisor. Help users plan their trips.\n"
        "Use get_travel_tips for general trip planning and attractions.\n"
        "Use get_packing_advice for packing and clothing questions.\n"
        "You can combine both tools when the user is planning a full trip.\n"
        "Supported cities: New York, London, Tokyo, Mumbai, New Delhi, Chicago, Milwaukee."
    ),
    tools=[get_travel_tips, get_packing_advice],
)


# ---------------------------------------------------------------------------
# COORDINATOR — the root agent that users talk to
# ---------------------------------------------------------------------------
# HOW IT WORKS:
#   - sub_agents lists the specialists
#   - The coordinator's LLM reads each specialist's `description`
#   - It delegates the request to whichever specialist fits best
#   - If a request spans multiple specialists (e.g. "weather AND travel tips
#     for Tokyo"), the coordinator can call multiple sub-agents in sequence
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="travel_assistant_coordinator",
    model=COORD_MODEL,
    description="A comprehensive travel assistant covering weather, time, and trip planning.",
    instruction=(
        "You are TravelBot, a comprehensive travel assistant.\n\n"
        "You coordinate three specialists:\n"
        "  - weather_specialist : weather conditions and UV info\n"
        "  - time_specialist    : local times and timezone differences\n"
        "  - travel_advisor     : attractions, trip planning, packing advice\n\n"
        "Guidelines:\n"
        "  - Route each part of the user's question to the right specialist.\n"
        "  - If a question touches multiple domains, consult all relevant specialists.\n"
        "  - Synthesise the answers into one clear, helpful response.\n"
        "  - Never make up data — always delegate to specialists.\n"
        "  - If unsure which specialist to use, ask the user to clarify.\n"
        "  - Supported cities: New York, London, Tokyo, Mumbai, New Delhi, Chicago, Milwaukee."
    ),
    sub_agents=[weather_agent, time_agent, travel_agent],
)


# ---------------------------------------------------------------------------
# BONUS CONCEPT: SequentialAgent
# ---------------------------------------------------------------------------
# A SequentialAgent runs sub-agents one after another in a fixed order,
# passing output_key state between them. Useful for pipelines where each
# step depends on the previous one.
#
# Example: a trip briefing pipeline:
#   Step 1 → weather_agent writes to state["weather_report"]
#   Step 2 → travel_agent reads {weather_report} from its instruction template
#
# Uncomment to explore:
#
# weather_step = Agent(
#     name="weather_step",
#     model=get_model(),
#     description="Fetches weather for the trip destination.",
#     instruction="Get the weather for {destination} and summarise it.",
#     tools=[get_weather_detailed],
#     output_key="weather_report",   # saves response to state["weather_report"]
# )
#
# travel_step = Agent(
#     name="travel_step",
#     model=get_model(),
#     description="Gives travel advice considering the weather.",
#     instruction=(
#         "Weather report: {weather_report}\n"   # reads from state
#         "Given this weather, give packing and travel tips for {destination}."
#     ),
#     tools=[get_travel_tips, get_packing_advice],
# )
#
# trip_pipeline = SequentialAgent(
#     name="trip_briefing_pipeline",
#     sub_agents=[weather_step, travel_step],
# )
