"""
shared_tools.py — Shared weather, time, and city data + tools
==============================================================
Single source of truth for all agent tools related to weather, time,
and city information. Import from here to avoid duplication.
"""

import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

WEATHER_DATA = {
    "new york":  {"condition": "Sunny",        "temp_c": 25, "humidity": "45%", "uv_index": 6},
    "london":    {"condition": "Cloudy",        "temp_c": 14, "humidity": "72%", "uv_index": 2},
    "tokyo":     {"condition": "Partly Cloudy", "temp_c": 18, "humidity": "60%", "uv_index": 4},
    "mumbai":    {"condition": "Hot & Humid",   "temp_c": 34, "humidity": "85%", "uv_index": 8},
    "new delhi": {"condition": "Hazy",          "temp_c": 32, "humidity": "55%", "uv_index": 7},
    "chicago":   {"condition": "Windy",         "temp_c": 10, "humidity": "58%", "uv_index": 3},
    "milwaukee": {"condition": "Overcast",      "temp_c": 8,  "humidity": "63%", "uv_index": 2},
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

CITY_INFO = {
    "new york": {
        "country": "USA", "currency": "USD ($)", "language": "English",
        "population": "8.3 million", "dial_code": "+1",
        "best_months": ["April", "May", "June", "September", "October", "November"],
    },
    "london": {
        "country": "UK", "currency": "GBP (£)", "language": "English",
        "population": "9 million", "dial_code": "+44",
        "best_months": ["June", "July", "August", "September"],
    },
    "tokyo": {
        "country": "Japan", "currency": "JPY (¥)", "language": "Japanese",
        "population": "14 million", "dial_code": "+81",
        "best_months": ["March", "April", "October", "November"],
    },
    "mumbai": {
        "country": "India", "currency": "INR (₹)", "language": "Hindi / Marathi",
        "population": "21 million", "dial_code": "+91",
        "best_months": ["November", "December", "January", "February"],
    },
    "new delhi": {
        "country": "India", "currency": "INR (₹)", "language": "Hindi / English",
        "population": "33 million", "dial_code": "+91",
        "best_months": ["October", "November", "December", "February", "March"],
    },
    "chicago": {
        "country": "USA", "currency": "USD ($)", "language": "English",
        "population": "2.7 million", "dial_code": "+1",
        "best_months": ["June", "July", "August", "September"],
    },
    "milwaukee": {
        "country": "USA", "currency": "USD ($)", "language": "English",
        "population": "577,000", "dial_code": "+1",
        "best_months": ["June", "July", "August"],
    },
}

SUPPORTED_CITIES = list(CITY_TIMEZONES.keys())
SUPPORTED_CITIES_DISPLAY = ", ".join(c.title() for c in SUPPORTED_CITIES)

# ---------------------------------------------------------------------------
# Basic tools (used in all phases)
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    """Returns the current weather report for a given city.

    Use this when the user asks about weather, temperature, or conditions
    in a specific city.

    Args:
        city: The name of the city (e.g. 'New York', 'Chicago', 'Tokyo').
    """
    key = city.lower().strip()
    if key in WEATHER_DATA:
        d = WEATHER_DATA[key]
        temp_f = round(int(d["temp_c"]) * 9 / 5 + 32)
        return {
            "status": "success",
            "city": city.title(),
            "report": f"{d['condition']}, {d['temp_c']}°C / {temp_f}°F, Humidity: {d['humidity']}",
        }
    return {"status": "error", "error_message": f"No data for '{city}'. Supported: {SUPPORTED_CITIES_DISPLAY}."}


def get_current_time(city: str) -> dict:
    """Returns the current local time for a given city.

    Use this when the user asks what time it is in a specific city.

    Args:
        city: The name of the city.
    """
    key = city.lower().strip()
    if key in CITY_TIMEZONES:
        tz = ZoneInfo(CITY_TIMEZONES[key])
        now = datetime.datetime.now(tz)
        return {
            "status": "success",
            "city": city.title(),
            "report": now.strftime("%A, %B %d %Y  %I:%M %p %Z"),
        }
    return {"status": "error", "error_message": f"No timezone for '{city}'. Supported: {SUPPORTED_CITIES_DISPLAY}."}


# ---------------------------------------------------------------------------
# New expanded tools
# ---------------------------------------------------------------------------

def get_all_times() -> dict:
    """Returns the current local time for ALL supported cities at once.

    Use this when the user asks for the time in all cities, wants a world
    clock overview, or asks 'what time is it everywhere?'
    """
    results = {}
    for city, tz_name in CITY_TIMEZONES.items():
        tz = ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
        results[city.title()] = now.strftime("%I:%M %p %Z  (%A)")
    return {"status": "success", "times": results}


def get_all_weather() -> dict:
    """Returns the current weather for ALL supported cities at once.

    Use this when the user wants a weather overview, asks 'weather everywhere',
    or wants to compare conditions across all cities.
    """
    results = {}
    for city, d in WEATHER_DATA.items():
        temp_f = round(int(d["temp_c"]) * 9 / 5 + 32)
        results[city.title()] = f"{d['condition']}, {d['temp_c']}°C / {temp_f}°F"
    return {"status": "success", "weather": results}


def compare_weather(city1: str, city2: str) -> dict:
    """Compares weather side-by-side between two cities.

    Use this when the user asks to compare weather between two specific cities,
    or asks 'which city is warmer/colder/better weather?'

    Args:
        city1: First city name.
        city2: Second city name.
    """
    k1, k2 = city1.lower().strip(), city2.lower().strip()
    missing = [c for c, k in [(city1, k1), (city2, k2)] if k not in WEATHER_DATA]
    if missing:
        return {"status": "error", "error_message": f"No data for: {', '.join(missing)}. Supported: {SUPPORTED_CITIES_DISPLAY}."}

    d1, d2 = WEATHER_DATA[k1], WEATHER_DATA[k2]
    temp_diff = int(d1["temp_c"]) - int(d2["temp_c"])

    return {
        "status": "success",
        city1.title(): {
            "condition": d1["condition"],
            "temp": f"{d1['temp_c']}°C / {round(int(d1['temp_c']) * 9/5 + 32)}°F",
            "humidity": d1["humidity"],
        },
        city2.title(): {
            "condition": d2["condition"],
            "temp": f"{d2['temp_c']}°C / {round(int(d2['temp_c']) * 9/5 + 32)}°F",
            "humidity": d2["humidity"],
        },
        "comparison": (
            f"{city1.title()} is {abs(temp_diff)}°C {'warmer' if temp_diff > 0 else 'cooler'} than {city2.title()}."
            if temp_diff != 0 else f"Both cities are the same temperature."
        ),
    }


def get_city_info(city: str) -> dict:
    """Returns general information about a city: country, currency, language,
    population, international dialling code, and best months to visit.

    Use this when the user asks about a city in general, wants to know the
    currency, language, or population, or asks 'tell me about X'.

    Args:
        city: The name of the city.
    """
    key = city.lower().strip()
    if key in CITY_INFO:
        info = CITY_INFO[key]
        return {
            "status": "success",
            "city": city.title(),
            **info,
        }
    return {"status": "error", "error_message": f"No info for '{city}'. Supported: {SUPPORTED_CITIES_DISPLAY}."}


def get_best_cities_for_month(month: str) -> dict:
    """Returns which cities are best to visit in a given month.

    Use this when the user asks 'where should I go in July?',
    'best city to visit in December?', or similar.

    Args:
        month: The month name (e.g. 'January', 'July', 'December').
    """
    month_title = month.strip().title()
    matches = [
        city.title()
        for city, info in CITY_INFO.items()
        if month_title in info["best_months"]
    ]
    if not matches:
        all_months = sorted({m for info in CITY_INFO.values() for m in info["best_months"]})
        return {
            "status": "error",
            "error_message": f"'{month}' not recognised or no cities rated for that month. Try: {', '.join(all_months)}.",
        }
    return {
        "status": "success",
        "month": month_title,
        "best_cities": matches,
        "tip": f"These cities are at their best in {month_title}: {', '.join(matches)}.",
    }


# ---------------------------------------------------------------------------
# Convenience list — import this into agent files
# ---------------------------------------------------------------------------

BASIC_TOOLS = [get_weather, get_current_time]

EXPANDED_TOOLS = [
    get_weather,
    get_current_time,
    get_all_times,
    get_all_weather,
    compare_weather,
    get_city_info,
    get_best_cities_for_month,
]
