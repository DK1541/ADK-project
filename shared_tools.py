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
    # North America
    "new york":     {"condition": "Sunny",           "temp_c": 25, "humidity": "45%", "uv_index": 6},
    "los angeles":  {"condition": "Sunny",            "temp_c": 24, "humidity": "45%", "uv_index": 7},
    "chicago":      {"condition": "Windy",            "temp_c": 10, "humidity": "58%", "uv_index": 3},
    "toronto":      {"condition": "Partly Cloudy",    "temp_c": 12, "humidity": "58%", "uv_index": 4},
    "mexico city":  {"condition": "Mild & Cloudy",    "temp_c": 18, "humidity": "55%", "uv_index": 6},
    "milwaukee":    {"condition": "Overcast",         "temp_c": 8,  "humidity": "63%", "uv_index": 2},
    # South America
    "sao paulo":    {"condition": "Partly Cloudy",    "temp_c": 22, "humidity": "75%", "uv_index": 6},
    "buenos aires": {"condition": "Partly Cloudy",    "temp_c": 17, "humidity": "65%", "uv_index": 5},
    # Europe
    "london":       {"condition": "Cloudy",           "temp_c": 14, "humidity": "72%", "uv_index": 2},
    "paris":        {"condition": "Partly Cloudy",    "temp_c": 16, "humidity": "65%", "uv_index": 3},
    "berlin":       {"condition": "Overcast",         "temp_c": 13, "humidity": "70%", "uv_index": 2},
    "rome":         {"condition": "Sunny",            "temp_c": 20, "humidity": "50%", "uv_index": 6},
    "barcelona":    {"condition": "Sunny",            "temp_c": 21, "humidity": "55%", "uv_index": 7},
    "amsterdam":    {"condition": "Overcast & Rainy", "temp_c": 12, "humidity": "78%", "uv_index": 2},
    "vienna":       {"condition": "Partly Cloudy",    "temp_c": 14, "humidity": "62%", "uv_index": 3},
    "zurich":       {"condition": "Overcast",         "temp_c": 12, "humidity": "67%", "uv_index": 2},
    "stockholm":    {"condition": "Cold & Cloudy",    "temp_c": 8,  "humidity": "72%", "uv_index": 1},
    "istanbul":     {"condition": "Mild & Cloudy",    "temp_c": 17, "humidity": "68%", "uv_index": 4},
    "moscow":       {"condition": "Cold & Overcast",  "temp_c": 5,  "humidity": "75%", "uv_index": 1},
    # Africa
    "cairo":        {"condition": "Sunny & Dry",      "temp_c": 30, "humidity": "25%", "uv_index": 9},
    "lagos":        {"condition": "Hot & Humid",      "temp_c": 29, "humidity": "80%", "uv_index": 7},
    "nairobi":      {"condition": "Mild & Cloudy",    "temp_c": 21, "humidity": "65%", "uv_index": 6},
    "johannesburg": {"condition": "Mild & Clear",     "temp_c": 20, "humidity": "42%", "uv_index": 6},
    # Middle East
    "dubai":        {"condition": "Sunny & Hot",      "temp_c": 38, "humidity": "40%", "uv_index": 9},
    # South Asia
    "mumbai":       {"condition": "Hot & Humid",      "temp_c": 34, "humidity": "85%", "uv_index": 8},
    "new delhi":    {"condition": "Hazy",             "temp_c": 32, "humidity": "55%", "uv_index": 7},
    # East Asia
    "tokyo":        {"condition": "Partly Cloudy",    "temp_c": 18, "humidity": "60%", "uv_index": 4},
    "beijing":      {"condition": "Hazy",             "temp_c": 20, "humidity": "50%", "uv_index": 5},
    "shanghai":     {"condition": "Partly Cloudy",    "temp_c": 18, "humidity": "65%", "uv_index": 4},
    "seoul":        {"condition": "Clear",            "temp_c": 16, "humidity": "52%", "uv_index": 5},
    "hong kong":    {"condition": "Sunny",            "temp_c": 23, "humidity": "70%", "uv_index": 6},
    # Southeast Asia
    "bangkok":      {"condition": "Hot & Humid",      "temp_c": 33, "humidity": "80%", "uv_index": 8},
    "singapore":    {"condition": "Humid & Showery",  "temp_c": 30, "humidity": "85%", "uv_index": 7},
    "kuala lumpur": {"condition": "Thundery",         "temp_c": 28, "humidity": "82%", "uv_index": 6},
    "jakarta":      {"condition": "Hot & Humid",      "temp_c": 30, "humidity": "83%", "uv_index": 7},
    "manila":       {"condition": "Hot & Humid",      "temp_c": 32, "humidity": "78%", "uv_index": 8},
    # Oceania
    "sydney":       {"condition": "Sunny",            "temp_c": 22, "humidity": "55%", "uv_index": 5},
}

CITY_TIMEZONES = {
    # North America
    "new york":     "America/New_York",
    "los angeles":  "America/Los_Angeles",
    "chicago":      "America/Chicago",
    "toronto":      "America/Toronto",
    "mexico city":  "America/Mexico_City",
    "milwaukee":    "America/Chicago",
    # South America
    "sao paulo":    "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    # Europe
    "london":       "Europe/London",
    "paris":        "Europe/Paris",
    "berlin":       "Europe/Berlin",
    "rome":         "Europe/Rome",
    "barcelona":    "Europe/Madrid",
    "amsterdam":    "Europe/Amsterdam",
    "vienna":       "Europe/Vienna",
    "zurich":       "Europe/Zurich",
    "stockholm":    "Europe/Stockholm",
    "istanbul":     "Europe/Istanbul",
    "moscow":       "Europe/Moscow",
    # Africa
    "cairo":        "Africa/Cairo",
    "lagos":        "Africa/Lagos",
    "nairobi":      "Africa/Nairobi",
    "johannesburg": "Africa/Johannesburg",
    # Middle East
    "dubai":        "Asia/Dubai",
    # South Asia
    "mumbai":       "Asia/Kolkata",
    "new delhi":    "Asia/Kolkata",
    # East Asia
    "tokyo":        "Asia/Tokyo",
    "beijing":      "Asia/Shanghai",
    "shanghai":     "Asia/Shanghai",
    "seoul":        "Asia/Seoul",
    "hong kong":    "Asia/Hong_Kong",
    # Southeast Asia
    "bangkok":      "Asia/Bangkok",
    "singapore":    "Asia/Singapore",
    "kuala lumpur": "Asia/Kuala_Lumpur",
    "jakarta":      "Asia/Jakarta",
    "manila":       "Asia/Manila",
    # Oceania
    "sydney":       "Australia/Sydney",
}

CITY_INFO = {
    # North America
    "new york": {
        "country": "USA", "currency": "USD ($)", "language": "English",
        "population": "8.3 million", "dial_code": "+1",
        "best_months": ["April", "May", "June", "September", "October", "November"],
    },
    "los angeles": {
        "country": "USA", "currency": "USD ($)", "language": "English",
        "population": "3.9 million", "dial_code": "+1",
        "best_months": ["March", "April", "May", "September", "October", "November"],
    },
    "chicago": {
        "country": "USA", "currency": "USD ($)", "language": "English",
        "population": "2.7 million", "dial_code": "+1",
        "best_months": ["June", "July", "August", "September"],
    },
    "toronto": {
        "country": "Canada", "currency": "CAD ($)", "language": "English / French",
        "population": "2.9 million", "dial_code": "+1",
        "best_months": ["June", "July", "August", "September"],
    },
    "mexico city": {
        "country": "Mexico", "currency": "MXN ($)", "language": "Spanish",
        "population": "9.2 million", "dial_code": "+52",
        "best_months": ["March", "April", "May", "October", "November"],
    },
    "milwaukee": {
        "country": "USA", "currency": "USD ($)", "language": "English",
        "population": "577,000", "dial_code": "+1",
        "best_months": ["June", "July", "August"],
    },
    # South America
    "sao paulo": {
        "country": "Brazil", "currency": "BRL (R$)", "language": "Portuguese",
        "population": "22 million", "dial_code": "+55",
        "best_months": ["April", "May", "June", "July", "August"],
    },
    "buenos aires": {
        "country": "Argentina", "currency": "ARS ($)", "language": "Spanish",
        "population": "3.1 million", "dial_code": "+54",
        "best_months": ["September", "October", "November", "March", "April"],
    },
    # Europe
    "london": {
        "country": "UK", "currency": "GBP (£)", "language": "English",
        "population": "9 million", "dial_code": "+44",
        "best_months": ["June", "July", "August", "September"],
    },
    "paris": {
        "country": "France", "currency": "EUR (€)", "language": "French",
        "population": "2.1 million", "dial_code": "+33",
        "best_months": ["April", "May", "June", "September", "October"],
    },
    "berlin": {
        "country": "Germany", "currency": "EUR (€)", "language": "German",
        "population": "3.7 million", "dial_code": "+49",
        "best_months": ["June", "July", "August", "September"],
    },
    "rome": {
        "country": "Italy", "currency": "EUR (€)", "language": "Italian",
        "population": "2.8 million", "dial_code": "+39",
        "best_months": ["April", "May", "June", "September", "October"],
    },
    "barcelona": {
        "country": "Spain", "currency": "EUR (€)", "language": "Spanish / Catalan",
        "population": "1.6 million", "dial_code": "+34",
        "best_months": ["May", "June", "September", "October"],
    },
    "amsterdam": {
        "country": "Netherlands", "currency": "EUR (€)", "language": "Dutch",
        "population": "873,000", "dial_code": "+31",
        "best_months": ["April", "May", "June", "July", "August"],
    },
    "vienna": {
        "country": "Austria", "currency": "EUR (€)", "language": "German",
        "population": "1.9 million", "dial_code": "+43",
        "best_months": ["April", "May", "June", "September", "October"],
    },
    "zurich": {
        "country": "Switzerland", "currency": "CHF (Fr)", "language": "German",
        "population": "434,000", "dial_code": "+41",
        "best_months": ["June", "July", "August", "September"],
    },
    "stockholm": {
        "country": "Sweden", "currency": "SEK (kr)", "language": "Swedish",
        "population": "975,000", "dial_code": "+46",
        "best_months": ["June", "July", "August"],
    },
    "istanbul": {
        "country": "Turkey", "currency": "TRY (₺)", "language": "Turkish",
        "population": "15.5 million", "dial_code": "+90",
        "best_months": ["April", "May", "September", "October"],
    },
    "moscow": {
        "country": "Russia", "currency": "RUB (₽)", "language": "Russian",
        "population": "12.5 million", "dial_code": "+7",
        "best_months": ["June", "July", "August"],
    },
    # Africa
    "cairo": {
        "country": "Egypt", "currency": "EGP (£)", "language": "Arabic",
        "population": "21 million", "dial_code": "+20",
        "best_months": ["October", "November", "December", "February", "March"],
    },
    "lagos": {
        "country": "Nigeria", "currency": "NGN (₦)", "language": "English",
        "population": "15 million", "dial_code": "+234",
        "best_months": ["November", "December", "January"],
    },
    "nairobi": {
        "country": "Kenya", "currency": "KES (KSh)", "language": "Swahili / English",
        "population": "4.9 million", "dial_code": "+254",
        "best_months": ["June", "July", "August", "January", "February"],
    },
    "johannesburg": {
        "country": "South Africa", "currency": "ZAR (R)", "language": "Zulu / Xhosa / Afrikaans / English",
        "population": "5.8 million", "dial_code": "+27",
        "best_months": ["May", "June", "July", "August", "September"],
    },
    # Middle East
    "dubai": {
        "country": "UAE", "currency": "AED (د.إ)", "language": "Arabic",
        "population": "3.5 million", "dial_code": "+971",
        "best_months": ["November", "December", "January", "February", "March"],
    },
    # South Asia
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
    # East Asia
    "tokyo": {
        "country": "Japan", "currency": "JPY (¥)", "language": "Japanese",
        "population": "14 million", "dial_code": "+81",
        "best_months": ["March", "April", "October", "November"],
    },
    "beijing": {
        "country": "China", "currency": "CNY (¥)", "language": "Mandarin",
        "population": "21.9 million", "dial_code": "+86",
        "best_months": ["April", "May", "September", "October"],
    },
    "shanghai": {
        "country": "China", "currency": "CNY (¥)", "language": "Mandarin",
        "population": "24.9 million", "dial_code": "+86",
        "best_months": ["April", "May", "October", "November"],
    },
    "seoul": {
        "country": "South Korea", "currency": "KRW (₩)", "language": "Korean",
        "population": "9.7 million", "dial_code": "+82",
        "best_months": ["April", "May", "September", "October"],
    },
    "hong kong": {
        "country": "China (SAR)", "currency": "HKD ($)", "language": "Cantonese / English",
        "population": "7.5 million", "dial_code": "+852",
        "best_months": ["October", "November", "December", "March", "April"],
    },
    # Southeast Asia
    "bangkok": {
        "country": "Thailand", "currency": "THB (฿)", "language": "Thai",
        "population": "10.5 million", "dial_code": "+66",
        "best_months": ["November", "December", "January", "February"],
    },
    "singapore": {
        "country": "Singapore", "currency": "SGD ($)", "language": "English / Malay / Mandarin / Tamil",
        "population": "5.9 million", "dial_code": "+65",
        "best_months": ["February", "March", "July", "August"],
    },
    "kuala lumpur": {
        "country": "Malaysia", "currency": "MYR (RM)", "language": "Malay / English / Mandarin",
        "population": "1.8 million", "dial_code": "+60",
        "best_months": ["May", "June", "July", "August"],
    },
    "jakarta": {
        "country": "Indonesia", "currency": "IDR (Rp)", "language": "Indonesian",
        "population": "10.6 million", "dial_code": "+62",
        "best_months": ["June", "July", "August", "September"],
    },
    "manila": {
        "country": "Philippines", "currency": "PHP (₱)", "language": "Filipino / English",
        "population": "1.8 million", "dial_code": "+63",
        "best_months": ["November", "December", "January", "February", "March"],
    },
    # Oceania
    "sydney": {
        "country": "Australia", "currency": "AUD ($)", "language": "English",
        "population": "5.3 million", "dial_code": "+61",
        "best_months": ["September", "October", "November", "March", "April"],
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
