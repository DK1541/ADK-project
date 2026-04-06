"""
ADK Multi-Agent System
=======================
One coordinator routes to six specialist agents. Each specialist is
an expert in its domain. The coordinator synthesises answers across
multiple specialists when a request spans domains.

Specialists:
  1. weather_time   — live weather, times, comparisons, world clock
  2. travel         — trip planning, packing, city info, best-month picks
  3. math_science   — calculations, unit conversions, logic, science Q&A
  4. language       — translation, grammar, summarisation, writing assistance
  5. code           — write, explain, debug, review code in any language
  6. knowledge      — research, general knowledge, history, explanations
  7. media          — image/video editing, slideshow creation, frame extraction

Model assignment is fully driven by .env — see model_config.py.
"""

import ast
import math
import os
import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from model_config import get_model
from media_tools import (
    get_image_info, edit_image, add_text_to_image,
    photos_to_video, get_video_info, edit_video,
    extract_video_frames, merge_videos,
    detect_objects_in_image, count_objects, detect_objects_in_video,
)
from shared_tools import (
    WEATHER_DATA, CITY_TIMEZONES, SUPPORTED_CITIES_DISPLAY,
    get_weather, get_current_time,
    get_all_times, get_all_weather, compare_weather,
    get_city_info, get_best_cities_for_month,
)

# ---------------------------------------------------------------------------
# Per-specialist model assignment (all driven by .env)
# ---------------------------------------------------------------------------

def _ollama(name: str):
    os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    return LiteLlm(model=f"ollama_chat/{name}")

def _hf(name: str):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
    return LiteLlm(model=name)

_provider = os.getenv("MODEL_PROVIDER", "google").lower()
_hf_default = os.getenv("HF_MODEL", "huggingface/sambanova/Qwen/Qwen2.5-72B-Instruct")
_ol_default = os.getenv("OLLAMA_MODEL", "llama3.2")

def _model(env_key: str, ollama_default: str = ""):
    """Picks the right model object for a specialist, driven by env vars.

    ollama_default: used when OLLAMA_MODEL_<SPECIALIST> is not set in .env.
    Falls back to OLLAMA_MODEL (the global default) if ollama_default is also empty.
    """
    if _provider == "ollama":
        return _ollama(os.getenv(env_key, ollama_default or _ol_default))
    if _provider == "huggingface":
        return _hf(os.getenv(env_key.replace("OLLAMA_MODEL_", "HF_MODEL_"), _hf_default))
    return get_model()

WEATHER_MODEL  = _model("OLLAMA_MODEL_WEATHER")
TIME_MODEL     = _model("OLLAMA_MODEL_TIME")
TRAVEL_MODEL   = _model("OLLAMA_MODEL_TRAVEL")
MATH_MODEL     = _model("OLLAMA_MODEL_MATH")
# aya:8b supports 23+ languages with tool calling; overridable via OLLAMA_MODEL_LANGUAGE
# Set OLLAMA_MODEL_LANGUAGE=aya-expanse:32b in .env for best translation quality
LANGUAGE_MODEL = _model("OLLAMA_MODEL_LANGUAGE", "aya:8b")
CODE_MODEL     = _model("OLLAMA_MODEL_CODE")
KNOWLEDGE_MODEL= _model("OLLAMA_MODEL_KNOWLEDGE")
COORD_MODEL    = _model("OLLAMA_MODEL_COORDINATOR")
# minicpm-v is a vision model — it can see uploaded images AND call tools
# overridable via OLLAMA_MODEL_MEDIA (e.g. llava:13b for higher quality)
MEDIA_MODEL    = _model("OLLAMA_MODEL_MEDIA", "minicpm-v")


# ===========================================================================
# SPECIALIST 1 — WEATHER & TIME
# ===========================================================================

def get_weather_detailed(city: str) -> dict:
    """Returns detailed weather: condition, temperature, humidity, and UV index.

    Use for any single-city weather question.

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
    return {"status": "error", "error_message": f"No data for '{city}'. Supported: {SUPPORTED_CITIES_DISPLAY}."}


def get_time_detailed(city: str) -> dict:
    """Returns the current local time, timezone name, and UTC offset for a city.

    Args:
        city: The city name.
    """
    key = city.lower().strip()
    if key in CITY_TIMEZONES:
        tz = ZoneInfo(CITY_TIMEZONES[key])
        now = datetime.datetime.now(tz)
        raw_offset = now.strftime("%z")
        return {
            "status": "success",
            "city": city.title(),
            "local_time": now.strftime("%A, %B %d %Y  %I:%M %p"),
            "timezone": CITY_TIMEZONES[key],
            "utc_offset": f"UTC{raw_offset[:3]}:{raw_offset[3:]}",
            "day_of_week": now.strftime("%A"),
        }
    return {"status": "error", "error_message": f"No timezone for '{city}'. Supported: {SUPPORTED_CITIES_DISPLAY}."}


def get_time_difference(city1: str, city2: str) -> dict:
    """Returns the time difference in hours between two cities.

    Use when the user asks 'what is the time difference between X and Y?'

    Args:
        city1: First city name.
        city2: Second city name.
    """
    k1, k2 = city1.lower().strip(), city2.lower().strip()
    missing = [c for c, k in [(city1, k1), (city2, k2)] if k not in CITY_TIMEZONES]
    if missing:
        return {"status": "error", "error_message": f"No timezone for: {', '.join(missing)}."}
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    off1 = now_utc.astimezone(ZoneInfo(CITY_TIMEZONES[k1])).utcoffset()
    off2 = now_utc.astimezone(ZoneInfo(CITY_TIMEZONES[k2])).utcoffset()
    if off1 is None or off2 is None:
        return {"status": "error", "error_message": "Could not determine UTC offsets."}
    diff = (off2 - off1).total_seconds() / 3600
    return {
        "status": "success",
        "city1": city1.title(), "city2": city2.title(),
        "difference_hours": diff,
        "description": (
            f"{city2.title()} is {abs(diff):.0f}h "
            f"{'ahead of' if diff > 0 else 'behind'} {city1.title()}."
        ),
    }


weather_time_agent = Agent(
    name="weather_time_specialist",
    model=WEATHER_MODEL,
    description=(
        "Expert in weather conditions and world times. Handles: current weather "
        "for one or all cities, UV index, weather comparisons, current local time, "
        "world clock (all cities), UTC offsets, and time differences."
    ),
    instruction=(
        "You are the Weather & Time Specialist — precise, data-driven, friendly.\n\n"
        "Weather tools:\n"
        "  - get_weather_detailed(city)    → full weather for one city\n"
        "  - get_all_weather()             → weather snapshot for all cities\n"
        "  - compare_weather(city1, city2) → side-by-side weather comparison\n\n"
        "Time tools:\n"
        "  - get_time_detailed(city)            → time + timezone + UTC offset\n"
        "  - get_all_times()                    → world clock for all cities\n"
        "  - get_time_difference(city1, city2)  → hours between two cities\n\n"
        "Always use tools. Never fabricate weather or time data.\n"
        f"Supported cities: {SUPPORTED_CITIES_DISPLAY}."
    ),
    tools=[
        get_weather_detailed, get_all_weather, compare_weather,
        get_time_detailed, get_all_times, get_time_difference,
    ],
)


# ===========================================================================
# SPECIALIST 2 — TRAVEL & LIFESTYLE
# ===========================================================================

TRAVEL_TIPS = {
    # North America
    "new york": {
        "best_season": "Spring (Apr–Jun) or Fall (Sep–Nov)",
        "must_see": ["Central Park", "Times Square", "Brooklyn Bridge", "MoMA", "The High Line"],
        "local_tips": "Get a MetroCard for the subway. Book popular restaurants weeks ahead. Tipping 18–20% is standard.",
    },
    "los angeles": {
        "best_season": "Spring (Mar–May) or Fall (Sep–Nov)",
        "must_see": ["Hollywood Walk of Fame", "Getty Museum", "Santa Monica Pier", "Griffith Observatory", "Venice Beach"],
        "local_tips": "Renting a car is almost essential — public transit is limited. FlyAway bus to LAX saves money. Sunscreen year-round.",
    },
    "chicago": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Millennium Park", "Art Institute of Chicago", "Navy Pier", "The 606 Trail", "Willis Tower Skydeck"],
        "local_tips": "Lake Michigan wind makes it feel colder — layer up. The 'L' train covers all major attractions.",
    },
    "toronto": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["CN Tower", "Distillery District", "Kensington Market", "Royal Ontario Museum", "Niagara Falls (day trip)"],
        "local_tips": "PRESTO card for all transit. Very multicultural — amazing food diversity. Niagara Falls is only 1.5 hours away.",
    },
    "mexico city": {
        "best_season": "Autumn/Spring (Oct–Nov, Mar–May)",
        "must_see": ["Zócalo", "Teotihuacán", "Frida Kahlo Museum", "Chapultepec Park", "Mercado de Jamaica"],
        "local_tips": "Metro is very cheap. Try tacos al pastor at street stalls. Altitude is 2,240 m — take it easy the first day.",
    },
    "milwaukee": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Milwaukee Art Museum", "Harley-Davidson Museum", "Third Ward", "Lakefront Brewery", "Mitchell Park Domes"],
        "local_tips": "Very walkable downtown. Famous for Friday fish fries, craft beer, and frozen custard. Summerfest in June.",
    },
    # South America
    "sao paulo": {
        "best_season": "Autumn/Winter (Apr–Aug)",
        "must_see": ["Ibirapuera Park", "Pinacoteca", "Vila Madalena street art", "Mercado Municipal", "MASP"],
        "local_tips": "Uber is reliable and affordable. Try brigadeiro and pão de queijo. Paulistanos eat dinner very late (9–10 pm).",
    },
    "buenos aires": {
        "best_season": "Spring/Autumn (Sep–Nov, Mar–Apr)",
        "must_see": ["Teatro Colón", "La Boca", "Recoleta Cemetery", "San Telmo Market", "Palermo"],
        "local_tips": "Tango shows abound — book ahead. Late nights are standard; restaurants fill up after 9 pm. Try asado and empanadas.",
    },
    # Europe
    "london": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Tower of London", "British Museum", "Hyde Park", "Borough Market", "Tate Modern"],
        "local_tips": "Tap your contactless card on the Tube. Weather changes fast — always carry a jacket. Mind the gap.",
    },
    "paris": {
        "best_season": "Spring (Apr–Jun) or Fall (Sep–Oct)",
        "must_see": ["Eiffel Tower", "Louvre", "Montmartre", "Musée d'Orsay", "Notre-Dame"],
        "local_tips": "Buy a carnet of Metro tickets to save money. Tipping not mandatory but 5–10% is appreciated. Most museums closed Monday or Tuesday.",
    },
    "berlin": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Brandenburg Gate", "Berlin Wall Memorial", "Museum Island", "Reichstag", "Checkpoint Charlie"],
        "local_tips": "Berlin WelcomeCard covers transit and museum discounts. Most museums open Tuesday–Sunday. Street food scene is excellent.",
    },
    "rome": {
        "best_season": "Spring/Autumn (Apr–Jun, Sep–Oct)",
        "must_see": ["Colosseum", "Vatican Museums", "Sistine Chapel", "Trevi Fountain", "Pantheon"],
        "local_tips": "Book Vatican tickets weeks ahead. Validate bus tickets before boarding. No swim shorts or bare shoulders in churches.",
    },
    "barcelona": {
        "best_season": "Spring/Autumn (May–Jun, Sep–Oct)",
        "must_see": ["Sagrada Família", "Park Güell", "La Boqueria", "Gothic Quarter", "Camp Nou"],
        "local_tips": "Book Gaudí sites in advance — they sell out. Watch for pickpockets on Las Ramblas. Late dinners (9–10 pm) are the norm.",
    },
    "amsterdam": {
        "best_season": "Spring/Summer (Apr–Aug)",
        "must_see": ["Rijksmuseum", "Anne Frank House", "Van Gogh Museum", "Canal cruise", "Jordaan district"],
        "local_tips": "Book Anne Frank House online well in advance. Cyclists have absolute priority — watch both ways. GVB day pass for all transit.",
    },
    "vienna": {
        "best_season": "Spring/Autumn (Apr–Jun, Sep–Oct)",
        "must_see": ["Schönbrunn Palace", "Kunsthistorisches Museum", "Belvedere", "Vienna State Opera", "Naschmarkt"],
        "local_tips": "Vienna City Card covers transit and museum discounts. Coffee house culture is unmissable — sit as long as you like.",
    },
    "zurich": {
        "best_season": "Summer (Jun–Sep)",
        "must_see": ["Old Town (Altstadt)", "Lake Zurich", "Kunsthaus", "Rhine Falls (day trip)", "Uetliberg"],
        "local_tips": "Swiss Travel Pass covers trains, trams and buses. Very expensive city — budget accordingly. Chocolate and fondue are staples.",
    },
    "stockholm": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Vasa Museum", "Gamla Stan (Old Town)", "ABBA Museum", "Djurgården island", "Archipelago boat trip"],
        "local_tips": "SL card for all transit. Cash is rarely used — cards accepted everywhere. Midsommar celebrations in June are magical.",
    },
    "istanbul": {
        "best_season": "Spring/Autumn (Apr–May, Sep–Oct)",
        "must_see": ["Hagia Sophia", "Blue Mosque", "Grand Bazaar", "Topkapi Palace", "Bosphorus Cruise"],
        "local_tips": "Istanbulkart for all public transit. Try simit, baklava, and çay (tea). The city straddles Europe and Asia — take the ferry across.",
    },
    "moscow": {
        "best_season": "Summer (Jun–Aug)",
        "must_see": ["Red Square", "Kremlin", "St Basil's Cathedral", "Bolshoi Theatre", "Tretyakov Gallery"],
        "local_tips": "Tap water is safe to drink. Metro is ornate and very efficient. Learn a few Cyrillic letters — it helps enormously.",
    },
    # Africa
    "cairo": {
        "best_season": "Winter (Oct–Mar)",
        "must_see": ["Pyramids of Giza", "Egyptian Museum", "Khan el-Khalili bazaar", "Citadel of Saladin", "Nile Cruise"],
        "local_tips": "Dress conservatively. Bargain at markets — initial prices are always negotiable. Book Pyramid tickets in advance.",
    },
    "lagos": {
        "best_season": "November–January",
        "must_see": ["Nike Art Gallery", "Lekki Conservation Centre", "Tarkwa Bay Beach", "Freedom Park", "Balogun Market"],
        "local_tips": "Traffic (go-slow) can be severe — plan extra time. Try jollof rice and suya at local spots. Bolt/Uber for safe rides.",
    },
    "nairobi": {
        "best_season": "Year-round (Jun–Aug peak)",
        "must_see": ["Nairobi National Park", "Karen Blixen Museum", "Giraffe Centre", "Maasai Market", "Bomas of Kenya"],
        "local_tips": "M-Pesa mobile payment is widely accepted. Great base for Kenya safari. Altitude is 1,700 m — stay hydrated.",
    },
    "johannesburg": {
        "best_season": "Winter (May–Sep)",
        "must_see": ["Apartheid Museum", "Soweto", "Constitution Hill", "Gold Reef City", "Cradle of Humankind"],
        "local_tips": "Use Gautrain for airport and key areas. Hire a guide for Soweto tours. Avoid walking alone at night in unfamiliar areas.",
    },
    # Middle East
    "dubai": {
        "best_season": "Winter (Nov–Mar)",
        "must_see": ["Burj Khalifa", "Dubai Mall", "Palm Jumeirah", "Dubai Frame", "Desert safari"],
        "local_tips": "Dress modestly in public areas and malls. Carry cash for souks. Alcohol is only available in licensed hotel venues.",
    },
    # South Asia
    "mumbai": {
        "best_season": "Winter (Nov–Feb)",
        "must_see": ["Gateway of India", "Elephanta Caves", "Marine Drive", "Dharavi", "Juhu Beach"],
        "local_tips": "Avoid monsoon season (Jun–Sep). Local trains are fast but extremely crowded. Bargain at markets.",
    },
    "new delhi": {
        "best_season": "Winter (Oct–Mar)",
        "must_see": ["Red Fort", "Qutub Minar", "India Gate", "Humayun's Tomb", "Chandni Chowk"],
        "local_tips": "Use the Delhi Metro — it's clean and efficient. Check AQI before outdoor activities. Carry water.",
    },
    # East Asia
    "tokyo": {
        "best_season": "Cherry blossom (Mar–Apr) or Autumn (Oct–Nov)",
        "must_see": ["Shinjuku Gyoen", "Senso-ji", "Shibuya Crossing", "Tsukiji Market", "teamLab Borderless"],
        "local_tips": "Get a Suica card for transit. Carry cash — many places don't take cards. Remove shoes when asked.",
    },
    "beijing": {
        "best_season": "Spring/Autumn (Apr–May, Sep–Oct)",
        "must_see": ["Great Wall", "Forbidden City", "Temple of Heaven", "Summer Palace", "798 Art District"],
        "local_tips": "Download a VPN before arriving — Google, WhatsApp and most social media are blocked. Air quality can be poor in winter.",
    },
    "shanghai": {
        "best_season": "Spring/Autumn (Apr–May, Oct–Nov)",
        "must_see": ["The Bund", "Yu Garden", "Shanghai Tower", "Xintiandi", "Zhujiajiao Water Town"],
        "local_tips": "Metro is world-class and very affordable. Alipay or WeChat Pay needed for most shops — set up before arriving.",
    },
    "seoul": {
        "best_season": "Spring/Autumn (Apr–May, Sep–Oct)",
        "must_see": ["Gyeongbokgung Palace", "Bukchon Hanok Village", "Myeongdong", "N Seoul Tower", "DMZ Tour"],
        "local_tips": "T-money card for metro and buses. Street food is exceptional and very cheap. K-beauty shops everywhere in Myeongdong.",
    },
    "hong kong": {
        "best_season": "Autumn/Winter (Oct–Apr)",
        "must_see": ["Victoria Peak", "Victoria Harbour", "Temple Street Night Market", "Lantau Island / Big Buddha", "Dim Sum in Yum Cha"],
        "local_tips": "Octopus card covers all transit. Dim sum breakfast (yum cha) is essential. Very walkable on Hong Kong Island.",
    },
    # Southeast Asia
    "bangkok": {
        "best_season": "Winter (Nov–Feb)",
        "must_see": ["Grand Palace", "Wat Pho", "Floating markets", "Chatuchak Weekend Market", "Chinatown (Yaowarat)"],
        "local_tips": "BTS Skytrain avoids traffic jams. Dress modestly at temples — cover shoulders and knees. Tuk-tuks — always negotiate price first.",
    },
    "singapore": {
        "best_season": "February–August",
        "must_see": ["Marina Bay Sands", "Gardens by the Bay", "Sentosa Island", "Chinatown", "Hawker centres"],
        "local_tips": "EZ-Link card for MRT and buses. Hawker centres offer incredible food at very low prices. Heavy fines for littering and chewing gum.",
    },
    "kuala lumpur": {
        "best_season": "March–August",
        "must_see": ["Petronas Twin Towers", "Batu Caves", "KL Bird Park", "Central Market", "Merdeka Square"],
        "local_tips": "Grab app for all taxis. Hawker food is excellent and very affordable. Dress modestly at religious sites.",
    },
    "jakarta": {
        "best_season": "June–September",
        "must_see": ["Kota Tua (Old Town)", "National Monument (Monas)", "Thousand Islands", "Ancol Dreamland", "Tanah Abang Market"],
        "local_tips": "Grab or Gojek for rides. Traffic is intense — always allow extra travel time. Try nasi goreng and gado-gado.",
    },
    "manila": {
        "best_season": "November–March",
        "must_see": ["Intramuros", "Rizal Park", "BGC art scene", "Taal Volcano (day trip)", "Palawan (side trip)"],
        "local_tips": "Grab for safe and affordable rides. Jeepney and tricycle for short local trips. Try adobo, sinigang, and halo-halo.",
    },
    # Oceania
    "sydney": {
        "best_season": "Spring/Autumn (Sep–Nov, Mar–Apr)",
        "must_see": ["Opera House", "Harbour Bridge", "Bondi Beach", "Blue Mountains", "Taronga Zoo"],
        "local_tips": "Opal card for all transit. Sunscreen is essential year-round. BYO (bring your own alcohol) restaurants save money.",
    },
}


def get_travel_tips(city: str) -> dict:
    """Returns travel tips, best visiting season, must-see attractions, and local advice.

    Use for trip planning, sightseeing, 'what should I do in X?', or local advice.

    Args:
        city: The destination city.
    """
    key = city.lower().strip()
    if key in TRAVEL_TIPS:
        return {"status": "success", "city": city.title(), **TRAVEL_TIPS[key]}
    return {"status": "error", "error_message": f"No travel data for '{city}'. Supported: {SUPPORTED_CITIES_DISPLAY}."}


def get_packing_advice(city: str) -> dict:
    """Returns a packing list based on current weather in the destination city.

    Use when the user asks what to pack, what to wear, or how to prepare for a trip.

    Args:
        city: The destination city.
    """
    key = city.lower().strip()
    if key not in WEATHER_DATA:
        return {"status": "error", "error_message": f"No weather data for '{city}'."}
    d = WEATHER_DATA[key]
    items = []
    temp = int(d["temp_c"])
    if temp < 5:
        items += ["Heavy winter coat", "Thermal underlayers", "Waterproof boots", "Gloves and hat", "Scarf"]
    elif temp < 15:
        items += ["Warm jacket", "Layers (jumper/sweater)", "Scarf", "Closed-toe shoes"]
    elif temp < 22:
        items += ["Light jacket or cardigan", "Mix of short and long sleeves", "Comfortable walking shoes"]
    else:
        items += ["Light breathable clothing", "Shorts or summer dress", "Sun hat"]
    if temp > 28:
        items.append("Sunscreen SPF 50+")
    if int(d["humidity"].strip("%")) > 70:
        items += ["Moisture-wicking fabrics", "Small umbrella or rain jacket"]
    if d["uv_index"] >= 6:
        items.append("Sunglasses (UV400 protection)")
    items.append("Comfortable walking shoes / trainers")
    items.append("Portable phone charger")
    return {
        "status": "success",
        "city": city.title(),
        "packing_list": items,
        "conditions": f"{d['condition']}, {d['temp_c']}°C, Humidity {d['humidity']}",
    }


travel_agent = Agent(
    name="travel_specialist",
    model=TRAVEL_MODEL,
    description=(
        "Expert travel planner. Handles: trip planning, must-see attractions, local tips, "
        "packing lists, city information (currency, language, population), and "
        "best-month-to-visit recommendations."
    ),
    instruction=(
        "You are the Travel Specialist — knowledgeable, enthusiastic, practical.\n\n"
        "Tools available:\n"
        "  - get_travel_tips(city)            → attractions, season, local advice\n"
        "  - get_packing_advice(city)          → weather-based packing list\n"
        "  - get_city_info(city)               → currency, language, population, dial code\n"
        "  - get_best_cities_for_month(month)  → best destinations for a given month\n\n"
        "Always combine tools when a question spans multiple areas "
        "(e.g. packing + travel tips + city info for a full trip brief).\n"
        f"Supported cities: {SUPPORTED_CITIES_DISPLAY}."
    ),
    tools=[get_travel_tips, get_packing_advice, get_city_info, get_best_cities_for_month],
)


# ===========================================================================
# SPECIALIST 3 — MATH & SCIENCE
# ===========================================================================

_SAFE_MATH_NAMES = {
    k: v for k, v in vars(math).items() if not k.startswith("_")
}
_SAFE_MATH_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})

def calculate(expression: str) -> dict:
    """Evaluates a mathematical expression and returns the result.

    Supports: arithmetic, powers, roots, trig, logarithms, constants (pi, e, tau).
    Examples: '2 ** 10', 'sqrt(144)', 'sin(pi/2)', 'log(1000, 10)', '(15 * 8) / 4'

    Args:
        expression: A mathematical expression as a string.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        # Whitelist-only AST nodes — prevents code injection
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
            ast.Add, ast.Sub, ast.Mul, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.UAdd, ast.USub, ast.Name, ast.Load,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return {"status": "error", "error_message": f"Unsafe expression — '{type(node).__name__}' not allowed."}
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_MATH_NAMES)  # noqa: S307
        return {"status": "success", "expression": expression, "result": result}
    except ZeroDivisionError:
        return {"status": "error", "error_message": "Division by zero."}
    except Exception as e:
        return {"status": "error", "error_message": f"Could not evaluate '{expression}': {e}"}


UNIT_CONVERSIONS: dict = {
    # Length
    ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
    ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
    ("cm", "inches"): 0.393701, ("inches", "cm"): 2.54,
    # Weight
    ("kg", "lbs"): 2.20462, ("lbs", "kg"): 0.453592,
    ("g", "oz"): 0.035274, ("oz", "g"): 28.3495,
    # Temperature handled separately
    # Volume
    ("liters", "gallons"): 0.264172, ("gallons", "liters"): 3.78541,
    ("ml", "fl oz"): 0.033814, ("fl oz", "ml"): 29.5735,
    # Speed
    ("kmh", "mph"): 0.621371, ("mph", "kmh"): 1.60934,
    ("ms", "kmh"): 3.6, ("kmh", "ms"): 0.277778,
    # Area
    ("sqm", "sqft"): 10.7639, ("sqft", "sqm"): 0.092903,
    ("hectares", "acres"): 2.47105, ("acres", "hectares"): 0.404686,
    # Data
    ("mb", "gb"): 0.001, ("gb", "mb"): 1000,
    ("gb", "tb"): 0.001, ("tb", "gb"): 1000,
}


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """Converts a value between units of measurement.

    Supports: length (km/miles/m/ft/cm/inches), weight (kg/lbs/g/oz),
    temperature (celsius/fahrenheit/kelvin), volume (liters/gallons/ml),
    speed (kmh/mph/ms), area (sqm/sqft/hectares/acres), data (mb/gb/tb).

    Args:
        value: The numeric value to convert.
        from_unit: The source unit (e.g. 'km', 'celsius', 'kg').
        to_unit: The target unit (e.g. 'miles', 'fahrenheit', 'lbs').
    """
    f, t = from_unit.lower().strip(), to_unit.lower().strip()

    # Temperature special case
    if f == "celsius" and t == "fahrenheit":
        result = value * 9 / 5 + 32
    elif f == "fahrenheit" and t == "celsius":
        result = (value - 32) * 5 / 9
    elif f == "celsius" and t == "kelvin":
        result = value + 273.15
    elif f == "kelvin" and t == "celsius":
        result = value - 273.15
    elif f == "fahrenheit" and t == "kelvin":
        result = (value - 32) * 5 / 9 + 273.15
    elif f == "kelvin" and t == "fahrenheit":
        result = (value - 273.15) * 9 / 5 + 32
    elif (f, t) in UNIT_CONVERSIONS:
        result = value * UNIT_CONVERSIONS[(f, t)]
    else:
        supported = ", ".join(f"{a}→{b}" for a, b in UNIT_CONVERSIONS)
        return {"status": "error", "error_message": f"Unknown conversion '{f}' → '{t}'. Supported: {supported}."}

    return {
        "status": "success",
        "original": f"{value} {from_unit}",
        "converted": f"{round(result, 6)} {to_unit}",
        "result": round(result, 6),
    }


math_science_agent = Agent(
    name="math_science_specialist",
    model=MATH_MODEL,
    description=(
        "Expert in mathematics, science, and logical reasoning. Handles: arithmetic, "
        "algebra, geometry, calculus concepts, unit conversions, physics and chemistry "
        "Q&A, statistics explanations, and step-by-step problem solving."
    ),
    instruction=(
        "You are the Math & Science Specialist — rigorous, clear, educational.\n\n"
        "Tools:\n"
        "  - calculate(expression)                       → evaluate any math expression\n"
        "  - convert_units(value, from_unit, to_unit)    → convert between units\n\n"
        "For conceptual questions (calculus, physics, chemistry, statistics) answer "
        "directly from your knowledge — tools are for computation only.\n\n"
        "Always show working steps for math problems. Use calculate() to verify answers.\n"
        "Be precise with significant figures and units."
    ),
    tools=[calculate, convert_units],
)


# ===========================================================================
# SPECIALIST 4 — LANGUAGE & WRITING
# ===========================================================================

language_writing_agent = Agent(
    name="language_writing_specialist",
    model=LANGUAGE_MODEL,
    description=(
        "Expert linguist and writing coach. Handles: translation between any languages, "
        "grammar correction, tone adjustment, text summarisation, creative writing, "
        "proofreading, email and essay drafting, paraphrasing, and language learning Q&A."
    ),
    instruction=(
        "You are the Language & Writing Specialist — articulate, creative, precise.\n\n"
        "Capabilities (no tools needed — use your knowledge directly):\n"
        "  - TRANSLATION: Translate text between any two languages accurately.\n"
        "    Always state the source language if not specified.\n"
        "  - GRAMMAR: Correct grammar, spelling, and punctuation. Explain errors.\n"
        "  - TONE: Rewrite text in formal, casual, professional, or persuasive tone.\n"
        "  - SUMMARISE: Condense long text into key points. Match requested length.\n"
        "  - CREATIVE WRITING: Stories, poems, scripts, metaphors, analogies.\n"
        "  - EMAILS & ESSAYS: Draft, improve, or restructure written content.\n"
        "  - PARAPHRASE: Reword text while preserving meaning.\n"
        "  - LANGUAGE LEARNING: Explain grammar rules, vocabulary, pronunciation tips.\n\n"
        "For translations, always provide a brief note on nuances or alternatives "
        "when the source text has ambiguous meaning."
    ),
    tools=[],
)


# ===========================================================================
# SPECIALIST 5 — CODE & DEVELOPMENT
# ===========================================================================

code_agent = Agent(
    name="code_specialist",
    model=CODE_MODEL,
    description=(
        "Expert software engineer. Handles: writing code in any programming language, "
        "explaining code, debugging, code review, architecture design, algorithm "
        "explanation, regex patterns, SQL queries, shell scripts, and DevOps tasks."
    ),
    instruction=(
        "You are the Code Specialist — expert-level software engineer across all languages.\n\n"
        "Capabilities:\n"
        "  - WRITE CODE: Generate clean, well-commented, production-quality code.\n"
        "    Always include error handling and edge cases.\n"
        "  - EXPLAIN CODE: Break down logic line by line when asked.\n"
        "  - DEBUG: Identify bugs, explain root causes, and provide fixes.\n"
        "  - REVIEW: Assess code quality, suggest improvements, spot security issues.\n"
        "  - ALGORITHMS: Explain complexity (Big-O), trade-offs, best approaches.\n"
        "  - ARCHITECTURE: Design patterns, system design, database schemas.\n"
        "  - SQL: Write, optimise, and explain queries.\n"
        "  - REGEX: Build and explain regular expressions.\n"
        "  - SHELL/DEVOPS: Bash scripts, Docker, CI/CD, cloud CLI commands.\n\n"
        "Always specify the language. Use code blocks with syntax highlighting.\n"
        "For complex problems, explain your approach before writing code."
    ),
    tools=[],
)


# ===========================================================================
# SPECIALIST 6 — KNOWLEDGE & RESEARCH
# ===========================================================================

knowledge_agent = Agent(
    name="knowledge_specialist",
    model=KNOWLEDGE_MODEL,
    description=(
        "Expert researcher and knowledge base. Handles: history, science facts, "
        "geography, culture, philosophy, current events context, how-things-work "
        "explanations, comparisons, definitions, and general Q&A on any topic."
    ),
    instruction=(
        "You are the Knowledge & Research Specialist — deeply informed, objective, thorough.\n"
        "Always respond in the same language the user used. Default to English.\n\n"
        "Capabilities:\n"
        "  - HISTORY: Events, timelines, causes and effects, historical figures.\n"
        "  - SCIENCE: Biology, chemistry, physics, astronomy, earth sciences.\n"
        "  - GEOGRAPHY: Countries, capitals, regions, physical geography, demographics.\n"
        "  - CULTURE: Art, music, literature, religion, traditions, food.\n"
        "  - PHILOSOPHY: Schools of thought, key thinkers, ethical frameworks.\n"
        "  - HOW THINGS WORK: Technology, engineering, natural phenomena.\n"
        "  - COMPARISONS: Weigh pros and cons, compare options, explain trade-offs.\n"
        "  - DEFINITIONS: Clear, accurate explanations of terms and concepts.\n"
        "  - RESEARCH SYNTHESIS: Combine information from multiple areas into a coherent answer.\n\n"
        "Be precise about what is established fact vs. interpretation or debate.\n"
        "Cite knowledge boundaries honestly — say when something is uncertain.\n"
        "Structure long answers with clear headings when helpful."
    ),
    tools=[],
)


# ===========================================================================
# SPECIALIST 7 — MEDIA (images & videos)
# ===========================================================================

media_agent = Agent(
    name="media_specialist",
    model=MEDIA_MODEL,
    description=(
        "Expert in image and video processing. Handles: converting photos to video "
        "slideshows, editing images (resize, crop, rotate, brightness, contrast, "
        "grayscale, format conversion, text overlay), editing videos (trim, speed, "
        "reverse, mute, merge), extracting frames from videos, and retrieving "
        "technical metadata from image and video files."
    ),
    instruction=(
        "You are the Media Specialist — precise and practical with images and videos.\n\n"
        "Image tools:\n"
        "  - get_image_info(image_path)         → dimensions, format, EXIF metadata\n"
        "  - edit_image(image_path, ...)         → resize, crop, rotate, brightness,\n"
        "                                          contrast, saturation, grayscale,\n"
        "                                          flip, format conversion\n"
        "  - add_text_to_image(image_path, text) → overlay a caption or watermark\n\n"
        "Video tools:\n"
        "  - photos_to_video(image_paths, ...)   → create slideshow video from images\n"
        "  - get_video_info(video_path)           → duration, fps, resolution, codec\n"
        "  - edit_video(video_path, ...)          → trim, speed, reverse, mute\n"
        "  - extract_video_frames(video_path)     → save frames as images at intervals\n"
        "  - merge_videos(video_paths, ...)       → concatenate multiple clips\n\n"
        "Object detection tools (YOLOv8):\n"
        "  - detect_objects_in_image(image_path) → detect + label all objects, save annotated image\n"
        "  - count_objects(image_path, object_class) → count all or specific objects\n"
        "  - detect_objects_in_video(video_path)  → detect objects across video frames\n\n"
        "Always ask for file paths if the user hasn't provided them.\n"
        "Confirm output path with the user after each operation.\n"
        "Supported input formats: JPEG, PNG, WEBP, TIFF, BMP, MP4, MOV, AVI, MKV.\n"
        "YOLOv8 model sizes: n=fastest, s=fast, m=balanced, l=accurate, x=most accurate."
    ),
    tools=[
        get_image_info, edit_image, add_text_to_image,
        photos_to_video, get_video_info, edit_video,
        extract_video_frames, merge_videos,
        detect_objects_in_image, count_objects, detect_objects_in_video,
    ],
)


# ===========================================================================
# COORDINATOR — root agent
# ===========================================================================

root_agent = Agent(
    name="assistant",
    model=COORD_MODEL,
    description="A powerful general-purpose assistant backed by seven specialist agents.",
    instruction=(
        "You are Assistant — a highly capable AI powered by seven specialist agents.\n\n"
        "LANGUAGE RULE (highest priority):\n"
        "  - Always respond in the SAME language the user writes in.\n"
        "  - If the user writes in English, respond entirely in English.\n"
        "  - Never switch languages unless the user explicitly asks for a translation.\n"
        "  - Do not address the user by any name — you do not know their name.\n\n"
        "Your specialists:\n"
        "  - weather_time_specialist     → weather, UV, world clock, time zones, differences\n"
        "  - travel_specialist           → trip planning, packing, city info, best months\n"
        "  - math_science_specialist     → calculations, unit conversions, science Q&A\n"
        "  - language_writing_specialist → translation, grammar, writing, summarisation\n"
        "  - code_specialist             → code in any language, debugging, architecture\n"
        "  - knowledge_specialist        → history, science facts, geography, research\n"
        "  - media_specialist            → edit images/videos, slideshow from photos,\n"
        "                                  extract frames, merge clips, image metadata\n\n"
        "Routing rules:\n"
        "  - Route EACH part of a multi-domain question to the right specialist.\n"
        "  - For questions spanning multiple domains, call ALL relevant specialists\n"
        "    and synthesise their answers into ONE coherent response.\n"
        "  - For simple greetings or meta questions about yourself, answer directly.\n"
        "  - Never guess data that a specialist can provide — always delegate.\n\n"
        "Tone: confident, clear, friendly. Match the user's level of expertise.\n"
        "If a request is ambiguous, ask one focused clarifying question."
    ),
    sub_agents=[
        weather_time_agent,
        travel_agent,
        math_science_agent,
        language_writing_agent,
        code_agent,
        knowledge_agent,
        media_agent,
    ],
)
