"""
config.py

Central configuration file.
Every setting that might need to change later (model name, limits, etc.)
lives here so you never have to hunt through multiple files.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# ---------------------------------------------------------
# API KEYS (never hardcode these — they come from .env)
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------
# GEMINI SETTINGS
# ---------------------------------------------------------
# Change the model here ONLY — nowhere else in the project.
GEMINI_MODEL = "gemini-3.5-flash"

# Keep Gemini's answer reasonably short to save tokens (fallback default).
MAX_OUTPUT_TOKENS = 12000

# ---------------------------------------------------------
# RESEARCH MODES
# Each mode controls the response length and how deep Gemini
# should go. Used by services/gemini_service.py.
# ---------------------------------------------------------
DEFAULT_MODE = "quick"

RESEARCH_MODES = {
    "quick": {
        "label": "Quick Summary",
        "icon": "⚡",
        "max_output_tokens": 1000,
        "depth_instruction": (
            "Be brief and to the point. Keep every section short — "
            "1-3 sentences or up to 3 bullet points per section."
        ),
    },
    "deep": {
        "label": "Deep Research",
        "icon": "🔎",
        "max_output_tokens": 2200,
        "depth_instruction": (
            "Be thorough. Cover multiple angles, compare what different "
            "sources say, and go deeper into the details each source provides."
        ),
    },
    "simple": {
        "label": "Explain Simply",
        "icon": "🎓",
        "max_output_tokens": 3000,
        "depth_instruction": (
            "Explain everything in simple, beginner-friendly language. "
            "Avoid technical jargon, and explain any necessary term in plain words."
        ),
    },
}

# ---------------------------------------------------------
# TAVILY / SEARCH SETTINGS
# ---------------------------------------------------------
# How many search results Tavily should fetch.
MAX_SEARCH_RESULTS = 10

# How many characters of each source's content we keep
# before sending it to Gemini (keeps token usage low).
MAX_CONTENT_LENGTH = 10000