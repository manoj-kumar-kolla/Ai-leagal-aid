"""
config.py — Central configuration for Ala.
All settings and secrets are loaded here. The rest of the app never
touches environment variables or file paths directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the directory this file lives in (project root)
_BASE_DIR: Path = Path(__file__).parent

# Load .env from the project root (same folder as this file)
load_dotenv(_BASE_DIR / ".env")


def _require(key: str) -> str:
    """Read a required env var. Raises EnvironmentError with clear guidance if missing."""
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"\n  [Config Error] Required environment variable '{key}' is not set.\n"
            f"  Create a file named '.env' in:\n"
            f"    {_BASE_DIR}\n"
            f"  and add:  {key}=your_value_here\n"
            f"  Get a free Groq key at: https://console.groq.com\n"
            f"  (Copy .env.example to .env and fill in your key.)\n"
        )
    return value


def _get_float(key: str, default: float, min_val: float = 0.0, max_val: float = 2.0) -> float:
    """Read a float env var with bounds-checking."""
    raw = os.getenv(key, str(default)).strip()
    try:
        val = float(raw)
    except ValueError:
        raise EnvironmentError(
            f"[Config Error] '{key}' must be a number, got: '{raw}'"
        )
    if not (min_val <= val <= max_val):
        raise EnvironmentError(
            f"[Config Error] '{key}' must be between {min_val} and {max_val}, got: {val}"
        )
    return val


def _get_int(key: str, default: int, min_val: int = 1, max_val: int = 100_000) -> int:
    """Read an integer env var with bounds-checking."""
    raw = os.getenv(key, str(default)).strip()
    try:
        val = int(raw)
    except ValueError:
        raise EnvironmentError(
            f"[Config Error] '{key}' must be an integer, got: '{raw}'"
        )
    if not (min_val <= val <= max_val):
        raise EnvironmentError(
            f"[Config Error] '{key}' must be between {min_val} and {max_val}, got: {val}"
        )
    return val


class Settings:
    """
    All application settings sourced from environment variables.
    Access via the module-level ``settings`` singleton.
    """

    # ── API ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str  = _require("GROQ_API_KEY")
    GROQ_MODEL:   str  = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip() or "qwen/qwen3.8-27b"

    # ── Model behaviour ──────────────────────────────────────────────────
    TEMPERATURE:  float = _get_float("TEMPERATURE", 0.2, 0.0, 2.0)
    MAX_TOKENS:   int   = _get_int("MAX_TOKENS", 4096, 256, 32_768)

    # ── RAG ──────────────────────────────────────────────────────────────
    KNOWLEDGE_DIR: Path = _BASE_DIR / "knowledge"
    RAG_TOP_K:     int  = _get_int("RAG_TOP_K", 5, 1, 20)

    # ── Conversation safety limits ────────────────────────────────────────
    # Maximum number of conversation turns kept in memory (user + assistant pairs).
    # Older turns are pruned to prevent excessively long context windows.
    MAX_HISTORY_TURNS: int = _get_int("MAX_HISTORY_TURNS", 20, 2, 100)

    # Maximum characters accepted in a single user input.
    MAX_INPUT_CHARS:   int = _get_int("MAX_INPUT_CHARS", 4000, 100, 20_000)

    # ── App identity ──────────────────────────────────────────────────────
    APP_NAME:    str = "Ala"
    APP_TAGLINE: str = "AI Legal Assistant (India)"
    APP_VERSION: str = "2.0.0"

    # ── Paths ─────────────────────────────────────────────────────────────
    BASE_DIR: Path = _BASE_DIR

    # ── IndianKanoon (optional) ───────────────────────────────────────────
    # Token for live case law search. Leave blank to disable.
    # Get a free trial (Rs.500 credit) at https://api.indiankanoon.org/
    INDIANKANOON_API_TOKEN: str = os.getenv("INDIANKANOON_API_TOKEN", "").strip()

    # ── Language detection ────────────────────────────────────────────────
    # Set to "false" to disable auto-detection and always respond in English.
    LANGUAGE_DETECTION_ENABLED: bool = (
        os.getenv("LANGUAGE_DETECTION_ENABLED", "true").strip().lower() != "false"
    )


# Single shared instance — import this everywhere
settings: Settings = Settings()
