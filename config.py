"""
config.py — Central configuration for Ala.
All settings and secrets are loaded here. The rest of the app never
touches environment variables or file paths directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the directory this file lives in
_BASE_DIR = Path(__file__).parent

# Load .env from the project root (same folder as this file)
load_dotenv(_BASE_DIR / ".env")


def _require(key: str) -> str:
    """Read a required env var. Raises a clear error if missing."""
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"\n  [Config Error] Required environment variable '{key}' is not set.\n"
            f"  Create a file named '.env' in:\n"
            f"    {_BASE_DIR}\n"
            f"  and add:  {key}=your_value_here\n"
            f"  Get a free Groq key at: https://console.groq.com\n"
        )
    return value


class Settings:
    # ── API ──────────────────────────────────────────────────
    GROQ_API_KEY: str = _require("GROQ_API_KEY")
    GROQ_MODEL:   str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    # ── Model behaviour ──────────────────────────────────────
    TEMPERATURE:  float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS:   int   = int(os.getenv("MAX_TOKENS",   "4096"))

    # ── RAG ──────────────────────────────────────────────────
    KNOWLEDGE_DIR: Path = _BASE_DIR / "knowledge"
    RAG_TOP_K:     int  = int(os.getenv("RAG_TOP_K", "5"))   # chunks to retrieve

    # ── App ──────────────────────────────────────────────────
    APP_NAME:    str = "Ala"
    APP_TAGLINE: str = "AI Legal Assistant (India)"


# Single shared instance — import this everywhere
settings = Settings()
