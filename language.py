"""
language.py -- Language detection and multilingual support for Ala.

Detects the language of user input and adapts the system prompt so Ala
responds in the same language. Supports:

    - English (default, no adaptation)
    - Hindi (Devanagari script + Hinglish)
    - Telugu
    - Tamil
    - Bengali
    - Marathi
    - Kannada
    - Malayalam
    - Gujarati
    - Punjabi
    - Odia
    - Urdu

Detection is script-based (Unicode block ranges) with a Hinglish keyword
fallback for Latin-script Hindi. This avoids any external API dependency.

Usage::

    lang = detect_language("Mere landlord ne mujhe ghar se nikal diya")
    # lang.code == "hi", lang.name == "Hindi"

    adapted = adapt_system_prompt(original_prompt, lang)
    # Returns the prompt with a language-instruction prefix.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# -----------------------------------------------------------------------------
#  Language descriptor
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Language:
    """Represents a detected language."""
    code: str              # ISO 639-1 (e.g. "hi", "te", "en")
    name: str              # Human-readable (e.g. "Hindi", "Telugu")
    script: str            # Unicode script name or "Latin"
    response_instruction: str  # What to tell the LLM


ENGLISH = Language(
    code="en",
    name="English",
    script="Latin",
    response_instruction="",   # Default -- no adaptation needed
)

# Map of Language objects keyed by ISO code
_LANGUAGES: dict[str, Language] = {
    "hi": Language(
        code="hi", name="Hindi", script="Devanagari",
        response_instruction=(
            "IMPORTANT: The user has written in Hindi. "
            "You MUST respond entirely in Hindi (Devanagari script). "
            "All section names, legal terms, and headings must also be in Hindi. "
            "Keep the same structured format."
        ),
    ),
    "te": Language(
        code="te", name="Telugu", script="Telugu",
        response_instruction=(
            "IMPORTANT: The user has written in Telugu. "
            "You MUST respond entirely in Telugu script. "
            "Keep the same structured format. Legal terms may be explained in both "
            "Telugu and English in parentheses."
        ),
    ),
    "ta": Language(
        code="ta", name="Tamil", script="Tamil",
        response_instruction=(
            "IMPORTANT: The user has written in Tamil. "
            "You MUST respond entirely in Tamil script. "
            "Keep the same structured format. Legal terms may be explained in both "
            "Tamil and English in parentheses."
        ),
    ),
    "bn": Language(
        code="bn", name="Bengali", script="Bengali",
        response_instruction=(
            "IMPORTANT: The user has written in Bengali. "
            "You MUST respond entirely in Bengali script. "
            "Keep the same structured format."
        ),
    ),
    "mr": Language(
        code="mr", name="Marathi", script="Devanagari",
        response_instruction=(
            "IMPORTANT: The user has written in Marathi. "
            "You MUST respond entirely in Marathi (Devanagari script). "
            "Keep the same structured format."
        ),
    ),
    "kn": Language(
        code="kn", name="Kannada", script="Kannada",
        response_instruction=(
            "IMPORTANT: The user has written in Kannada. "
            "You MUST respond entirely in Kannada script. "
            "Keep the same structured format."
        ),
    ),
    "ml": Language(
        code="ml", name="Malayalam", script="Malayalam",
        response_instruction=(
            "IMPORTANT: The user has written in Malayalam. "
            "You MUST respond entirely in Malayalam script. "
            "Keep the same structured format."
        ),
    ),
    "gu": Language(
        code="gu", name="Gujarati", script="Gujarati",
        response_instruction=(
            "IMPORTANT: The user has written in Gujarati. "
            "You MUST respond entirely in Gujarati script. "
            "Keep the same structured format."
        ),
    ),
    "pa": Language(
        code="pa", name="Punjabi", script="Gurmukhi",
        response_instruction=(
            "IMPORTANT: The user has written in Punjabi. "
            "You MUST respond entirely in Punjabi (Gurmukhi script). "
            "Keep the same structured format."
        ),
    ),
    "or": Language(
        code="or", name="Odia", script="Odia",
        response_instruction=(
            "IMPORTANT: The user has written in Odia. "
            "You MUST respond entirely in Odia script. "
            "Keep the same structured format."
        ),
    ),
    "ur": Language(
        code="ur", name="Urdu", script="Arabic",
        response_instruction=(
            "IMPORTANT: The user has written in Urdu. "
            "You MUST respond entirely in Urdu (Nastaliq script). "
            "Keep the same structured format."
        ),
    ),
}


# -----------------------------------------------------------------------------
#  Unicode block ranges for script detection
# -----------------------------------------------------------------------------

# Each entry: (start_codepoint, end_codepoint, language_code)
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    # Devanagari -- shared by Hindi and Marathi; distinguished by word list below
    (0x0900, 0x097F, "hi"),   # Devanagari
    (0x0900, 0x097F, "mr"),   # same block; handled separately
    # South Indian
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
    # East Indian
    (0x0980, 0x09FF, "bn"),   # Bengali
    (0x0B00, 0x0B7F, "or"),   # Odia
    # West Indian
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    # North Indian
    (0x0A00, 0x0A7F, "pa"),   # Gurmukhi (Punjabi)
    # Urdu / Arabic
    (0x0600, 0x06FF, "ur"),   # Arabic script (Urdu)
]

# Marathi-specific common words (to distinguish from Hindi in Devanagari)
_MARATHI_WORDS = frozenset(
    "माझ्या माझा माझे आहे आहेत नाही करा केला मला तुम्ही आपण".split()
)

# Hinglish keywords -- common Hindi words written in Latin script
_HINGLISH_KEYWORDS = frozenset(
    "mera meri mujhe mere hamara hamare unka unki unhe "
    "nahi haan kya kyun kaise kab kahan kitna "
    "police thana fir case ghar paisa rupaye "
    "landlord makaan malik kiraya rent "
    "garibi adalat vakil "
    "court judge judge bail "
    "bhai behan maa baap "
    "bharat india desh sarkaar".split()
)

# Minimum fraction of non-ASCII characters to trigger script detection
_MIN_SCRIPT_FRACTION = 0.25


# -----------------------------------------------------------------------------
#  Detection
# -----------------------------------------------------------------------------

def detect_language(text: str) -> Language:
    """
    Detect the primary language of *text*.

    Strategy:
      1. Count characters in each Unicode script block.
      2. If the dominant script block has >= 25% of total characters, use it.
      3. For Devanagari, apply a Marathi word list to distinguish from Hindi.
      4. If all characters are ASCII, try Hinglish keyword matching.
      5. Default to English.

    Args:
        text: The raw user input string.

    Returns:
        A :class:`Language` instance. Never raises.
    """
    if not text or not text.strip():
        return ENGLISH

    # Count characters per script code
    script_counts: dict[str, int] = {}
    total_chars = 0

    for ch in text:
        cp = ord(ch)
        if ch.isspace() or unicodedata.category(ch) in ("Po", "Ps", "Pe", "Pi", "Pf"):
            continue
        total_chars += 1
        for start, end, code in _SCRIPT_RANGES:
            if start <= cp <= end:
                script_counts[code] = script_counts.get(code, 0) + 1
                break

    if total_chars == 0:
        return ENGLISH

    if script_counts:
        # Find dominant script
        dominant_code = max(script_counts, key=lambda k: script_counts[k])
        dominant_count = script_counts[dominant_code]

        if dominant_count / total_chars >= _MIN_SCRIPT_FRACTION:
            # Devanagari: distinguish Hindi vs Marathi
            if dominant_code in ("hi", "mr"):
                words = set(text.split())
                if words & _MARATHI_WORDS:
                    return _LANGUAGES["mr"]
                return _LANGUAGES["hi"]

            lang = _LANGUAGES.get(dominant_code)
            if lang:
                return lang

    # Fallback: Hinglish detection (Latin-script Hindi)
    words_lower = set(text.lower().split())
    hinglish_matches = words_lower & _HINGLISH_KEYWORDS
    if len(hinglish_matches) >= 2:
        return _LANGUAGES["hi"]

    return ENGLISH


# -----------------------------------------------------------------------------
#  Prompt adaptation
# -----------------------------------------------------------------------------

def adapt_system_prompt(base_prompt: str, lang: Language) -> str:
    """
    Prepend a language instruction to *base_prompt* when *lang* is not English.

    The instruction is placed at the very top so the LLM sees it first.

    Args:
        base_prompt: The original system prompt string.
        lang:        Detected language from :func:`detect_language`.

    Returns:
        Modified system prompt. Returns *base_prompt* unchanged for English.
    """
    if not lang.response_instruction:
        return base_prompt
    return lang.response_instruction + "\n\n" + base_prompt


def format_language_note(lang: Language) -> str:
    """
    Return a short terminal note to display when a non-English language is detected.
    Returns an empty string for English.
    """
    if lang.code == "en":
        return ""
    return f"  [Language detected: {lang.name} — Ala will respond in {lang.name}]"
