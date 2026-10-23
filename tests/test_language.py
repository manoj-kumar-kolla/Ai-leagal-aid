"""
tests/test_language.py -- Unit tests for language.py

Tests cover:
  - Script-based detection for all supported languages
  - Hinglish (Latin-script Hindi) detection
  - Devanagari disambiguation (Hindi vs Marathi)
  - English default fallback
  - adapt_system_prompt behaviour
  - format_language_note behaviour
  - Edge cases (empty string, punctuation-only, very short input)
"""

import pytest
from language import (
    detect_language,
    adapt_system_prompt,
    format_language_note,
    ENGLISH,
    Language,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_lang(text: str, expected_code: str) -> None:
    result = detect_language(text)
    assert result.code == expected_code, (
        f"Expected {expected_code!r}, got {result.code!r} for input: {text!r}"
    )


# ---------------------------------------------------------------------------
# English default
# ---------------------------------------------------------------------------

class TestEnglishDefault:
    def test_plain_english_returns_english(self):
        assert_lang("My landlord locked me out of my room.", "en")

    def test_empty_string_returns_english(self):
        assert_lang("", "en")

    def test_whitespace_only_returns_english(self):
        assert_lang("   \n\t  ", "en")

    def test_punctuation_only_returns_english(self):
        assert_lang("!!! ??? ...", "en")

    def test_numbers_only_returns_english(self):
        assert_lang("12345 678 90", "en")

    def test_legal_english_query(self):
        assert_lang(
            "My employer has not paid my salary for three months. What can I do?",
            "en",
        )


# ---------------------------------------------------------------------------
# Hindi — Devanagari script
# ---------------------------------------------------------------------------

class TestHindi:
    def test_devanagari_hindi(self):
        assert_lang("मेरे मकान मालिक ने मुझे घर से निकाल दिया।", "hi")

    def test_devanagari_hindi_2(self):
        assert_lang("पुलिस ने बिना वजह गिरफ्तार किया।", "hi")

    def test_mixed_devanagari_dominant(self):
        # Mostly Hindi with some English words — should still be Hindi
        assert_lang("मुझे FIR दर्ज करनी है, police नहीं मान रही।", "hi")


# ---------------------------------------------------------------------------
# Hinglish (Hindi in Latin script)
# ---------------------------------------------------------------------------

class TestHinglish:
    def test_hinglish_basic(self):
        assert_lang("mujhe ghar se nikal diya landlord ne", "hi")

    def test_hinglish_more_keywords(self):
        assert_lang("mere paas koi vakil nahi hai. kya karu?", "hi")

    def test_hinglish_police_context(self):
        assert_lang("police ne mera paisa le liya aur FIR nahi ki", "hi")

    def test_single_hinglish_word_returns_english(self):
        # Only 1 match — below threshold of 2
        assert_lang("My nahi situation is unclear", "en")


# ---------------------------------------------------------------------------
# Marathi (Devanagari, but distinct word set)
# ---------------------------------------------------------------------------

class TestMarathi:
    def test_marathi_detected(self):
        assert_lang("माझ्या मालकाने मला घरातून हाकलले आहे.", "mr")

    def test_marathi_with_common_word(self):
        assert_lang("माझा कामगार नाही आहे.", "mr")


# ---------------------------------------------------------------------------
# South Indian scripts
# ---------------------------------------------------------------------------

class TestSouthIndian:
    def test_telugu(self):
        assert_lang("నా యజమాని నన్ను ఇంటి నుండి వెళ్ళగొట్టాడు.", "te")

    def test_tamil(self):
        assert_lang("என் வீட்டு உரிமையாளர் என்னை வெளியேற்றினார்.", "ta")

    def test_kannada(self):
        assert_lang("ನನ್ನ ಮಾಲೀಕ ನನ್ನನ್ನು ಮನೆಯಿಂದ ಹೊರಹಾಕಿದ್ದಾರೆ.", "kn")

    def test_malayalam(self):
        assert_lang("എന്റെ വീട്ടുടമ എന്നെ ഒഴിപ്പിച്ചു.", "ml")


# ---------------------------------------------------------------------------
# East Indian scripts
# ---------------------------------------------------------------------------

class TestEastIndian:
    def test_bengali(self):
        assert_lang("আমার বাড়িওয়ালা আমাকে বাড়ি থেকে বের করে দিয়েছেন।", "bn")

    def test_odia(self):
        # Odia script sample: "my landlord evicted me"
        assert_lang("ମୋ ଘର ମାଲିକ ମୋତେ ଘର ଛାଡ଼ିବାକୁ କହିଲେ।", "or")


# ---------------------------------------------------------------------------
# West / North Indian scripts
# ---------------------------------------------------------------------------

class TestOtherScripts:
    def test_gujarati(self):
        assert_lang("મારા મકાનમાલિકે મને ઘરમાંથી કાઢી મૂક્યો.", "gu")

    def test_punjabi(self):
        assert_lang("ਮੇਰੇ ਮਕਾਨ ਮਾਲਕ ਨੇ ਮੈਨੂੰ ਘਰੋਂ ਕੱਢ ਦਿੱਤਾ।", "pa")

    def test_urdu(self):
        assert_lang("میرے مکان مالک نے مجھے گھر سے نکال دیا۔", "ur")


# ---------------------------------------------------------------------------
# adapt_system_prompt
# ---------------------------------------------------------------------------

class TestAdaptSystemPrompt:
    BASE = "You are Ala, an AI legal assistant."

    def test_english_returns_unchanged(self):
        result = adapt_system_prompt(self.BASE, ENGLISH)
        assert result == self.BASE

    def test_hindi_prepends_instruction(self):
        from language import _LANGUAGES
        hindi = _LANGUAGES["hi"]
        result = adapt_system_prompt(self.BASE, hindi)
        assert result.startswith(hindi.response_instruction)
        assert self.BASE in result

    def test_telugu_prepends_instruction(self):
        from language import _LANGUAGES
        telugu = _LANGUAGES["te"]
        result = adapt_system_prompt(self.BASE, telugu)
        assert result.startswith(telugu.response_instruction)

    def test_separator_between_instruction_and_prompt(self):
        from language import _LANGUAGES
        hindi = _LANGUAGES["hi"]
        result = adapt_system_prompt(self.BASE, hindi)
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# format_language_note
# ---------------------------------------------------------------------------

class TestFormatLanguageNote:
    def test_english_returns_empty(self):
        note = format_language_note(ENGLISH)
        assert note == ""

    def test_hindi_returns_non_empty(self):
        from language import _LANGUAGES
        note = format_language_note(_LANGUAGES["hi"])
        assert "Hindi" in note
        assert len(note) > 0

    def test_telugu_contains_telugu(self):
        from language import _LANGUAGES
        note = format_language_note(_LANGUAGES["te"])
        assert "Telugu" in note

    def test_all_languages_have_notes(self):
        from language import _LANGUAGES
        for code, lang in _LANGUAGES.items():
            note = format_language_note(lang)
            assert len(note) > 0, f"Empty note for language code: {code}"


# ---------------------------------------------------------------------------
# Language dataclass
# ---------------------------------------------------------------------------

class TestLanguageDataclass:
    def test_language_is_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            ENGLISH.code = "xx"  # type: ignore[misc]

    def test_english_has_no_instruction(self):
        assert ENGLISH.response_instruction == ""

    def test_english_code(self):
        assert ENGLISH.code == "en"
