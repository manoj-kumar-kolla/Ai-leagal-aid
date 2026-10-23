"""
tests/test_config.py — Unit tests for the configuration module.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSettingsDefaults:
    """Settings loaded via the real .env — just check types and ranges."""

    def test_groq_api_key_is_string(self):
        from config import settings
        assert isinstance(settings.GROQ_API_KEY, str)
        assert len(settings.GROQ_API_KEY) > 0

    def test_groq_model_is_string(self):
        from config import settings
        assert isinstance(settings.GROQ_MODEL, str)
        assert len(settings.GROQ_MODEL) > 0

    def test_temperature_in_range(self):
        from config import settings
        assert 0.0 <= settings.TEMPERATURE <= 2.0

    def test_max_tokens_positive(self):
        from config import settings
        assert settings.MAX_TOKENS >= 256

    def test_rag_top_k_positive(self):
        from config import settings
        assert settings.RAG_TOP_K >= 1

    def test_knowledge_dir_is_path(self):
        from config import settings
        assert isinstance(settings.KNOWLEDGE_DIR, Path)

    def test_max_history_turns_valid(self):
        from config import settings
        assert settings.MAX_HISTORY_TURNS >= 2

    def test_max_input_chars_valid(self):
        from config import settings
        assert settings.MAX_INPUT_CHARS >= 100

    def test_app_version_string(self):
        from config import settings
        assert isinstance(settings.APP_VERSION, str)
        assert settings.APP_VERSION != ""

    def test_base_dir_exists(self):
        from config import settings
        assert settings.BASE_DIR.exists()


class TestRequire:
    """Test the _require function in isolation."""

    def test_missing_key_raises_environment_error(self, monkeypatch):
        monkeypatch.delenv("SOME_NONEXISTENT_KEY_XYZ", raising=False)
        from config import _require
        with pytest.raises(EnvironmentError) as exc_info:
            _require("SOME_NONEXISTENT_KEY_XYZ")
        assert "SOME_NONEXISTENT_KEY_XYZ" in str(exc_info.value)

    def test_empty_value_raises_environment_error(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_ALA", "")
        from config import _require
        with pytest.raises(EnvironmentError):
            _require("TEST_KEY_ALA")

    def test_whitespace_only_raises_environment_error(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_ALA", "   ")
        from config import _require
        with pytest.raises(EnvironmentError):
            _require("TEST_KEY_ALA")

    def test_valid_key_returns_value(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_ALA", "valid_value")
        from config import _require
        assert _require("TEST_KEY_ALA") == "valid_value"


class TestGetFloat:
    def test_valid_float(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "0.5")
        from config import _get_float
        assert _get_float("TEST_FLOAT", 0.2) == 0.5

    def test_uses_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT_MISSING", raising=False)
        from config import _get_float
        assert _get_float("TEST_FLOAT_MISSING", 0.3) == 0.3

    def test_out_of_range_raises(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "5.0")
        from config import _get_float
        with pytest.raises(EnvironmentError):
            _get_float("TEST_FLOAT", 0.2, 0.0, 2.0)

    def test_invalid_string_raises(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "notanumber")
        from config import _get_float
        with pytest.raises(EnvironmentError):
            _get_float("TEST_FLOAT", 0.2)


class TestGetInt:
    def test_valid_int(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "10")
        from config import _get_int
        assert _get_int("TEST_INT", 5) == 10

    def test_uses_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_INT_MISSING", raising=False)
        from config import _get_int
        assert _get_int("TEST_INT_MISSING", 7) == 7

    def test_float_string_raises(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "3.5")
        from config import _get_int
        with pytest.raises(EnvironmentError):
            _get_int("TEST_INT", 5)

    def test_out_of_range_raises(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "0")
        from config import _get_int
        with pytest.raises(EnvironmentError):
            _get_int("TEST_INT", 5, min_val=1, max_val=100)
