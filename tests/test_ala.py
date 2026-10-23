"""
tests/test_ala.py -- Unit tests for ala.py core functions.

Tests cover:
  - Input sanitisation (_sanitise): whitespace, unicode, control chars, length capping
  - History pruning (_prune_history): enforces MAX_HISTORY_TURNS * 2
  - Error mapping (_handle_api_error): 401, 429, timeout, model errors
  - Supported extensions list sanity
"""

import pytest
from unittest.mock import patch, MagicMock

import ala
from config import settings


# ─────────────────────────────────────────────────────────────────────────────
#  _sanitise
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitise:
    def test_strip_whitespace(self):
        assert ala._sanitise("   hello world   ") == "hello world"

    def test_collapse_internal_spaces(self):
        assert ala._sanitise("hello    multiple   spaces") == "hello multiple spaces"

    def test_collapse_excessive_newlines(self):
        assert ala._sanitise("para 1\n\n\n\n\npara 2") == "para 1\n\npara 2"

    def test_strip_control_characters(self):
        raw = "hello\x00\x07world\x1f"
        assert ala._sanitise(raw) == "helloworld"

    def test_preserves_newlines_and_normalises_tabs(self):
        raw = "line 1\nline 2\tindented"
        assert ala._sanitise(raw) == "line 1\nline 2 indented"

    def test_nfc_normalisation(self):
        # Combining character vs precomposed
        decomposed = "e\u0301"  # e + acute accent
        precomposed = "\u00e9"  # é
        assert ala._sanitise(decomposed) == precomposed

    def test_truncates_long_input(self):
        long_text = "a" * (settings.MAX_INPUT_CHARS + 500)
        result = ala._sanitise(long_text)
        assert len(result) == settings.MAX_INPUT_CHARS

    def test_empty_string(self):
        assert ala._sanitise("") == ""
        assert ala._sanitise("   \n\t   ") == ""


# ─────────────────────────────────────────────────────────────────────────────
#  _prune_history
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneHistory:
    def test_short_history_unchanged(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert ala._prune_history(history) == history

    def test_exact_limit_unchanged(self):
        max_messages = settings.MAX_HISTORY_TURNS * 2
        history = [{"role": "user", "content": f"msg {i}"} for i in range(max_messages)]
        assert len(ala._prune_history(history)) == max_messages

    def test_excess_history_pruned(self):
        max_messages = settings.MAX_HISTORY_TURNS * 2
        history = [{"role": "user", "content": f"msg {i}"} for i in range(max_messages + 10)]
        pruned = ala._prune_history(history)
        assert len(pruned) == max_messages
        # Ensure the most recent messages are kept
        assert pruned[-1]["content"] == f"msg {max_messages + 9}"


# ─────────────────────────────────────────────────────────────────────────────
#  _handle_api_error
# ─────────────────────────────────────────────────────────────────────────────

class TestHandleApiError:
    def test_auth_error_hints_api_key(self, capsys):
        ala._handle_api_error(Exception("401 Unauthorized invalid api_key"))
        out = capsys.readouterr().out
        assert "API key" in out or "invalid" in out

    def test_rate_limit_error(self, capsys):
        ala._handle_api_error(Exception("429 rate_limit exceeded"))
        out = capsys.readouterr().out
        assert "Rate limit" in out

    def test_timeout_error(self, capsys):
        ala._handle_api_error(Exception("connection timeout error"))
        out = capsys.readouterr().out
        assert "Network" in out or "connection" in out

    def test_service_unavailable(self, capsys):
        ala._handle_api_error(Exception("503 service_unavailable"))
        out = capsys.readouterr().out
        assert "unavailable" in out or "Try again" in out


# ─────────────────────────────────────────────────────────────────────────────
#  Supported extensions sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_pdf_included(self):
        assert ".pdf" in ala.ALL_UPLOAD_EXTS

    def test_images_included(self):
        assert {".jpg", ".jpeg", ".png", ".webp"}.issubset(ala.ALL_UPLOAD_EXTS)

    def test_media_included(self):
        assert {".mp4", ".mp3", ".wav"}.issubset(ala.ALL_UPLOAD_EXTS)
