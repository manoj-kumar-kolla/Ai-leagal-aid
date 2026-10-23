"""
tests/test_retriever.py — Unit tests for the BM25 legal retriever.

Run with:  python -m pytest tests/ -v
"""

import pytest
import json
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
#  Helper fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ENTRIES = [
    {
        "act": "Test Act 2024",
        "section": "Section 1",
        "title": "Theft of property",
        "keywords": "theft stolen taking property dishonest",
        "plain_text": "Theft is taking property without consent. Punishment: 3 years.",
    },
    {
        "act": "Test Act 2024",
        "section": "Section 2",
        "title": "Cheating and fraud",
        "keywords": "fraud cheat deceive false promise money",
        "plain_text": "Cheating by deception carries up to 7 years imprisonment.",
    },
    {
        "act": "Test Act 2024",
        "section": "Section 3",
        "title": "Assault and hurt",
        "keywords": "assault beating punch physical attack hurt",
        "plain_text": "Voluntarily causing hurt: up to 1 year or fine.",
    },
]


@pytest.fixture
def knowledge_dir(tmp_path):
    """Create a temp knowledge directory with sample JSON."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "test_laws.json").write_text(
        json.dumps(SAMPLE_ENTRIES), encoding="utf-8"
    )
    return kdir


@pytest.fixture
def retriever_instance(knowledge_dir, monkeypatch):
    """
    Create a LegalRetriever loaded from the temp knowledge directory.
    Monkeypatches settings.KNOWLEDGE_DIR to point at our temp dir.
    """
    from retriever import LegalRetriever
    import config

    monkeypatch.setattr(config.settings, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(config.settings, "RAG_TOP_K", 3)
    return LegalRetriever()


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic_tokenisation(self):
        from retriever import _tokenize
        result = _tokenize("Theft of property")
        assert "theft" in result
        assert "property" in result

    def test_removes_punctuation(self):
        from retriever import _tokenize
        result = _tokenize("Section 498A, IPC!")
        assert "498a" in result or "section" in result

    def test_removes_single_char_tokens(self):
        from retriever import _tokenize
        result = _tokenize("a b c hello world")
        # Single characters should be filtered
        assert "a" not in result
        assert "b" not in result
        assert "hello" in result

    def test_empty_string(self):
        from retriever import _tokenize
        assert _tokenize("") == []

    def test_unicode_handling(self):
        from retriever import _tokenize
        result = _tokenize("भारतीय न्याय संहिता theft")
        # Non-ASCII stripped, English preserved
        assert "theft" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Retriever tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLegalRetriever:
    def test_loads_entries(self, retriever_instance):
        assert retriever_instance.entry_count == len(SAMPLE_ENTRIES)

    def test_is_ready(self, retriever_instance):
        assert retriever_instance.is_ready is True

    def test_retrieve_returns_string(self, retriever_instance):
        result = retriever_instance.retrieve("someone stole my wallet")
        assert isinstance(result, str)

    def test_retrieve_relevant_result(self, retriever_instance):
        result = retriever_instance.retrieve("my property was stolen by a neighbour")
        assert result != ""
        assert "Theft" in result or "theft" in result.lower()

    def test_retrieve_empty_query_returns_empty(self, retriever_instance):
        result = retriever_instance.retrieve("")
        assert result == ""

    def test_retrieve_whitespace_only_returns_empty(self, retriever_instance):
        result = retriever_instance.retrieve("   ")
        assert result == ""

    def test_retrieve_irrelevant_query(self, retriever_instance):
        # A query with no matching tokens should return empty
        result = retriever_instance.retrieve("zzzzxxxxxnonexistentword")
        assert result == ""

    def test_retrieve_top_k_limit(self, retriever_instance, monkeypatch):
        import config
        monkeypatch.setattr(config.settings, "RAG_TOP_K", 1)
        result = retriever_instance.retrieve("theft fraud assault")
        # With top_k=1, only 1 result should appear — check only one header bracket
        sections = result.split("\n[")
        assert len(sections) <= 3  # header + 1 result + footer

    def test_retrieve_truncates_long_query(self, retriever_instance):
        long_query = "theft " * 500  # 3000 chars
        result = retriever_instance.retrieve(long_query)
        # Should not crash
        assert isinstance(result, str)

    def test_retrieve_top_k_override(self, retriever_instance):
        result = retriever_instance.retrieve("theft fraud hurt", top_k=2)
        assert isinstance(result, str)

    def test_format_includes_act_and_title(self, retriever_instance):
        result = retriever_instance.retrieve("cheating fraud money")
        assert "Test Act 2024" in result
        assert "Cheating" in result

    def test_format_has_header_and_footer(self, retriever_instance):
        result = retriever_instance.retrieve("theft property")
        assert "RELEVANT INDIAN LAW SECTIONS" in result
        assert "END OF RETRIEVED SECTIONS" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Entry validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryValidation:
    def test_missing_required_field_skipped(self, tmp_path, monkeypatch):
        """Entry missing 'plain_text' should be skipped silently."""
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        bad_entries = [
            {"act": "Test Act", "section": "1", "title": "No plain text"},  # missing plain_text
            {
                "act": "Good Act", "section": "2", "title": "Good Entry",
                "keywords": "good", "plain_text": "Valid entry."
            },
        ]
        (kdir / "mixed.json").write_text(json.dumps(bad_entries), encoding="utf-8")

        import config
        from retriever import LegalRetriever
        monkeypatch.setattr(config.settings, "KNOWLEDGE_DIR", kdir)
        monkeypatch.setattr(config.settings, "RAG_TOP_K", 3)

        ret = LegalRetriever()
        # Only the good entry should be loaded
        assert ret.entry_count == 1

    def test_corrupt_json_skipped(self, tmp_path, monkeypatch):
        """Corrupt JSON file should not crash the app."""
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        (kdir / "corrupt.json").write_text("{bad json{{", encoding="utf-8")
        (kdir / "good.json").write_text(json.dumps(SAMPLE_ENTRIES), encoding="utf-8")

        import config
        from retriever import LegalRetriever
        monkeypatch.setattr(config.settings, "KNOWLEDGE_DIR", kdir)
        monkeypatch.setattr(config.settings, "RAG_TOP_K", 3)

        ret = LegalRetriever()
        assert ret.entry_count == len(SAMPLE_ENTRIES)

    def test_empty_knowledge_dir(self, tmp_path, monkeypatch):
        """Empty knowledge directory: retriever should be not-ready without crashing."""
        kdir = tmp_path / "empty_knowledge"
        kdir.mkdir()

        import config
        from retriever import LegalRetriever
        monkeypatch.setattr(config.settings, "KNOWLEDGE_DIR", kdir)

        ret = LegalRetriever()
        assert not ret.is_ready
        assert ret.retrieve("theft") == ""

    def test_entries_wrapped_in_entries_key(self, tmp_path, monkeypatch):
        """JSON in {"entries": [...]} format should also load correctly."""
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        wrapped = {"entries": SAMPLE_ENTRIES}
        (kdir / "wrapped.json").write_text(json.dumps(wrapped), encoding="utf-8")

        import config
        from retriever import LegalRetriever
        monkeypatch.setattr(config.settings, "KNOWLEDGE_DIR", kdir)
        monkeypatch.setattr(config.settings, "RAG_TOP_K", 3)

        ret = LegalRetriever()
        assert ret.entry_count == len(SAMPLE_ENTRIES)
