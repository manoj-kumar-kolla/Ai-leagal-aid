"""
retriever.py — BM25-based RAG retriever for Ala.

Loads all JSON files from the knowledge/ folder, indexes every law entry
with BM25, and retrieves the most relevant chunks for any user query.
No embedding API needed — runs 100% locally.

Security / robustness notes:
- All file reads are wrapped in try/except so a corrupt JSON cannot crash the app.
- Query tokens are sanitised before BM25 scoring.
- A minimum score threshold filters truly irrelevant results.
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from config import settings

logger = logging.getLogger(__name__)

# Minimum BM25 score below which results are discarded as irrelevant.
_MIN_SCORE: float = 0.0


def _tokenize(text: str) -> list[str]:
    """
    Lowercase, strip all non-alphanumeric characters, and split into tokens.
    Empty tokens and single-character noise are removed.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _validate_entry(entry: object, filepath: str) -> Optional[dict]:
    """
    Validate that a knowledge-base entry has the expected structure.
    Returns the entry dict if valid, else None (and logs a warning).
    """
    if not isinstance(entry, dict):
        logger.warning("[RAG] Skipping non-dict entry in %s", filepath)
        return None
    required = ("act", "section", "title", "plain_text")
    if not all(k in entry for k in required):
        missing = [k for k in required if k not in entry]
        logger.warning("[RAG] Entry in %s missing keys: %s", filepath, missing)
        return None
    return entry


class LegalRetriever:
    """
    BM25-based legal document retriever.

    On construction, every JSON file under ``settings.KNOWLEDGE_DIR`` is
    loaded and indexed.  Call :meth:`retrieve` to get a formatted context
    block ready for injection into the LLM prompt.
    """

    def __init__(self) -> None:
        self._chunks: list[dict] = []          # raw validated records
        self._corpus: list[list[str]] = []     # tokenised texts for BM25
        self._bm25: Optional[BM25Okapi] = None
        self._load_knowledge()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_knowledge(self) -> None:
        """Read every .json file in the knowledge directory and build the BM25 index."""
        knowledge_dir: Path = settings.KNOWLEDGE_DIR

        if not knowledge_dir.exists():
            logger.warning("[RAG] Knowledge directory not found: %s", knowledge_dir)
            print(f"  [RAG] Warning: knowledge directory not found at {knowledge_dir}")
            return

        json_files = sorted(knowledge_dir.glob("*.json"))
        if not json_files:
            logger.warning("[RAG] No JSON files found in %s", knowledge_dir)
            print(f"  [RAG] Warning: no JSON files found in {knowledge_dir}")
            return

        for filepath in json_files:
            self._load_file(filepath)

        if not self._chunks:
            logger.warning("[RAG] Knowledge base is empty after loading all files.")
            return

        # Tokenise each chunk and build BM25 index
        for chunk in self._chunks:
            searchable = self._build_searchable_text(chunk)
            self._corpus.append(_tokenize(searchable))

        self._bm25 = BM25Okapi(self._corpus)
        print(
            f"  [RAG] Indexed {len(self._chunks)} legal entries "
            f"from {len(json_files)} file(s)."
        )

    def _load_file(self, filepath: Path) -> None:
        """Parse one JSON file and add valid entries to self._chunks."""
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("[RAG] JSON parse error in %s: %s", filepath.name, exc)
            print(f"  [RAG] Could not parse {filepath.name}: {exc}")
            return
        except OSError as exc:
            logger.error("[RAG] Cannot read %s: %s", filepath.name, exc)
            print(f"  [RAG] Cannot read {filepath.name}: {exc}")
            return

        # Support both a bare list and {"entries": [...]} wrapper
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and "entries" in data:
            entries = data["entries"]
        else:
            logger.warning("[RAG] %s has unexpected top-level structure; skipping.", filepath.name)
            return

        loaded = 0
        for entry in entries:
            validated = _validate_entry(entry, filepath.name)
            if validated is not None:
                self._chunks.append(validated)
                loaded += 1

        logger.debug("[RAG] Loaded %d entries from %s", loaded, filepath.name)

    @staticmethod
    def _build_searchable_text(chunk: dict) -> str:
        """Combine all text fields of a chunk into one searchable string."""
        parts = [
            chunk.get("title", ""),
            chunk.get("act", ""),
            chunk.get("section", ""),
            chunk.get("keywords", ""),
            chunk.get("summary", ""),
            chunk.get("plain_text", ""),
        ]
        return " ".join(str(p) for p in parts if p)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve the most relevant law sections for *query*.

        Returns a formatted string ready to inject into the LLM prompt,
        or an empty string when the index is unavailable or no results match.

        Args:
            query:  The user's plain-language description of their situation.
            top_k:  Override the default ``RAG_TOP_K`` setting.

        Returns:
            Formatted context block (str) or ``""`` if nothing useful was found.
        """
        if self._bm25 is None or not self._chunks:
            return ""

        # Sanitise and tokenise the query
        query = query.strip()
        if not query:
            return ""

        # Truncate excessively long queries to prevent score distortion
        if len(query) > 2000:
            query = query[:2000]

        k = top_k if top_k is not None else settings.RAG_TOP_K
        tokens = _tokenize(query)
        if not tokens:
            return ""

        scores = self._bm25.get_scores(tokens)

        # Sort by score descending, keep only positive-scoring results
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = [
            self._chunks[i]
            for i in top_indices
            if scores[i] > _MIN_SCORE
        ]

        if not results:
            return ""

        return self._format_results(results)

    @staticmethod
    def _format_results(chunks: list[dict]) -> str:
        """Format retrieved chunks into a clean context block for the LLM."""
        lines = ["--- RELEVANT INDIAN LAW SECTIONS (Retrieved by RAG) ---"]
        for chunk in chunks:
            act     = chunk.get("act", "")
            section = chunk.get("section", "")
            title   = chunk.get("title", "")
            text    = chunk.get("plain_text", chunk.get("summary", ""))

            header = " | ".join(filter(None, [act, section, title]))
            lines.append(f"\n[{header}]")
            if text:
                lines.append(text.strip())
        lines.append("\n--- END OF RETRIEVED SECTIONS ---")
        return "\n".join(lines)

    # ── Diagnostics ────────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        """Number of knowledge-base entries currently indexed."""
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        """True when the BM25 index is built and at least one entry is indexed."""
        return self._bm25 is not None and len(self._chunks) > 0


# Singleton — created once at import time, reused across all queries
retriever: LegalRetriever = LegalRetriever()
