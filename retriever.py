"""
retriever.py — BM25-based RAG retriever for Ala.

Loads all JSON files from the knowledge/ folder, indexes every law entry
with BM25, and retrieves the most relevant chunks for any user query.
No embedding API needed — runs 100% locally.
"""

import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

from config import settings


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class LegalRetriever:
    def __init__(self):
        self._chunks: list[dict] = []   # raw records
        self._corpus: list[list[str]] = []  # tokenized texts for BM25
        self._bm25: BM25Okapi | None = None
        self._load_knowledge()

    # ── Loading ────────────────────────────────────────────────────────

    def _load_knowledge(self) -> None:
        """Read every .json file in the knowledge directory and index it."""
        knowledge_dir: Path = settings.KNOWLEDGE_DIR

        if not knowledge_dir.exists():
            print(f"  [RAG] Warning: knowledge directory not found at {knowledge_dir}")
            return

        json_files = list(knowledge_dir.glob("*.json"))
        if not json_files:
            print(f"  [RAG] Warning: no JSON files found in {knowledge_dir}")
            return

        for filepath in json_files:
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._chunks.extend(data)
                elif isinstance(data, dict) and "entries" in data:
                    self._chunks.extend(data["entries"])
            except Exception as e:
                print(f"  [RAG] Could not load {filepath.name}: {e}")

        if not self._chunks:
            return

        # Build the searchable text for each chunk and tokenize
        for chunk in self._chunks:
            searchable = self._build_searchable_text(chunk)
            self._corpus.append(_tokenize(searchable))

        self._bm25 = BM25Okapi(self._corpus)
        print(f"  [RAG] Indexed {len(self._chunks)} legal entries from {len(json_files)} file(s).")

    @staticmethod
    def _build_searchable_text(chunk: dict) -> str:
        """Combine relevant fields into one searchable string."""
        parts = [
            chunk.get("title", ""),
            chunk.get("act", ""),
            chunk.get("section", ""),
            chunk.get("keywords", ""),
            chunk.get("summary", ""),
            chunk.get("plain_text", ""),
        ]
        return " ".join(str(p) for p in parts if p)

    # ── Retrieval ──────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """
        Retrieve the most relevant law sections for a query.
        Returns a formatted string ready to inject into the LLM prompt.
        """
        if self._bm25 is None or not self._chunks:
            return ""

        k = top_k or settings.RAG_TOP_K
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Filter out zero-score results
        results = [self._chunks[i] for i in top_indices if scores[i] > 0]

        if not results:
            return ""

        return self._format_results(results)

    @staticmethod
    def _format_results(chunks: list[dict]) -> str:
        """Format retrieved chunks into a clean context block for the LLM."""
        lines = ["--- RELEVANT INDIAN LAW SECTIONS (Retrieved) ---"]
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


# Singleton — created once, reused across all queries
retriever = LegalRetriever()
