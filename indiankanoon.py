"""
indiankanoon.py — IndianKanoon API client for live case law search.

Provides optional enrichment of Ala's RAG context with real court judgements
sourced directly from the Indian Kanoon legal database.

Authentication:
    Set INDIANKANOON_API_TOKEN in your .env file.
    Get a token at: https://api.indiankanoon.org/
    Rs.500 free trial credit is provided on signup.
    Non-commercial projects may apply for Rs.10,000/month free credit.

If the token is absent or a request fails, the module degrades gracefully:
    - retrieve() returns an empty string (no crash, no noise).
    - The BM25 RAG context remains the only legal reference.

API pricing (pay-as-you-go):
    Search:            Rs.0.50 / request
    Document fetch:    Rs.0.20 / request
    Document fragment: Rs.0.05 / request
    Doc meta-info:     Rs.0.02 / request
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import so the rest of the app does not fail if `requests` is missing
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.debug("[IK] 'requests' package not installed -- IndianKanoon disabled.")


# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------

_BASE_URL    = "https://api.indiankanoon.org"
_TIMEOUT_SEC = 8        # Hard timeout per HTTP request
_MAX_RESULTS = 3        # Max case summaries to include in context
_MAX_SNIPPET = 800      # Characters from each document snippet


# -----------------------------------------------------------------------------
#  Client
# -----------------------------------------------------------------------------

class IndianKanoonClient:
    """
    Thin wrapper around the IndianKanoon REST API.

    Usage::

        client = IndianKanoonClient()          # reads token from env
        context = client.retrieve("cheque bounce dishonour NI Act")
        # context is a formatted string ready for prompt injection, or ""

    All methods return empty strings / None on failure -- never raise to caller.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self._token: str = (token or os.getenv("INDIANKANOON_API_TOKEN", "")).strip()
        self._enabled: bool = bool(self._token) and _REQUESTS_AVAILABLE
        if not self._token:
            logger.debug(
                "[IK] INDIANKANOON_API_TOKEN not set -- live case law search disabled."
            )
        elif not _REQUESTS_AVAILABLE:
            logger.debug("[IK] 'requests' package missing -- live case law search disabled.")

    # ---- Public API ---------------------------------------------------------

    def retrieve(self, query: str, max_results: int = _MAX_RESULTS) -> str:
        """
        Search IndianKanoon for cases relevant to *query* and return a
        formatted context block ready for LLM prompt injection.

        Returns:
            A non-empty string when results are found, otherwise "".
        """
        if not self._enabled or not query.strip():
            return ""

        results = self._search(query.strip(), max_results)
        if not results:
            return ""

        snippets: list[str] = []
        for item in results:
            doc_id  = item.get("tid", "")
            title   = item.get("title", "Unknown case")
            court   = item.get("docsource", "")
            date    = item.get("publishdate", "")
            snippet = self._get_snippet(doc_id, query) if doc_id else ""

            header_parts = [title]
            if court:
                header_parts.append(court)
            if date:
                header_parts.append(date)
            header = " | ".join(header_parts)

            entry = f"[{header}]"
            if snippet:
                entry += f"\n{snippet}"
            snippets.append(entry)

        if not snippets:
            return ""

        block = (
            "--- RELEVANT CASE LAW (Live from IndianKanoon) ---\n"
            + "\n\n".join(snippets)
            + "\n--- END OF CASE LAW ---"
        )
        return block

    @property
    def is_enabled(self) -> bool:
        """True when a token is configured and the requests library is available."""
        return self._enabled

    # ---- Internal helpers ---------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._token}"}

    def _search(self, query: str, max_results: int) -> list[dict]:
        """
        Hit the /search/ endpoint and return a list of result dicts.
        Returns [] on any failure.
        """
        url = f"{_BASE_URL}/search/"
        params = {"formInput": query, "pagenum": 0}

        try:
            resp = _requests.post(
                url,
                data=params,
                headers=self._auth_headers(),
                timeout=_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("docs", [])
            return docs[:max_results]

        except Exception as exc:  # noqa: BLE001
            logger.warning("[IK] Search failed for '%s': %s", query[:60], exc)
            return []

    def _get_snippet(self, doc_id: str, query: str) -> str:
        """
        Fetch a short document fragment from IndianKanoon.
        Returns an empty string on failure.
        """
        url = f"{_BASE_URL}/docfragment/{doc_id}/"
        params = {"formInput": query}

        try:
            resp = _requests.post(
                url,
                data=params,
                headers=self._auth_headers(),
                timeout=_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()

            # The fragment field can be called "fragment" or "doc"
            fragment = data.get("fragment") or data.get("doc") or ""

            # Strip any residual HTML tags
            fragment = re.sub(r"<[^>]+>", " ", fragment)
            fragment = re.sub(r"\s{2,}", " ", fragment).strip()

            return fragment[:_MAX_SNIPPET] if fragment else ""

        except Exception as exc:  # noqa: BLE001
            logger.debug("[IK] Fragment fetch failed for doc %s: %s", doc_id, exc)
            return ""


# -----------------------------------------------------------------------------
#  Module-level singleton (reads token from env at import time)
# -----------------------------------------------------------------------------

kanoon: IndianKanoonClient = IndianKanoonClient()
