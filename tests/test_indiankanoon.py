"""
tests/test_indiankanoon.py -- Unit tests for indiankanoon.py

Tests cover:
  - Client disabled when token is absent
  - Client disabled when requests is missing (simulated)
  - retrieve() returns empty string when disabled
  - _search() error handling
  - _get_snippet() error handling and HTML stripping
  - Full retrieve() flow with mocked HTTP responses
  - is_enabled property
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(token: str = "fake-token", requests_available: bool = True):
    """Create an IndianKanoonClient with controlled environment."""
    with patch.dict("os.environ", {"INDIANKANOON_API_TOKEN": token}):
        with patch("indiankanoon._REQUESTS_AVAILABLE", requests_available):
            from indiankanoon import IndianKanoonClient
            return IndianKanoonClient()


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_disabled_when_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            from indiankanoon import IndianKanoonClient
            client = IndianKanoonClient(token="")
        assert not client.is_enabled

    def test_disabled_when_requests_unavailable(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", False):
            from indiankanoon import IndianKanoonClient
            client = IndianKanoonClient(token="fake-token")
        assert not client.is_enabled

    def test_enabled_with_token_and_requests(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", True):
            from indiankanoon import IndianKanoonClient
            client = IndianKanoonClient(token="fake-token")
        assert client.is_enabled


# ---------------------------------------------------------------------------
# retrieve() when disabled
# ---------------------------------------------------------------------------

class TestRetrieveDisabled:
    def test_returns_empty_when_disabled(self):
        from indiankanoon import IndianKanoonClient
        client = IndianKanoonClient(token="")
        result = client.retrieve("cheque bounce NI act")
        assert result == ""

    def test_returns_empty_for_empty_query(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", True):
            from indiankanoon import IndianKanoonClient
            client = IndianKanoonClient(token="fake-token")
        # Even with token, empty query returns ""
        result = client.retrieve("   ")
        assert result == ""


# ---------------------------------------------------------------------------
# _search() error handling
# ---------------------------------------------------------------------------

class TestSearchErrorHandling:
    def test_network_error_returns_empty(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", True):
            from indiankanoon import IndianKanoonClient
            client = IndianKanoonClient(token="fake-token")

        with patch("indiankanoon._requests") as mock_req:
            mock_req.post.side_effect = ConnectionError("Network unreachable")
            result = client._search("cheque bounce", 3)

        assert result == []

    def test_http_error_returns_empty(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", True):
            from indiankanoon import IndianKanoonClient
            client = IndianKanoonClient(token="fake-token")

        with patch("indiankanoon._requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")
            mock_req.post.return_value = mock_resp
            result = client._search("cheque bounce", 3)

        assert result == []

    def test_malformed_json_returns_empty(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", True):
            from indiankanoon import IndianKanoonClient
            client = IndianKanoonClient(token="fake-token")

        with patch("indiankanoon._requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.side_effect = ValueError("Invalid JSON")
            mock_req.post.return_value = mock_resp
            result = client._search("cheque bounce", 3)

        assert result == []


# ---------------------------------------------------------------------------
# _get_snippet() — HTML stripping
# ---------------------------------------------------------------------------

class TestGetSnippet:
    def _client(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", True):
            from indiankanoon import IndianKanoonClient
            return IndianKanoonClient(token="fake-token")

    def test_html_tags_stripped(self):
        client = self._client()
        html_fragment = (
            "<b>The cheque was dishonoured</b> under <em>Section 138</em> "
            "of the <a href='#'>Negotiable Instruments Act</a>."
        )
        with patch("indiankanoon._requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"fragment": html_fragment}
            mock_req.post.return_value = mock_resp

            result = client._get_snippet("12345", "cheque bounce")

        assert "<b>" not in result
        assert "<em>" not in result
        assert "Section 138" in result
        assert "cheque was dishonoured" in result

    def test_network_error_returns_empty(self):
        client = self._client()
        with patch("indiankanoon._requests") as mock_req:
            mock_req.post.side_effect = TimeoutError("Timeout")
            result = client._get_snippet("12345", "query")
        assert result == ""

    def test_snippet_truncated_to_max(self):
        from indiankanoon import _MAX_SNIPPET
        client = self._client()
        long_text = "A" * (_MAX_SNIPPET + 500)

        with patch("indiankanoon._requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"fragment": long_text}
            mock_req.post.return_value = mock_resp

            result = client._get_snippet("12345", "query")

        assert len(result) <= _MAX_SNIPPET

    def test_empty_fragment_returns_empty(self):
        client = self._client()
        with patch("indiankanoon._requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"fragment": ""}
            mock_req.post.return_value = mock_resp

            result = client._get_snippet("12345", "query")

        assert result == ""


# ---------------------------------------------------------------------------
# Full retrieve() flow with mocked responses
# ---------------------------------------------------------------------------

class TestRetrieveFullFlow:
    def _client(self):
        with patch("indiankanoon._REQUESTS_AVAILABLE", True):
            from indiankanoon import IndianKanoonClient
            return IndianKanoonClient(token="fake-token")

    def _mock_search_response(self, docs: list) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"docs": docs}
        return mock_resp

    def _mock_fragment_response(self, text: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"fragment": text}
        return mock_resp

    def test_returns_formatted_context_block(self):
        client = self._client()
        docs = [
            {"tid": "111", "title": "Ramesh v State", "docsource": "Supreme Court", "publishdate": "2020-01-01"},
        ]
        snippet = "The cheque was returned unpaid by the bank on 15th March 2020."

        with patch("indiankanoon._requests") as mock_req:
            mock_req.post.side_effect = [
                self._mock_search_response(docs),
                self._mock_fragment_response(snippet),
            ]
            result = client.retrieve("cheque bounce NI Act 138")

        assert "IndianKanoon" in result
        assert "Ramesh v State" in result
        assert "Supreme Court" in result
        assert snippet in result

    def test_no_results_returns_empty(self):
        client = self._client()
        with patch("indiankanoon._requests") as mock_req:
            mock_req.post.return_value = self._mock_search_response([])
            result = client.retrieve("some very obscure query nobody would ever ask")
        assert result == ""

    def test_respects_max_results(self):
        from indiankanoon import _MAX_RESULTS
        client = self._client()

        # Create more docs than _MAX_RESULTS
        many_docs = [
            {"tid": str(i), "title": f"Case {i}", "docsource": "HC", "publishdate": "2023"}
            for i in range(_MAX_RESULTS + 5)
        ]

        # All fragment calls return the same simple text
        simple_frag = MagicMock()
        simple_frag.raise_for_status.return_value = None
        simple_frag.json.return_value = {"fragment": "short text"}

        search_resp = self._mock_search_response(many_docs)

        with patch("indiankanoon._requests") as mock_req:
            # First call is search, all subsequent are fragments
            mock_req.post.side_effect = [search_resp] + [simple_frag] * (_MAX_RESULTS + 5)
            result = client.retrieve("property dispute")

        # Count how many case headers appear
        import re
        case_headers = re.findall(r"\[Case \d+", result)
        assert len(case_headers) <= _MAX_RESULTS

    def test_case_without_snippet_still_included(self):
        client = self._client()
        docs = [{"tid": "999", "title": "No Snippet Case", "docsource": "", "publishdate": ""}]

        with patch("indiankanoon._requests") as mock_req:
            search_resp = self._mock_search_response(docs)
            frag_resp = MagicMock()
            frag_resp.raise_for_status.return_value = None
            frag_resp.json.return_value = {"fragment": ""}

            mock_req.post.side_effect = [search_resp, frag_resp]
            result = client.retrieve("property dispute")

        # Even without a snippet the case title should appear
        assert "No Snippet Case" in result

    def test_search_failure_returns_empty(self):
        client = self._client()
        with patch("indiankanoon._requests") as mock_req:
            mock_req.post.side_effect = ConnectionError("down")
            result = client.retrieve("any query")
        assert result == ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def test_singleton_exists(self):
        from indiankanoon import kanoon, IndianKanoonClient
        assert isinstance(kanoon, IndianKanoonClient)

    def test_singleton_is_disabled_without_token(self):
        """In the test environment INDIANKANOON_API_TOKEN is likely not set."""
        import os
        if os.getenv("INDIANKANOON_API_TOKEN"):
            pytest.skip("Token is set in environment — skipping disabled test")
        from indiankanoon import kanoon
        assert not kanoon.is_enabled
