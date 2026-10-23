"""
tests/test_analyzer.py — Unit tests for the prompt-building module.

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import analyzer


class TestBuildAnalysisPrompt:
    def test_basic_situation(self):
        result = analyzer.build_analysis_prompt("My landlord locked me out")
        assert "My landlord locked me out" in result

    def test_rag_context_prepended(self):
        result = analyzer.build_analysis_prompt(
            situation="theft",
            rag_context="--- RELEVANT INDIAN LAW SECTIONS ---\n[Test Law]\nSome text",
        )
        # RAG context should come before user situation
        rag_pos = result.find("RELEVANT INDIAN LAW SECTIONS")
        sit_pos = result.find("theft")
        assert rag_pos < sit_pos

    def test_attached_content_included(self):
        result = analyzer.build_analysis_prompt(
            situation="I uploaded a document",
            attached_content="Contract clause 5 says payment is due in 30 days",
        )
        assert "Contract clause 5" in result
        assert "UPLOADED FILE CONTENT" in result

    def test_empty_situation(self):
        result = analyzer.build_analysis_prompt("")
        assert isinstance(result, str)

    def test_no_rag_no_attachment(self):
        result = analyzer.build_analysis_prompt("simple question")
        assert "simple question" in result
        assert "RELEVANT" not in result
        assert "UPLOADED" not in result


class TestBuildDocumentAnalysisPrompt:
    def test_document_name_included(self):
        result = analyzer.build_document_analysis_prompt(
            document_content="This is a legal notice",
            document_name="notice.pdf",
            document_type="PDF",
        )
        assert "notice.pdf" in result

    def test_document_content_included(self):
        result = analyzer.build_document_analysis_prompt(
            document_content="Pay Rs. 50,000 within 15 days",
            document_name="demand.pdf",
            document_type="PDF",
        )
        assert "Pay Rs. 50,000" in result

    def test_rag_context_prepended(self):
        result = analyzer.build_document_analysis_prompt(
            document_content="contract text",
            document_name="contract.pdf",
            document_type="PDF",
            rag_context="--- RELEVANT SECTIONS ---",
        )
        rag_pos = result.find("RELEVANT SECTIONS")
        doc_pos = result.find("contract text")
        assert rag_pos < doc_pos

    def test_user_context_appended(self):
        result = analyzer.build_document_analysis_prompt(
            document_content="notice text",
            document_name="n.pdf",
            document_type="PDF",
            user_context="This was sent to me by my employer",
        )
        assert "This was sent to me by my employer" in result

    def test_document_analysis_instruction_present(self):
        result = analyzer.build_document_analysis_prompt(
            document_content="text",
            document_name="file.pdf",
            document_type="PDF",
        )
        assert "Document Analysis Format" in result


class TestBuildMediaAnalysisPrompt:
    def test_media_name_included(self):
        result = analyzer.build_media_analysis_prompt(
            media_content="[Audio Transcript]\nThreat was made",
            media_name="evidence.mp4",
            media_type="Video",
        )
        assert "evidence.mp4" in result

    def test_transcript_included(self):
        result = analyzer.build_media_analysis_prompt(
            media_content="[Audio Transcript]\nI will hurt you",
            media_name="clip.mp4",
            media_type="Video",
        )
        assert "I will hurt you" in result

    def test_video_instruction_for_video(self):
        result = analyzer.build_media_analysis_prompt(
            media_content="frames",
            media_name="v.mp4",
            media_type="Video",
        )
        assert "video evidence" in result.lower()

    def test_audio_instruction_for_audio(self):
        result = analyzer.build_media_analysis_prompt(
            media_content="transcript",
            media_name="a.mp3",
            media_type="Audio",
        )
        assert "audio recording" in result.lower()


class TestSystemPrompts:
    def test_system_prompt_not_empty(self):
        assert len(analyzer.SYSTEM_PROMPT) > 100

    def test_document_system_prompt_not_empty(self):
        assert len(analyzer.DOCUMENT_SYSTEM_PROMPT) > 100

    def test_get_system_prompt_general(self):
        result = analyzer.get_system_prompt_for_mode("general")
        assert result == analyzer.SYSTEM_PROMPT

    def test_get_system_prompt_document(self):
        result = analyzer.get_system_prompt_for_mode("document")
        assert result == analyzer.DOCUMENT_SYSTEM_PROMPT

    def test_get_system_prompt_default(self):
        result = analyzer.get_system_prompt_for_mode("unknown_mode")
        assert result == analyzer.SYSTEM_PROMPT

    def test_system_prompt_contains_rules(self):
        # Critical rules must be in the system prompt
        prompt = analyzer.SYSTEM_PROMPT
        assert "Never guarantee" in prompt or "never guarantee" in prompt.lower()
        assert "lying" in prompt.lower() or "lie" in prompt.lower()
        assert "NALSA" in prompt

    def test_document_system_prompt_contains_document_sections(self):
        prompt = analyzer.DOCUMENT_SYSTEM_PROMPT
        assert "DOCUMENT OVERVIEW" in prompt
        assert "WEAKNESSES" in prompt
        assert "DEADLINES" in prompt


class TestTruncateContent:
    def test_short_content_unchanged(self):
        text = "short text"
        result = analyzer.truncate_content(text, max_chars=100)
        assert result == text

    def test_long_content_truncated(self):
        text = "a" * 500
        result = analyzer.truncate_content(text, max_chars=100)
        assert len(result) <= 250  # 100 + truncation note
        assert "truncated" in result.lower()

    def test_exact_limit_unchanged(self):
        text = "a" * 100
        result = analyzer.truncate_content(text, max_chars=100)
        assert result == text

    def test_truncation_note_mentions_original_length(self):
        text = "b" * 1000
        result = analyzer.truncate_content(text, max_chars=100)
        assert "1000" in result
