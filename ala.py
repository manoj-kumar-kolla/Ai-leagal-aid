"""
ala.py — Main entry point for Ala, the AI Legal Assistant for India.
"""

from __future__ import annotations

import io
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

# ── Force UTF-8 on Windows terminals ──────────────────────────────────────────
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from groq import Groq

from config import settings
from retriever import retriever
from indiankanoon import kanoon
from language import detect_language, adapt_system_prompt, format_language_note
import analyzer
from document_processor import (
    ALL_SUPPORTED_EXTS as DOC_EXTS,
    process_file,
    describe_file,
    MAX_UPLOAD_BYTES,
)
from media_processor import (
    ALL_MEDIA_EXTS as MEDIA_EXTS,
    process_media,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# All file extensions the system can handle
ALL_UPLOAD_EXTS: set[str] = DOC_EXTS | MEDIA_EXTS

# ─────────────────────────────────────────────────────────────────────────────
#  Input sanitisation
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise(raw: str) -> str:
    """
    Clean user input:
      1. Strip leading/trailing whitespace.
      2. NFC normalise Unicode.
      3. Remove ASCII control chars except tab / newline.
      4. Collapse excessive whitespace.
      5. Enforce MAX_INPUT_CHARS limit.
    """
    text = raw.strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > settings.MAX_INPUT_CHARS:
        text = text[: settings.MAX_INPUT_CHARS]
        _note(
            f"Your input was truncated to {settings.MAX_INPUT_CHARS} characters. "
            "Please split long descriptions into multiple messages."
        )
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  Conversation history management
# ─────────────────────────────────────────────────────────────────────────────

def _prune_history(history: list[dict]) -> list[dict]:
    """Keep only the most recent MAX_HISTORY_TURNS pairs."""
    max_messages = settings.MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  LLM streaming
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    client: Groq,
    history: list[dict],
    system_prompt: Optional[str] = None,
) -> str:
    """
    Stream a response from Groq.

    Strips Qwen3 <think>...</think> blocks on the fly.
    Returns the full raw response (think tags included) for history storage.

    Args:
        client:        Groq client.
        history:       Conversation history (already contains the latest user message).
        system_prompt: Override the default system prompt (e.g. for document mode).

    Raises:
        Exception — re-raised after printing a user-friendly message.
    """
    prompt = system_prompt or analyzer.SYSTEM_PROMPT

    stream = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            *history,
        ],
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        stream=True,
    )

    print("-" * _W)
    print("  ALA:")
    print("-" * _W)
    print()

    full_response: str = ""
    in_think: bool = False
    buf: str = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        full_response += delta
        buf += delta

        # Stream-strip <think> tags
        while True:
            if in_think:
                end = buf.find("</think>")
                if end == -1:
                    buf = ""
                    break
                in_think = False
                buf = buf[end + len("</think>"):]
            else:
                start = buf.find("<think>")
                if start == -1:
                    print(buf, end="", flush=True)
                    buf = ""
                    break
                print(buf[:start], end="", flush=True)
                buf = buf[start + len("<think>"):]
                in_think = True

    if buf and not in_think:
        print(buf, end="", flush=True)
    print()
    return full_response


# ─────────────────────────────────────────────────────────────────────────────
#  Error handling
# ─────────────────────────────────────────────────────────────────────────────

def _handle_api_error(exc: Exception) -> None:
    """Map Groq / network exceptions to user-friendly messages."""
    msg = str(exc)
    _error(f"Could not reach the AI service: {msg}")
    if "401" in msg or "api_key" in msg.lower() or "authentication" in msg.lower():
        _hint("Your API key appears invalid. Check GROQ_API_KEY in your .env file.")
        _hint("Get a free key at: https://console.groq.com")
    elif "429" in msg or "rate_limit" in msg.lower():
        _hint("Rate limit reached. Wait a moment and try again.")
    elif "model" in msg.lower() and ("not found" in msg.lower() or "invalid" in msg.lower()):
        _hint(f"Model '{settings.GROQ_MODEL}' may not be available on your Groq plan.")
        _hint("Set GROQ_MODEL in your .env to a valid model name.")
    elif "timeout" in msg.lower() or "connection" in msg.lower():
        _hint("Network issue. Check your internet connection.")
    elif "503" in msg or "service_unavailable" in msg.lower():
        _hint("Groq service is temporarily unavailable. Try again in a minute.")
    else:
        _hint("Check your internet connection and .env configuration.")


# ─────────────────────────────────────────────────────────────────────────────
#  File upload handling
# ─────────────────────────────────────────────────────────────────────────────

def _handle_upload(
    raw_path: str,
    client: Groq,
    conversation_history: list[dict],
    user_context: str = "",
) -> None:
    """
    Process an uploaded file and run the appropriate analysis.

    Modifies *conversation_history* in-place (appends user + assistant turns).
    """
    path = Path(raw_path.strip().strip('"').strip("'"))

    # ── Validate ─────────────────────────────────────────────────────────
    if not path.exists():
        _error(f"File not found: {path}")
        _hint("Check the path and try again. Use forward slashes or double backslashes on Windows.")
        return

    if not path.is_file():
        _error(f"That is a directory, not a file: {path}")
        return

    ext = path.suffix.lower()
    if ext not in ALL_UPLOAD_EXTS:
        _error(f"Unsupported file type: '{ext}'")
        _hint(f"Supported document types: {', '.join(sorted(DOC_EXTS))}")
        _hint(f"Supported media types:    {', '.join(sorted(MEDIA_EXTS))}")
        return

    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        mb = size / (1024 * 1024)
        _error(f"File too large: {mb:.1f} MB (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")
        return

    print()
    print(f"  Processing: {describe_file(path)}")
    print("  This may take a moment…")
    print()

    # ── Extract content ───────────────────────────────────────────────────
    if ext in MEDIA_EXTS:
        try:
            media_result = process_media(path, client)
        except (FileNotFoundError, ValueError) as exc:
            _error(str(exc))
            return
        except Exception as exc:
            logger.exception("Unexpected media processing error")
            _error(f"Media processing failed unexpectedly: {exc}")
            return

        if media_result.error and media_result.is_empty:
            _error(f"Could not process media file: {media_result.error}")
            return
        if media_result.error:
            _note(f"Partial processing (some steps failed): {media_result.error}")

        query_text = user_context or path.stem.replace("_", " ")
        rag_context = retriever.retrieve(query_text)
        kanoon_context = kanoon.retrieve(query_text)
        combined_context = "\n\n".join(
            part for part in (rag_context, kanoon_context) if part
        )
        content = analyzer.truncate_content(
            media_result.full_content, label=f"{media_result.media_type} content"
        )
        user_msg = analyzer.build_media_analysis_prompt(
            media_content=content,
            media_name=path.name,
            media_type=media_result.media_type,
            rag_context=combined_context,
            user_context=user_context,
        )
        sys_prompt = analyzer.SYSTEM_PROMPT

    else:
        # Document or image
        try:
            doc_result = process_file(path, groq_client=client)
        except (FileNotFoundError, ValueError) as exc:
            _error(str(exc))
            return
        except Exception as exc:
            logger.exception("Unexpected document processing error")
            _error(f"Document processing failed unexpectedly: {exc}")
            return

        if doc_result.error and doc_result.is_empty:
            _error(f"Could not extract content from file: {doc_result.error}")
            return
        if doc_result.error:
            _note(f"Partial extraction: {doc_result.error}")

        query_text = user_context or doc_result.raw_text[:500] or path.stem.replace("_", " ")
        rag_context = retriever.retrieve(query_text)
        kanoon_context = kanoon.retrieve(query_text)
        combined_context = "\n\n".join(
            part for part in (rag_context, kanoon_context) if part
        )
        content = analyzer.truncate_content(
            doc_result.full_content, label=f"{doc_result.file_type} content"
        )

        # Images use general analysis mode; text documents use document mode
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}:
            user_msg = analyzer.build_analysis_prompt(
                situation=user_context or "Please analyse this uploaded image.",
                rag_context=combined_context,
                attached_content=content,
            )
            sys_prompt = analyzer.SYSTEM_PROMPT
        else:
            user_msg = analyzer.build_document_analysis_prompt(
                document_content=content,
                document_name=path.name,
                document_type=doc_result.file_type,
                rag_context=combined_context,
                user_context=user_context,
            )
            sys_prompt = analyzer.DOCUMENT_SYSTEM_PROMPT

    # ── Language detection for upload context ─────────────────────────────
    if user_context:
        lang = detect_language(user_context)
        lang_note = format_language_note(lang)
        if lang_note:
            print(lang_note)
        sys_prompt = adapt_system_prompt(sys_prompt, lang)

    # ── Send to LLM ───────────────────────────────────────────────────────
    conversation_history.append({"role": "user", "content": user_msg})
    conversation_history[:] = _prune_history(conversation_history)

    print("  Ala is analysing your file…")
    print()

    try:
        full_response = _call_llm(client, conversation_history, system_prompt=sys_prompt)
    except Exception as exc:
        logger.exception("LLM call failed during file analysis")
        _handle_api_error(exc)
        conversation_history.pop()
        return

    _sep()

    # Store clean versions in history
    conversation_history[-1]["content"] = (
        f"[Uploaded {path.name}]\n" + (user_context or "(no additional context)")
    )
    conversation_history.append({"role": "assistant", "content": full_response})

    print("  Ask follow-up questions, upload another file, or type 'new' / 'quit'.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal UI helpers
# ─────────────────────────────────────────────────────────────────────────────
_W = 66


def _banner() -> None:
    print()
    print("=" * _W)
    print(f"  {settings.APP_NAME}  —  {settings.APP_TAGLINE}  (v{settings.APP_VERSION})")
    print("=" * _W)
    print("  Describe your legal situation in plain language, or upload a file.")
    print("  Supports: English, Hindi, Telugu, Tamil, Bengali, Marathi + more.")
    print()
    print("  Commands:")
    print("    upload <path>  — Analyse a document, image, or video")
    print("    new            — Clear history, start a fresh case")
    print("    help           — Show all commands")
    print("    quit           — Exit")
    if not retriever.is_ready:
        print()
        print(
            "  [Warning] Legal knowledge base not loaded. "
            "Answers will rely on the LLM alone."
        )
    if kanoon.is_enabled:
        print()
        print("  [Live case law: IndianKanoon connected]")
    print("=" * _W)
    print()


def _sep() -> None:
    print("\n" + "-" * _W + "\n")


def _error(msg: str) -> None:
    print(f"\n  [Error] {msg}\n")


def _hint(msg: str) -> None:
    print(f"  > {msg}")


def _note(msg: str) -> None:
    print(f"\n  [Note] {msg}\n")


def _show_help() -> None:
    print()
    print("─" * _W)
    print("  ALA — HELP")
    print("─" * _W)
    print()
    print("  TYPING YOUR SITUATION")
    print("  Just type what happened — no legal knowledge needed.")
    print("  Include: what happened, when, where, who was involved.")
    print("  Ask follow-up questions at any time.")
    print()
    print("  UPLOADING FILES")
    print("  Type:  upload <full path to file>")
    print("  Then optionally add context about the file.")
    print()
    print("  Supported documents:  PDF, DOCX, TXT, MD")
    print("  Supported images:     JPG, PNG, BMP, WEBP, TIFF, GIF")
    print("  Supported media:      MP4, MOV, AVI, MKV, MP3, WAV, M4A")
    print()
    print("  COMMANDS")
    print("    upload <path>  — Analyse a file")
    print("    new            — Start a fresh case (clears history)")
    print("    help           — Show this help")
    print("    quit / exit    — Exit Ala")
    print()
    print("  FREE LEGAL AID")
    print("    NALSA helpline: 15100 (free legal advice, all India)")
    print("    District Legal Services Authority (DLSA) — at every district court")
    print()
    print("  DISCLAIMER")
    print("    Ala provides legal information only — not legal advice.")
    print("    For serious matters, always consult a qualified lawyer.")
    print()
    print("─" * _W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main conversation loop
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    """
    Start the interactive Ala CLI session.

    Raises:
        EnvironmentError — propagated from config.py when GROQ_API_KEY is missing.
    """
    client = Groq(api_key=settings.GROQ_API_KEY)
    conversation_history: list[dict] = []

    _banner()
    print("  Describe your situation below.")
    print("  (What happened? When? Where? Who was involved?)")
    print()

    while True:
        # ── Read user input ─────────────────────────────────────────────
        try:
            raw_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye. Stay safe and know your rights!\n")
            break

        # ── Sanitise ────────────────────────────────────────────────────
        user_input = _sanitise(raw_input)
        if not user_input:
            continue

        cmd_lower = user_input.lower().strip()

        # ── Built-in commands ───────────────────────────────────────────
        if cmd_lower in ("quit", "exit", "q", "bye"):
            print("\n  Goodbye. Stay safe and know your rights!\n")
            break

        if cmd_lower in ("new", "reset", "clear"):
            conversation_history.clear()
            print()
            print("=" * _W)
            print("  Fresh case started. Describe your new situation:")
            print("=" * _W)
            print()
            continue

        if cmd_lower in ("help", "?", "h"):
            _show_help()
            continue

        # ── Upload command ──────────────────────────────────────────────
        # Syntax:  upload <path>  OR  upload <path> | <context>
        if cmd_lower.startswith("upload "):
            rest = user_input[7:].strip()
            if not rest:
                _error("No file path provided.")
                _hint("Usage:  upload /path/to/file.pdf")
                continue

            # Optional context after a pipe separator
            if "|" in rest:
                file_part, context_part = rest.split("|", 1)
                file_path = file_part.strip()
                user_context = context_part.strip()
            else:
                file_path = rest
                user_context = ""

            _handle_upload(file_path, client, conversation_history, user_context)
            continue

        # ── Bare file path detection (convenience) ──────────────────────
        # If the user pastes a path without the upload command
        stripped = user_input.strip().strip('"').strip("'")
        maybe_path = Path(stripped)
        if (
            maybe_path.suffix.lower() in ALL_UPLOAD_EXTS
            and maybe_path.exists()
            and maybe_path.is_file()
        ):
            print()
            print(
                f"  [Detected a file path: {maybe_path.name}]"
            )
            _handle_upload(stripped, client, conversation_history)
            continue

        # ── Language detection ──────────────────────────────────────────
        lang = detect_language(user_input)
        lang_note = format_language_note(lang)
        if lang_note:
            print(lang_note)

        # ── General situation query ─────────────────────────────────────
        rag_context = retriever.retrieve(user_input)

        # Enrich with live case law if IndianKanoon is configured
        kanoon_context = kanoon.retrieve(user_input)

        # Combine contexts: local BM25 first, then live case law
        combined_context = "\n\n".join(
            part for part in (rag_context, kanoon_context) if part
        )

        # Adapt system prompt for detected language
        sys_prompt = adapt_system_prompt(analyzer.SYSTEM_PROMPT, lang)

        user_msg = analyzer.build_analysis_prompt(
            situation=user_input,
            rag_context=combined_context,
        )

        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history[:] = _prune_history(conversation_history)

        print()
        print("  Ala is analysing your situation…")
        print()

        try:
            full_response = _call_llm(client, conversation_history, system_prompt=sys_prompt)
        except Exception as exc:
            logger.exception("LLM call failed")
            _handle_api_error(exc)
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()
            continue

        _sep()

        # Store clean versions in history (no RAG context prefix)
        conversation_history[-1]["content"] = user_input
        conversation_history.append({"role": "assistant", "content": full_response})

        print("  Ask a follow-up, upload a file, type 'new' or 'quit'.")
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run()
    except EnvironmentError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unhandled exception in main")
        print(f"\n  Unexpected error: {exc}", file=sys.stderr)
        sys.exit(2)

