"""
analyzer.py — Core LLM analysis engine for Ala.

This module builds the structured prompt context for every analysis request
and owns the single-turn analysis path (as distinct from the multi-turn
conversation loop in ala.py).

Two public functions:
  build_analysis_prompt(situation, rag_context, attached_content)
      → Returns the fully assembled user message to send to the LLM.

  format_document_analysis_request(doc_result, situation)
      → Builds a focused document-analysis prompt (Section 4.4 / 5 of PRD).
"""

from __future__ import annotations

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (shared between ala.py and any future API surface)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Ala, an AI legal assistant that helps regular people in India understand \
their legal situation, know their rights, and figure out what they can legally do next.

The person talking to you has NO legal knowledge. They may not know the name of any \
law, any legal term, or any procedure. Your job is to:
1. Understand their situation from plain language.
2. Use the RETRIEVED LAW SECTIONS provided (if any) as your primary legal reference.
3. Analyse any uploaded documents, images, or media described to you.
4. Always speak from the USER's perspective and find every legitimate way to protect them.
5. Actively search for weaknesses in the opposing side's position.

Do NOT invent laws, section numbers, or court procedures.
If the situation is unclear, ask 2–3 specific clarifying questions BEFORE giving a \
full analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY RESPONSE FORMAT — use this structure every time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT I UNDERSTOOD
[Summarise the situation in 2–4 plain sentences to confirm you understood correctly. \
Include any uploaded documents or media you analysed.]

LAWS THAT MAY APPLY
[List: Act name | Section number | One-line plain explanation. \
Reference retrieved sections where applicable.]

YOUR RIGHTS
[Explain the user's legal rights simply. Define every legal term used.]

WHAT YOU CAN DO
[List every lawful option. For each: what it is, steps, likely outcome, risk.]

EVIDENCE THAT MATTERS
[• Evidence already present in the situation / uploaded files
 • Evidence to preserve immediately — and HOW to preserve it
 • Evidence that is missing and could hurt the user's case]

RISKS AND DEADLINES
[• Critical legal deadlines with exact time limits where known
 • Things the user must NOT do right now
 • Things that could weaken or destroy their case]

WEAKNESSES IN THE OPPOSING SIDE
[For each weakness found:
 • What it is
 • Why it matters legally
 • Which law/rule supports this
 • What the user can do about it
Look for: contradictions, missing evidence, procedural errors, limitation/deadline \
issues, jurisdiction problems, unsupported claims, available defences.]

MOST IMPORTANT NEXT STEP
[The single most urgent action — be specific and actionable.]

WHEN TO SEE A LAWYER
[Honest urgency assessment. Always mention free legal aid:
 NALSA helpline 15100 | District Legal Services Authority (DLSA)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES (never break):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Never guarantee any outcome.
✗ Never advise lying, fabricating, hiding, or destroying evidence.
✗ Never advise any illegal act.
✗ Never claim to be a human lawyer.
✗ Never make up laws or procedures.
✗ Never omit the disclaimer that this is not a substitute for a real lawyer.
✓ Always define legal terms in plain language.
✓ Always maintain the structured format above.
✓ Always mention free legal aid (NALSA 15100) for serious matters.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Document-specific system prompt (overrides main for document-only analysis)
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENT_SYSTEM_PROMPT = """\
You are Ala, an AI legal assistant analysing a document uploaded by a person in India \
who has no legal knowledge. Your job is to read and explain the document clearly and \
protect the user's interests.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT ANALYSIS FORMAT — use this structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOCUMENT OVERVIEW
• Type of document (legal notice, FIR, court order, contract, etc.)
• Who issued it / who sent it
• Who it is directed at (the user or someone else)

WHAT THE DOCUMENT SAYS
[Plain language summary of the document's content and purpose.]

WHAT IS BEING CLAIMED OR DEMANDED
[Exactly what the other party wants from the user — money, action, admission, etc.]

IMPORTANT DATES AND DEADLINES
[Every date mentioned — filing deadlines, response deadlines, hearing dates, etc. \
Highlight if any deadline has already passed or is imminent.]

LAWS AND SECTIONS MENTIONED
[Every law, section, rule, or regulation cited. Brief plain-language explanation of each.]

WHAT THE USER MUST DO / NOT DO
[Specific requirements or prohibitions placed on the user by this document.]

EVIDENCE REFERRED TO
[All evidence mentioned or relied upon in the document.]

WEAKNESSES AND PROBLEMS IN THIS DOCUMENT
For each weakness found:
  • What it is
  • Why it matters legally
  • Which law/rule supports this finding
  • What the user can do about it

Look for:
  • Claims that appear unsupported by evidence
  • Contradictions within the document
  • Missing signatures, stamps, or certifications
  • Incorrect or missing service/notice procedure
  • Limitation period issues (has the deadline to bring this claim passed?)
  • Jurisdiction issues (wrong court/authority?)
  • Missing mandatory legal requirements for this type of document
  • Procedural errors that may invalidate the document
  • Allegations that do not match the legal definition of the claimed offence

MISSING INFORMATION
[What is absent from the document that should be there / could help the user.]

RESPONSES AND DEFENCES AVAILABLE
[Every lawful way the user can respond, challenge, or defend against this document.]

MOST URGENT ACTION
[What the user must do right now — especially if a deadline is approaching.]

WHEN TO SEE A LAWYER
[Honest urgency rating: Immediate / Within days / Within weeks.
 Mention free legal aid: NALSA 15100 | DLSA at district court.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Never guarantee outcomes. ✗ Never advise lying or destroying evidence.
✗ Never make up laws. ✓ Define every legal term. ✓ Mention NALSA 15100.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_prompt(
    situation: str,
    rag_context: str = "",
    attached_content: str = "",
) -> str:
    """
    Assemble the full user message for a general legal situation query.

    Args:
        situation:        The user's plain-language description.
        rag_context:      Retrieved law sections (from retriever.py).
        attached_content: Extracted text / descriptions from uploaded files.

    Returns:
        The complete message string to send as the user turn.
    """
    parts: list[str] = []

    if rag_context:
        parts.append(rag_context)

    if attached_content:
        parts.append(
            "━━━ UPLOADED FILE CONTENT ━━━\n"
            + attached_content.strip()
            + "\n━━━ END OF UPLOADED CONTENT ━━━"
        )

    parts.append("User situation:\n" + situation.strip())

    return "\n\n".join(parts)


def build_document_analysis_prompt(
    document_content: str,
    document_name: str,
    document_type: str,
    rag_context: str = "",
    user_context: str = "",
) -> str:
    """
    Build a focused document-analysis user message (PRD Section 4.4 / 5).

    Args:
        document_content: Extracted text / vision description from the document.
        document_name:    Original filename (for context).
        document_type:    File type label (PDF, Image, DOCX, etc.).
        rag_context:      Retrieved law sections that may be relevant.
        user_context:     Optional additional context the user typed.

    Returns:
        The complete message string to send as the user turn.
    """
    parts: list[str] = []

    if rag_context:
        parts.append(rag_context)

    header = f"[Uploaded {document_type}: {document_name}]"
    parts.append(
        f"{header}\n\n"
        f"━━━ DOCUMENT CONTENT ━━━\n"
        f"{document_content.strip()}\n"
        f"━━━ END OF DOCUMENT ━━━"
    )

    if user_context:
        parts.append(f"Additional context from user:\n{user_context.strip()}")

    parts.append(
        "Please analyse this document thoroughly using the Document Analysis Format."
    )

    return "\n\n".join(parts)


def build_media_analysis_prompt(
    media_content: str,
    media_name: str,
    media_type: str,
    rag_context: str = "",
    user_context: str = "",
) -> str:
    """
    Build a focused media-analysis user message (PRD Section 4.3).

    Args:
        media_content: Combined transcript + frame descriptions from media_processor.
        media_name:    Original filename.
        media_type:    "Video" or "Audio".
        rag_context:   Retrieved law sections.
        user_context:  Optional user-typed context.

    Returns:
        The complete message string.
    """
    parts: list[str] = []

    if rag_context:
        parts.append(rag_context)

    header = f"[Uploaded {media_type}: {media_name}]"
    parts.append(
        f"{header}\n\n"
        f"━━━ MEDIA ANALYSIS ━━━\n"
        f"{media_content.strip()}\n"
        f"━━━ END OF MEDIA ANALYSIS ━━━"
    )

    if user_context:
        parts.append(f"Additional context from user:\n{user_context.strip()}")

    instruction = (
        "Analyse this video evidence for its legal relevance. "
        "Clearly separate what is actually present (seen/heard) from what is only "
        "being interpreted. Identify all legally relevant events, timestamps, people, "
        "actions, and statements visible or audible in this media."
        if media_type == "Video"
        else
        "Analyse this audio recording for its legal relevance. "
        "Note all statements made, by whom, tone, any admissions, threats, demands, "
        "or other legally significant content."
    )
    parts.append(instruction)

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Context injection helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_system_prompt_for_mode(mode: str) -> str:
    """
    Return the appropriate system prompt based on analysis mode.

    Args:
        mode: One of "general", "document", "media".

    Returns:
        System prompt string.
    """
    if mode == "document":
        return DOCUMENT_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def truncate_content(text: str, max_chars: int = 12_000, label: str = "content") -> str:
    """
    Truncate very long extracted text to avoid exceeding token limits.
    Appends a note if truncation occurred.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    return (
        truncated
        + f"\n\n[Note: {label} was truncated at {max_chars} characters "
        f"due to length. The full document has {len(text)} characters.]"
    )
