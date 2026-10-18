"""
ala.py - Main entry point for Ala, the AI Legal Assistant for India.
API key and settings are managed in config.py (never here).
Legal knowledge retrieval is handled by retriever.py (RAG).
"""

import sys

# Force UTF-8 on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from groq import Groq
from config import settings
from retriever import retriever

# ─────────────────────────────────────────────
#  SYSTEM PROMPT — The brain of Ala
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Ala, an AI legal assistant that helps regular people in India understand their legal situation, know their rights, and figure out what they can legally do next.

The person talking to you has NO legal knowledge. They may not know the name of any law, any legal term, or any procedure. Your job is to:
1. Understand their situation from plain language.
2. Use the RETRIEVED LAW SECTIONS provided to ground your analysis in actual Indian law.
3. Always speak from the USER perspective and look for every legitimate way to protect them.

When RETRIEVED LAW SECTIONS are provided at the top of the user message, use them as your primary legal reference. Do not make up laws or sections not in your knowledge.

RESPONSE FORMAT - Always use this exact structure:

WHAT I UNDERSTOOD
[Summarize the situation in 2-4 simple sentences to confirm you understood correctly.]

LAWS THAT MAY APPLY
[List relevant Indian laws with Act name, Section number, and one-line plain explanation. Reference the retrieved sections where applicable.]

YOUR RIGHTS
[Explain the user's legal rights in plain language. No jargon without explanation.]

WHAT YOU CAN DO
[List every lawful option with: what it is, how to do it (brief steps), likely outcome, risk involved.]

EVIDENCE THAT MATTERS
[- Evidence already mentioned
 - Evidence to preserve immediately
 - Evidence that is missing and could hurt them]

RISKS AND DEADLINES
[- Important legal deadlines
 - Things the user must NOT do
 - Things that could seriously hurt their case]

WEAKNESSES IN THE OPPOSING SIDE
[Contradictions, missing evidence, procedural errors, unsupported claims, jurisdiction issues, defences available.]

MOST IMPORTANT NEXT STEP
[Single most urgent action to take right now.]

WHEN TO SEE A LAWYER
[Honest advice on when professional help is needed. Mention urgency.]

STRICT RULES:
- Only cite Indian law (IPC/BNS, CrPC/BNSS, CPC, Constitution, relevant Acts).
- Explain every legal term you use.
- Never guarantee victory.
- Never advise lying, hiding, destroying evidence, or anything illegal.
- Never pretend to be a human lawyer.
- Never make up laws or procedures.
- If information is insufficient, ask 2-3 specific questions first.
"""


# ─────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────
def print_banner() -> None:
    w = 62
    print()
    print("=" * w)
    print(f"  {settings.APP_NAME}  -  {settings.APP_TAGLINE}")
    print("=" * w)
    print("  * Describe your situation in plain language")
    print("  * Ask follow-up questions at any time")
    print("  * Type  'new'   to start a fresh case")
    print("  * Type  'quit'  to exit")
    print("=" * w)
    print()


def sep() -> None:
    print("\n" + "-" * 62 + "\n")


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def run() -> None:
    client = Groq(api_key=settings.GROQ_API_KEY)
    conversation_history: list[dict] = []

    print_banner()
    print("  Describe your situation below.")
    print("  (What happened? When? Where? Who was involved?)")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Exiting Ala. Stay safe and know your rights!\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Exiting Ala. Stay safe and know your rights!\n")
            break

        if user_input.lower() in ("new", "reset", "clear"):
            conversation_history = []
            print()
            print("=" * 62)
            print("  Starting a fresh case. Describe your situation:")
            print("=" * 62)
            print()
            continue

        # ── RAG: retrieve relevant law sections ──────────────────────
        context = retriever.retrieve(user_input)
        if context:
            augmented_input = context + "\n\nUser situation:\n" + user_input
        else:
            augmented_input = user_input

        conversation_history.append({"role": "user", "content": augmented_input})

        print()
        print("  Ala is analyzing your situation...")
        print()

        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *conversation_history,
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                stream=True,
            )

            print("-" * 62)
            print("  ALA:")
            print("-" * 62)
            print()

            full_response = ""
            in_think = False
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    # Strip Qwen3 internal reasoning tags
                    if "<think>" in delta:
                        in_think = True
                    if in_think:
                        if "</think>" in delta:
                            in_think = False
                        continue
                    print(delta, end="", flush=True)

            print()
            sep()

            # Store the clean response (without context prefix) for follow-ups
            conversation_history[-1]["content"] = user_input   # keep history clean
            conversation_history.append({"role": "assistant", "content": full_response})

            print("  Ask a follow-up, add more details, type 'new' or 'quit'.")
            print()

        except Exception as e:
            msg = str(e)
            print(f"\n  Error: {msg}")
            if "401" in msg or "api_key" in msg.lower():
                print("  -> API key is invalid. Check your .env file.")
            elif "429" in msg:
                print("  -> Rate limit hit. Wait a moment and try again.")
            elif "model" in msg.lower():
                print(f"  -> Model '{settings.GROQ_MODEL}' not available.")
            else:
                print("  -> Check your internet connection.")
            print()
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()


if __name__ == "__main__":
    try:
        run()
    except EnvironmentError as e:
        print(e)
        sys.exit(1)
