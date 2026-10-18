# ⚖️ Ala — AI Legal Assistant for India

> **Understand your rights. Know your options. Protect your interests.**

Ala is an open-source AI-powered legal assistant designed to help everyday people in India navigate legal situations — without needing any prior legal knowledge. Simply describe what happened in plain language, and Ala explains the relevant Indian laws, your rights, what you can do, what evidence matters, and what risks to watch out for.

---

## 🎯 The Problem

Regular people frequently face legal situations but:

- Do not know which law applies to their situation
- Cannot understand complex legal terminology
- Do not know what procedure to follow
- Do not know what evidence is useful or important
- May not even know whether something happening to them is legal or illegal
- Cannot afford or immediately access a lawyer
- May unknowingly give up their rights or make costly mistakes

**Ala bridges this gap.** It is not a replacement for a lawyer — it is a guide that helps you understand your situation and make informed decisions.

---

## ✨ What Ala Does

For any situation you describe, Ala provides:

| Section | What you get |
|---|---|
| 📋 **What I Understood** | Confirms it understood your situation correctly |
| ⚖️ **Laws That May Apply** | Relevant Indian Acts, Sections, Rules in plain language |
| 🛡️ **Your Rights** | Your legal rights explained simply |
| 📝 **What You Can Do** | Every lawful option with steps, outcomes, and risks |
| 🔍 **Evidence That Matters** | What you have, what to preserve, what is missing |
| ⚠️ **Risks & Deadlines** | Critical deadlines, things to avoid, things that could hurt you |
| 🔎 **Weaknesses in the Opposing Side** | Contradictions, procedural errors, missing evidence |
| 💡 **Most Important Next Step** | The single most urgent action to take right now |
| 🤝 **When to See a Lawyer** | Honest advice on when professional help is necessary |

---

## 🏗️ Architecture

```
ala_local/
├── ala.py                  ← Main CLI app (entry point)
├── config.py               ← All settings & API key abstraction
├── retriever.py            ← BM25-based RAG retrieval engine
├── knowledge/
│   └── indian_laws.json    ← Open Indian legal knowledge base (34+ entries)
├── .env                    ← Your API key (NOT committed to git)
├── .env.example            ← Template for setting up .env
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

### How It Works — RAG Pipeline

```
User types their situation in plain language
              ↓
BM25 retrieval searches 34+ Indian law entries locally
(no embedding API, no external call — 100% local)
              ↓
Top 5 most relevant law sections retrieved
              ↓
Retrieved sections injected into LLM prompt as grounding context
              ↓
Qwen 3.8B 27B (via Groq) generates structured legal analysis
grounded in actual Indian law text
              ↓
Clean, structured output printed to terminal
```

---

## 🧠 Tech Stack

| Component | Technology | Why |
|---|---|---|
| **LLM** | Qwen 3.8B-27B via [Groq](https://console.groq.com) | Free, fast, open-source model |
| **RAG** | BM25 (`rank-bm25`) | Local, no embedding API needed |
| **Knowledge Base** | Hand-curated JSON from open Indian legal sources | Grounded, accurate legal data |
| **Config** | `python-dotenv` + `config.py` | API key never touches main code |
| **Interface** | Python CLI (terminal) | Simple, works everywhere |

---

## 📚 Legal Knowledge Base

The RAG knowledge base (`knowledge/indian_laws.json`) covers 34+ entries across:

- **Criminal Law** — BNS/IPC (theft, assault, cheating, threats, defamation, rape)
- **Procedure** — BNSS/CrPC (FIR, Zero FIR, arrest rights, anticipatory bail)
- **Family Law** — Domestic Violence Act, Dowry Prohibition Act, POCSO
- **Consumer Law** — Consumer Protection Act 2019
- **Property & Tenancy** — Transfer of Property Act, Rent Control
- **Contract Law** — Indian Contract Act 1872
- **Cyber Law** — IT Act 2000 (hacking, OTP fraud, identity theft, revenge porn)
- **Labour Law** — Industrial Disputes Act, Payment of Wages Act
- **Civil Law** — Cheque Bounce (NI Act), Injunctions, Limitation Act
- **Constitutional Rights** — Articles 21, 22, 32, 226 (fundamental rights, writs)
- **Social Protection** — SC/ST Atrocities Act, POSH Act, Senior Citizens Act
- **Access to Justice** — RTI Act, Free Legal Aid (NALSA/DLSA)
- **Evidence** — Indian Evidence Act / Bharatiya Sakshya Adhiniyam (electronic evidence)

> The knowledge base is easily extensible — add more entries to `knowledge/indian_laws.json` in the same format and they are automatically indexed on the next run.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com) (no credit card required)

### 1. Clone the repository

```bash
git clone https://github.com/manoj-kumar-kolla/Ai-leagal-aid.git
cd Ai-leagal-aid
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project folder:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free key at [console.groq.com](https://console.groq.com) → API Keys → Create API Key.

### 4. Run Ala

```bash
python ala.py
```

---

## 💬 Example Usage

```
==============================================================
  Ala  -  AI Legal Assistant (India)
  Understand your rights. Know your options.
==============================================================
  * Describe your situation in plain language
  * Ask follow-up questions at any time
  * Type  'new'   to start a fresh case
  * Type  'quit'  to exit
==============================================================

  Describe your situation below.
  (What happened? When? Where? Who was involved?)

You: My landlord locked my room while I was at work and is now
     demanding extra money or he will not let me back in.

  Ala is analyzing your situation...

--------------------------------------------------------------
  ALA:
--------------------------------------------------------------
WHAT I UNDERSTOOD
Your landlord has forcefully locked you out of your rented
premises without a court order and is demanding additional
money as a condition to let you back in...

LAWS THAT MAY APPLY
- Transfer of Property Act 1882, Section 108 — A tenant has
  the right to peaceful enjoyment of the rented property...
- BNS Section 329 / IPC Section 441 — Criminal trespass...
...
```

---

## ⚙️ Configuration

All settings are controlled via environment variables in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | LLM model to use |
| `TEMPERATURE` | `0.2` | Lower = more factual/consistent |
| `MAX_TOKENS` | `4096` | Maximum response length |
| `RAG_TOP_K` | `5` | Number of law sections to retrieve per query |

---

## ⚠️ Disclaimer

**Ala is an AI tool for legal information and general guidance — it is NOT a substitute for professional legal advice.**

- Ala does not guarantee any legal outcome.
- Ala does not create a lawyer-client relationship.
- For serious legal matters, always consult a qualified lawyer.
- For free legal help in India, contact your nearest **District Legal Services Authority (DLSA)** or call **NALSA helpline: 15100**.

**Ala will never advise you to:**
- Lie or fabricate evidence
- Hide or destroy evidence
- Mislead a court or authority
- Do anything illegal

---

## 🗺️ Roadmap

- [x] Terminal CLI with conversation history
- [x] BM25 RAG over Indian legal knowledge base
- [x] API key abstraction via config module
- [ ] Image and PDF upload support (legal notices, contracts)
- [ ] Video analysis for evidence
- [ ] Hindi and regional language support
- [ ] Web interface
- [ ] Expanded knowledge base (state-specific laws, case law)
- [ ] Integration with IndianKanoon API for live case law search

---

## 🤝 Contributing

Contributions are welcome! The most impactful way to contribute right now is to **expand the knowledge base**:

1. Add entries to `knowledge/indian_laws.json` following the existing format:

```json
{
  "act": "Name of the Act and Year",
  "section": "Section number(s)",
  "title": "Short descriptive title",
  "keywords": "space separated keywords for search",
  "plain_text": "Plain language explanation of the law, rights, and procedure."
}
```

2. Open a Pull Request with your additions.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Built by [Manoj Kumar Kolla](https://github.com/manoj-kumar-kolla)

*Inspired by the belief that legal rights should be accessible to everyone, not just those who can afford a lawyer.*
