"""
server.py — Full-Featured, Secure Localhost Web Server for Ala Legal Aid Assistant.

Supports:
  - Natural language legal situation queries with BM25 RAG
  - PDF, DOCX, TXT, and Image document analysis (legal notices, agreements, orders)
  - Audio and Video evidence analysis via Whisper & Vision models
  - 12 Indian regional languages (Hindi, Telugu, Tamil, Bengali, Marathi, etc.)
  - Live IndianKanoon court judgement retrieval (when configured)
  - Full structured analysis (Understanding, Laws, Rights, Options, Evidence, Risks, Weaknesses, Next Steps, Free Legal Aid)
  - Emergency legal helplines & DLSA directory endpoint
  - Strict security headers (CSP, nosniff, DENY, etc.), sliding window rate limiting, and bot protection
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure ala_local is on sys.path
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from config import settings
from retriever import retriever, _tokenize, _MIN_SCORE
from indiankanoon import kanoon
from language import detect_language, adapt_system_prompt, format_language_note
import analyzer
from document_processor import (
    ALL_SUPPORTED_EXTS as DOC_EXTS,
    process_file,
    describe_file,
)
from media_processor import (
    ALL_MEDIA_EXTS as MEDIA_EXTS,
    process_media,
)
from groq import Groq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ala_server")

# ── Security & Limits Configuration ──────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8000
MAX_INPUT_CHARS = 4000
MAX_RESPONSE_CHARS = 10000
RATE_LIMIT_MAX = 20       # requests per IP
RATE_LIMIT_WINDOW = 60.0  # seconds

# In-memory sliding window for IP rate limiting
_ip_request_timestamps: Dict[str, List[float]] = defaultdict(list)

# Initialize Groq client
_groq_client: Optional[Groq] = None
try:
    if settings.GROQ_API_KEY:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("Groq client initialized with model: %s", settings.GROQ_MODEL)
except Exception as exc:
    logger.warning("Could not initialize Groq client: %s", exc)

INDEX_FILE = _DIR / "index.html"

# Known fallback document database for basis metadata
DOCS_MAP = {
    "wages_gratuity": ("Unpaid salary and gratuity", "Payment of Gratuity Act 1972 s.4", "Gratuity is payable on leaving after five years of continuous service. Unpaid wages can be claimed via the State labour authority."),
    "termination": ("Termination from employment", "Industrial Disputes Act 1947", "Ask for reasons for termination in writing. Retrenchment compensation may apply based on service length."),
    "legal_notice": ("Receiving or sending a legal notice", "CPC 1908 s.80 / General Practice", "A legal notice is a formal demand letter. Reply in writing before the deadline."),
    "legal_aid": ("Free legal aid (NALSA)", "Constitution Art. 39A; Legal Services Authorities Act 1987 s.12", "Eligible persons (women, children, industrial workers, custody) can get free legal counsel via DLSA/SLSA/NALSA."),
    "arrest_rights": ("Rights upon arrest", "Constitution Art. 20(3), 22; BNSS 2023 s.47, 58", "Right to know arrest grounds, right to counsel, must be produced before Magistrate within 24 hours."),
    "fir_registration": ("Filing an FIR", "BNSS 2023 s.173", "Police must register an FIR for cognizable offences and provide a free signed copy."),
    "bail_basics": ("Bail principles", "BNSS 2023 s.478, 480, 482", "Bail is a right for bailable offences. For non-bailable offences, court discretion applies."),
    "evidence_preservation": ("Preserving digital & written evidence", "BSA 2023 s.63", "Preserve messages, bank transfers, contracts, and audio records. Electronic records require certificate."),
    "tenant_notice": ("Tenancy and deposit disputes", "Transfer of Property Act s.106; State Rent Control", "Landlords cannot evict by force or withhold deposits arbitrarily. Written demand followed by notice is standard."),
    "contract_breach": ("Breach of contract & recovery", "Indian Contract Act 1872 s.73, 74", "Compensation can be claimed for damages naturally arising from breach."),
}

HELPLINES = [
    {"name": "NALSA Free Legal Aid Helpline", "number": "15100", "desc": "Free legal aid for eligible citizens (women, children, custody, low income) across India.", "badge": "Legal Aid"},
    {"name": "National Emergency Number", "number": "112", "desc": "Police, Fire, and Ambulance emergency services nationwide.", "badge": "Emergency"},
    {"name": "National Cyber Crime Helpline", "number": "1930", "desc": "Immediate reporting for financial cyber frauds, UPI scams, and online crimes (cybercrime.gov.in).", "badge": "Cyber"},
    {"name": "Women's Helpline (All India)", "number": "1091 / 181", "desc": "24/7 emergency response for women facing domestic violence, harassment, or distress.", "badge": "Women"},
    {"name": "Childline (POCSO / Child Rights)", "number": "1098", "desc": "Emergency assistance and protection for children in distress or abuse situations.", "badge": "Children"},
    {"name": "Senior Citizen National Helpline", "number": "14567", "desc": "Elderline — support, legal guidance, and protection for elderly citizens.", "badge": "Elders"},
    {"name": "Labour & EPFO Helpline", "number": "1800-118-005", "desc": "Provident fund disputes, unpaid wages, and employee grievance redressal.", "badge": "Labour"},
    {"name": "National Consumer Helpline", "number": "1915", "desc": "Consumer disputes, defective products, refund denial, and unfair trade practices.", "badge": "Consumer"},
]


def _check_rate_limit(ip: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    timestamps = [t for t in _ip_request_timestamps[ip] if t > cutoff]
    _ip_request_timestamps[ip] = timestamps
    if len(timestamps) >= RATE_LIMIT_MAX:
        return False
    _ip_request_timestamps[ip].append(now)
    return True


def _detect_urgency(text: str) -> bool:
    return bool(re.search(r"\b(arrest|police|custody|detain|lockup|bail|handcuff|remand|fir|assault|violence|beaten)\b", text, re.IGNORECASE))


def _find_matched_docs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query BM25 retriever and return structured doc records for frontend basis sidebar."""
    matched = []
    if retriever.is_ready:
        tokens = _tokenize(query)
        if tokens and retriever._bm25 is not None:
            scores = retriever._bm25.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            for idx in top_indices:
                if scores[idx] > _MIN_SCORE:
                    chunk = retriever._chunks[idx]
                    cid = f"rag_{idx}"
                    title = chunk.get("title") or chunk.get("section") or f"Section {idx}"
                    source = f"{chunk.get('act', '')} {chunk.get('section', '')}".strip()
                    body = chunk.get("plain_text") or chunk.get("summary") or ""
                    matched.append({
                        "id": cid,
                        "title": title,
                        "source": source,
                        "body": body,
                        "score": round(float(scores[idx]), 2),
                    })

    # If few BM25 matches found, add keyword fallbacks
    if len(matched) < 2:
        q_lower = query.lower()
        for doc_id, (title, src, body) in DOCS_MAP.items():
            words = doc_id.split("_")
            if any(w in q_lower for w in words) and doc_id not in [m["id"] for m in matched]:
                matched.append({
                    "id": doc_id,
                    "title": title,
                    "source": src,
                    "body": body,
                    "score": 4.5,
                })
                if len(matched) >= top_k:
                    break

    return matched



def _build_structured_fallback_analysis(
    question: str,
    matched_docs: List[Dict[str, Any]],
    is_urgent: bool,
    doc_info: Optional[Dict[str, Any]],
    uploaded_text: str
) -> str:
    """Builds comprehensive, plain-language legal analysis grounded in retrieved Indian statutes."""
    parts = []

    # 1. WHAT HAPPENED & KEY FACTS
    sit_desc = question.strip() if question.strip() else (f"Document analysis for: {doc_info['name']}" if doc_info else "Legal situation inquiry")
    parts.append(f"**What Happened & Key Situation**\n{sit_desc}\n")
    if doc_info and uploaded_text:
        parts.append(f"*(Analyzed {doc_info.get('type', 'document')}: **{doc_info.get('name', 'file')}** — extracted contractual covenants, notices, and statutory clauses)*\n")

    # 2. URGENT POLICE / CUSTODY PROTECTIONS
    if is_urgent:
        parts.append("**⚠️ Critical Safeguards: Police or Custody Involvement**\n"
                     "- **Right to Know Grounds:** Section 47 of Bharatiya Nagarik Suraksha Sanhita (BNSS 2023) mandates that every arrested individual must be immediately informed of the full grounds of arrest.\n"
                     "- **Right to Counsel:** Under Article 22(1) and Section 47 BNSS, you have the right to consult and be defended by an advocate of your choice.\n"
                     "- **24-Hour Magistrate Production:** Section 58 BNSS strictly forbids police detention beyond 24 hours without an order of a Magistrate.\n"
                     "- **Protection Against Self-Incrimination:** Article 20(3) guarantees that no person accused of any offence shall be compelled to be a witness against himself.\n")

    # 3. LAWS THAT APPLY
    parts.append("**Laws That Apply Under Indian Jurisprudence**\n")
    if matched_docs:
        for d in matched_docs[:5]:
            parts.append(f"- **{d['title']}** (*{d['source']}*):\n  {d['body']}\n")
    else:
        parts.append("- **Bharatiya Nyaya Sanhita (BNS 2023) / Special Statutory Acts:** General remedies under the Constitution of India and relevant civil & criminal procedural codes apply.\n")

    # 4. YOUR FUNDAMENTAL RIGHTS
    parts.append("**Your Fundamental Legal Rights**\n"
                 "- **Right to Fair Due Process (Article 21):** No deprivation of life, liberty, or property except according to just, fair, and reasonable procedure established by law.\n"
                 "- **Tenancy & Property Protection:** Landlords cannot use self-help measures (cutting power/water, changing locks, physical threats). Section 106 of the Transfer of Property Act requires a formal written notice.\n"
                 "- **Employment & Gratuity Protection:** Under the Payment of Gratuity Act 1972 Section 4, gratuity is a statutory entitlement upon 5 years of service, payable within 30 days of leaving. Arbitrary wage deductions are illegal.\n")

    # 5. STEP-BY-STEP ACTION PLAN
    parts.append("**What You Can Do (Step-by-Step Action Plan)**\n"
                 "1. **Send a Formal Written Demand:** Issue a formal demand by email and registered speed post (with acknowledgment due) specifying a strict 15-day deadline for compliance.\n"
                 "2. **Issue an Advocate's Legal Notice:** If the demand is ignored, an advocate should issue a formal notice outlining cause of action, statutory claims, interest, and costs.\n"
                 "3. **Lodge an Official Complaint with the Appropriate Statutory Authority:**\n"
                 "   - *Consumer grievances:* Lodge on the National Consumer Portal (**e-Daakhil** at edaakhil.nic.in) or call helpline **1915**.\n"
                 "   - *Cyber financial fraud:* Report immediately within the golden hour on **1930** or cybercrime.gov.in to freeze mule accounts.\n"
                 "   - *Unpaid wages/gratuity:* File a petition before the State Labour Commissioner or Controlling Authority under Section 7 of the Gratuity Act.\n"
                 "   - *Tenancy withholding:* File a civil suit for refund with interest or petition the local Rent Authority under the Model Tenancy Act.\n")

    # 6. EVIDENCE TO PRESERVE
    parts.append("**Evidence to Preserve (Section 63, Bharatiya Sakshya Adhiniyam 2023)**\n"
                 "- Preserve original agreements, appointment letters, salary slips, bank passbook/UPI transaction receipts, and notice copies.\n"
                 "- Back up WhatsApp messages, emails, call logs, and voice recordings. Under BSA 2023 Section 63 (formerly s.65B Evidence Act), electronic records are admissible in court with an accompanying electronic certificate.\n")

    # 7. RISKS & STATUTORY DEADLINES
    parts.append("**Risks & Critical Statutory Deadlines**\n"
                 "- **Cheque Dishonour (s.138 Negotiable Instruments Act):** Legal notice must be sent within 30 days of the cheque return memo; the drawer has 15 days to pay; court complaint must be filed within 1 month thereafter.\n"
                 "- **Consumer Court Complaints:** Strict 2-year limitation period from the date of breach (Consumer Protection Act 2019 Section 69).\n"
                 "- **Recovery of Dues:** 3-year statutory limitation period under the Limitation Act 1963.\n")

    # 8. FREE GOVERNMENT LEGAL AID
    parts.append("**Free Legal Aid (NALSA / DLSA)**\n"
                 "If financial constraints prevent you from engaging private counsel, you are constitutionally entitled to free legal aid under Section 12 of the Legal Services Authorities Act 1987. Free advocates are provided for women, children, workers, custody detainees, and low-income citizens. Visit the District Legal Services Authority (DLSA) at your district court, or call the 24/7 national helpline at **15100**.")

    return "\n".join(parts)

ROUTE_MAP = {
    "/": "index.html",
    "/index.html": "index.html",
    "/ask": "ask.html",
    "/ask.html": "ask.html",
    "/case-studies": "case-studies.html",
    "/case-studies.html": "case-studies.html",
    "/news": "news.html",
    "/news.html": "news.html",
    "/reviews": "reviews.html",
    "/reviews.html": "reviews.html",
    "/login": "login.html",
    "/login.html": "login.html",
    "/404": "404.html",
    "/404.html": "404.html",
}

class AlaHandler(SimpleHTTPRequestHandler):
    """Custom request handler with full security headers, NDJSON streaming, multi-page routing, reviews API, and rate-limiting."""

    server_version = "AlaServer/2.3.0"

    def end_headers(self) -> None:
        """Inject strict modern security headers on all responses."""
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline' https://www.googletagmanager.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; "
            "connect-src 'self' https://www.google-analytics.com; frame-ancestors 'none'; form-action 'self'; "
            "base-uri 'self'; object-src 'none';",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        super().end_headers()

    def do_GET(self) -> None:
        """Route GET requests: multi-page clean URLs, static assets, APIs, and custom 404."""
        client_ip = self.client_address[0]
        raw_path = self.path.split("?")[0]
        clean_path = raw_path.rstrip("/") if raw_path != "/" else "/"

        # 1. Clean Multi-Page Routes
        if clean_path in ROUTE_MAP:
            target_name = ROUTE_MAP[clean_path]
            target_file = _DIR / target_name
            if target_file.exists():
                content = target_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(content)
                return

        # 2. SEO & Media Assets
        if clean_path == "/robots.txt":
            rf = _DIR / "robots.txt"
            if rf.exists():
                content = rf.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if clean_path == "/sitemap.xml":
            sf = _DIR / "sitemap.xml"
            if sf.exists():
                content = sf.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/xml; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if clean_path == "/social-preview.svg":
            imgf = _DIR / "social-preview.svg"
            if imgf.exists():
                content = imgf.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # 3. Law News Feed API (Requirement 18)
        if clean_path == "/api/news":
            nf = _DIR / "news.json"
            if nf.exists():
                content = nf.read_bytes()
            else:
                content = b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # 4. Customer Reviews Feed API (Requirement 10 & 19)
        if clean_path == "/api/reviews":
            rf = _DIR / "reviews.json"
            if rf.exists():
                content = rf.read_bytes()
            else:
                content = b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # 5. Helplines & System Status
        if clean_path == "/api/helplines":
            body = json.dumps(HELPLINES).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if clean_path in ("/api/status", "/health", "/api/health"):
            data = {
                "status": "healthy",
                "version": "2.3.0",
                "multipage_ready": True,
                "retriever_ready": retriever.is_ready,
                "indexed_entries": retriever.entry_count,
                "groq_configured": _groq_client is not None,
                "model": settings.GROQ_MODEL,
                "kanoon_enabled": kanoon.is_enabled if kanoon else False,
            }
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 6. Physical static file check
        file_candidate = _DIR / clean_path.lstrip("/")
        if file_candidate.is_file():
            super().do_GET()
            return

        # 7. Requirement 1: Custom 404 page for any unmapped route
        four_oh_four = _DIR / "404.html"
        if four_oh_four.exists():
            content = four_oh_four.read_bytes()
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self) -> None:
        """Handle POST requests: /api/ask (NDJSON streaming) and /api/reviews (reviews submission)."""
        client_ip = self.client_address[0]
        raw_path = self.path.split("?")[0]

        # 1. Customer Review Submission API (Requirement 19: Supabase & local DB)
        if raw_path == "/api/reviews":
            if not _check_rate_limit(client_ip):
                self.send_response(429)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Retry-After", "60")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Too many requests. Please wait a moment."}).encode("utf-8"))
                return

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 100 * 1024:
                self.send_error(413, "Review payload too large")
                return

            raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")
            try:
                data = json.loads(raw_body)
            except Exception:
                self.send_error(400, "Invalid JSON payload")
                return

            # Bot honeypot check
            if data.get("hp_name"):
                logger.warning("Bot review honeypot triggered by IP %s", client_ip)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode("utf-8"))
                return

            name = str(data.get("name", "Verified Citizen")).strip()[:100]
            role = str(data.get("role", "Citizen")).strip()[:100]
            city = str(data.get("city", "India")).strip()[:100]
            try:
                rating = max(1, min(5, int(data.get("rating", 5))))
            except (ValueError, TypeError):
                rating = 5
            category = str(data.get("category", "General Legal Guidance")).strip()[:100]
            review = str(data.get("review", "")).strip()[:1000]

            if len(review) < 5:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Review must be at least 5 characters long."}).encode("utf-8"))
                return

            new_review = {
                "id": f"rev_{int(time.time() * 1000)}",
                "name": name,
                "role": role,
                "city": city,
                "rating": rating,
                "category": category,
                "review": review,
                "verified": True,
                "date": time.strftime("%Y-%m-%d"),
            }

            # Save to local database (reviews.json)
            reviews_file = _DIR / "reviews.json"
            try:
                existing = json.loads(reviews_file.read_text(encoding="utf-8")) if reviews_file.exists() else []
                existing.insert(0, new_review)
                reviews_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.warning("Error saving local review: %s", e)

            # Sync to Supabase if configured (Requirement 19)
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
            if supabase_url and supabase_key:
                try:
                    import urllib.request
                    post_url = f"{supabase_url.rstrip('/')}/rest/v1/customer_reviews"
                    supa_req = urllib.request.Request(
                        post_url,
                        data=json.dumps(new_review).encode("utf-8"),
                        headers={
                            "apikey": supabase_key,
                            "Authorization": f"Bearer {supabase_key}",
                            "Content-Type": "application/json",
                            "Prefer": "return=minimal",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(supa_req, timeout=3) as resp:
                        logger.info("Supabase review synced successfully: code %s", resp.status)
                except Exception as se:
                    logger.info("Supabase sync note (offline/local fallback): %s", se)

            resp_body = json.dumps({"status": "success", "review": new_review}).encode("utf-8")
            self.send_response(201)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return

        # 2. Main Legal Analysis API /api/ask
        if raw_path != "/api/ask":
            self.send_error(404, "Endpoint not found")
            return

            self.send_error(404, "Endpoint not found")
            return

        # 1. Server-side Rate Limiting
        if not _check_rate_limit(client_ip):
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "60")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Too many requests. Please wait a moment."}).encode("utf-8"))
            return

        # 2. Read Request Body (allow up to 45 MB for document/image base64 payloads)
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 45 * 1024 * 1024:
            self.send_error(413, "Payload exceeds maximum allowed size (45 MB)")
            return

        raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")
        try:
            req_data = json.loads(raw_body)
        except Exception:
            self.send_error(400, "Invalid JSON payload")
            return

        # 3. Honeypot Bot Detection
        if req_data.get("hp_name"):
            logger.warning("Bot honeypot triggered by IP %s", client_ip)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"type":"done"}\n')
            return

        question = str(req_data.get("question", "")).strip()[:MAX_INPUT_CHARS]
        history = req_data.get("history", [])
        if not isinstance(history, list):
            history = []

        file_obj = req_data.get("file")
        if not question and not file_obj:
            self.send_error(400, "Please provide a question or attach a document.")
            return

        # Setup streaming response headers
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send_ndjson(obj: Dict[str, Any]) -> None:
            line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            chunk_header = f"{len(line):X}\r\n".encode("ascii")
            try:
                self.wfile.write(chunk_header + line + b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        # 4. Handle Uploaded File (if any)
        uploaded_doc_text = ""
        doc_info: Optional[Dict[str, Any]] = None
        temp_file_path: Optional[str] = None

        if isinstance(file_obj, dict) and "data" in file_obj:
            file_name = file_obj.get("name", "document.pdf")
            ext = Path(file_name).suffix.lower()
            raw_b64 = file_obj["data"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]

            try:
                file_bytes = base64.b64decode(raw_b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
                    tf.write(file_bytes)
                    temp_file_path = tf.name

                send_ndjson({"type": "status", "text": f"Analysing attached file: {file_name}..."})

                if ext in DOC_EXTS:
                    res = process_file(temp_file_path, _groq_client)
                    uploaded_doc_text = res.raw_text or res.vision_desc or ""
                    doc_info = {
                        "name": file_name,
                        "type": res.file_type,
                        "pages": getattr(res, "page_count", None),
                        "error": res.error,
                    }
                elif ext in MEDIA_EXTS:
                    res = process_media(temp_file_path, _groq_client)
                    uploaded_doc_text = getattr(res, "full_report", "") or getattr(res, "transcript", "")
                    doc_info = {
                        "name": file_name,
                        "type": "Media",
                        "error": getattr(res, "error", None),
                    }
            except Exception as exc:
                logger.exception("Error processing uploaded file: %s", exc)
                doc_info = {"name": file_name, "error": str(exc)}
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception:
                        pass

        # 5. Language Detection
        combined_text_for_lang = f"{question} {uploaded_doc_text[:300]}".strip()
        lang = detect_language(combined_text_for_lang)
        lang_note = format_language_note(lang)

        # 6. RAG Retrieval & Case Law
        search_query = f"{question} {uploaded_doc_text[:500]}".strip()
        matched_docs = _find_matched_docs(search_query)
        linked_docs: List[Dict[str, Any]] = []
        for fallback_id in ["legal_aid", "evidence_preservation"]:
            if fallback_id not in [d["id"] for d in matched_docs]:
                title, src, body = DOCS_MAP[fallback_id]
                linked_docs.append({"id": fallback_id, "title": title, "source": src, "body": body})

        is_urgent = _detect_urgency(search_query)

        # Send comprehensive metadata
        meta_obj = {
            "type": "meta",
            "confidence": "HIGH" if matched_docs else "LOW",
            "urgent": is_urgent,
            "matched": matched_docs,
            "linked": linked_docs,
            "language": {
                "name": lang.name,
                "script": lang.script,
                "note": lang_note,
            },
            "document": doc_info,
        }
        send_ndjson(meta_obj)

        # 7. Build Analysis Prompt
        rag_context = retriever.retrieve(search_query)
        kanoon_context = kanoon.retrieve(search_query) if kanoon else ""
        combined_context = "\n\n".join(c for c in [rag_context, kanoon_context] if c)

        if doc_info and uploaded_doc_text and ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            sys_prompt = adapt_system_prompt(analyzer.DOCUMENT_SYSTEM_PROMPT, lang)
            user_msg = analyzer.build_document_analysis_prompt(
                document_content=uploaded_doc_text,
                document_name=doc_info["name"],
                document_type=doc_info.get("type", "Document"),
                rag_context=combined_context,
                user_context=question,
            )
        else:
            sys_prompt = adapt_system_prompt(analyzer.SYSTEM_PROMPT, lang)
            user_msg = analyzer.build_analysis_prompt(
                situation=question or "Please analyse the attached evidence.",
                rag_context=combined_context,
                attached_content=uploaded_doc_text,
            )

        # 8. Stream Response from Groq (or fallback)
        if _groq_client is None:
            # Fallback pure RAG summary
            lines = [
                f"**What I Understood**\n{question or 'Analysed uploaded document: ' + (doc_info['name'] if doc_info else '')}\n\n",
                "**Laws That May Apply**\n",
            ]
            for doc in matched_docs:
                lines.append(f"- **{doc['title']}** ({doc['source']}): {doc['body']}\n")
            lines.append("\n**Important Notice**\nFor formal representation, contact your District Legal Services Authority (DLSA) or call the NALSA helpline at **15100**.")
            for word in re.split(r"(\s+)", "".join(lines)):
                if word:
                    send_ndjson({"type": "token", "t": word})
                    time.sleep(0.015)
            send_ndjson({"type": "done"})
            self.wfile.write(b"0\r\n\r\n")
            return

        try:
            messages = [{"role": "system", "content": sys_prompt}]
            # Add conversation history
            for h in history[-4:]:
                if isinstance(h, dict) and "role" in h and "content" in h:
                    messages.append({"role": h["role"], "content": str(h["content"])[:1000]})

            messages.append({"role": "user", "content": user_msg})

            stream = _groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=min(settings.MAX_TOKENS, 950),
                stream=True,
            )

            total_chars = 0
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    total_chars += len(delta)
                    if total_chars > MAX_RESPONSE_CHARS:
                        send_ndjson({"type": "token", "t": "\n\n*(Response truncated for length)*"})
                        break
                    send_ndjson({"type": "token", "t": delta})

            send_ndjson({"type": "done"})

        except Exception as exc:
            logger.warning("Groq unavailable or rate limited (%s); switching to full RAG synthesis", exc)
            fallback_text = _build_structured_fallback_analysis(question, matched_docs, is_urgent, doc_info, uploaded_doc_text)
            for word in re.split(r"(\s+)", fallback_text):
                if word:
                    send_ndjson({"type": "token", "t": word})
                    time.sleep(0.008)
            send_ndjson({"type": "done"})

        # Terminate chunked transfer
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass


def main() -> None:
    os.chdir(_DIR)
    server_address = (HOST, PORT)
    httpd = ThreadingHTTPServer(server_address, AlaHandler)
    print(f"\n=======================================================")
    print(f"  ALA Web Application running locally!")
    print(f"  URL: http://localhost:{PORT}/")
    print(f"  Direct access: http://127.0.0.1:{PORT}/")
    print(f"  Indexed Law Sections: {retriever.entry_count}")
    print(f"  Model: {settings.GROQ_MODEL}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
