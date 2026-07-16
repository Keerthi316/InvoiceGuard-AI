# AP Invoice & Contract Exception Agent

A production-ready Accounts Payable automation system that uses **LangGraph**, **LLM extraction**, **Pydantic validation**, and a **Streamlit UI** to automate invoice processing, PO matching, contract compliance checking, and exception routing — with a complete audit trail.

---

## Features

| Feature | Details |
|---|---|
| **Document Extraction** | PDF (text layer + OCR fallback), PNG/JPG, DOCX, TXT/JSON/CSV |
| **LLM Extraction** | Invoice, PO, Contract fields via structured JSON prompt |
| **Schema Validation** | Pydantic v2 — rejects incomplete/invalid invoices |
| **PO Matching** | Vendor, PO number, quantities, unit prices, totals |
| **Contract Compliance** | Contracted prices, approval thresholds, expiry, status |
| **Exception Routing** | PRICE_VARIANCE, MISSING_APPROVAL, UNKNOWN_VENDOR, and 9 more |
| **Decision Gate** | STP (Straight Through Processing) vs Human Review |
| **Audit Trail** | JSONL — every field, every LLM call, tokens, cost, latency |
| **Prompt Injection Defense** | Heuristic detection + forced pipeline execution |
| **Duplicate Detection** | Same vendor+invoice number within 90-day window |
| **Dashboard** | STP rate, exception rate, top exception codes, cost |
| **Multi-Provider LLM** | OpenAI, Google Gemini, Anthropic Claude — UI-selectable |

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd capstone_project
pip install -r requirements.txt
```

> Python 3.11+ recommended.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
# OpenRouter (https://openrouter.ai/keys)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Any model slug from https://openrouter.ai/models
MODEL=google/gemini-2.5-flash
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload Invoice** (required) — PDF, image, DOCX, or text file
2. **Upload Purchase Order** (optional) — enables PO matching
3. **Upload Contract** (optional) — enables contract compliance checking
4. Select your **LLM Provider** and **Model** in the sidebar
5. Click **Analyze Invoice**
6. Review results across 6 tabs: Extracted Fields, Validation, Matching, Exceptions, Decision, Audit

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)              │
└─────────────────────┬───────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │  InvoiceProcessingWorkflow  │  ← LangGraph StateGraph
         └────────────┬────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
  DocumentProcessing  Extraction  Validation
  Agent              Agent       Agent
  (OCR + injection   (LLM JSON   (Pydantic
  detection)         parsing)    rules)
          │
          ▼
  MatchingAgent       ←── PO + Contract arithmetic
          │
          ▼
  ExceptionRoutingAgent  ←── aggregates + prioritizes
          │
          ▼
  DecisionAgent       ←── STP vs HUMAN_REVIEW vs REJECTED
          │
          ▼
  AuditRecord saved to  audit/audit_log.jsonl
```

### Pipeline Order (enforced — cannot be bypassed by document content)

```
document_processing → extraction → validation → matching → exception_routing → decision → audit
```

---

## Project Structure

```
capstone_project/
├── app.py                          # Streamlit UI
├── .env.example                    # Environment template
├── requirements.txt
├── README.md
│
├── agents/                         # One agent per file
│   ├── base_agent.py
│   ├── document_processing_agent.py
│   ├── extraction_agent.py
│   ├── validation_agent.py
│   ├── matching_agent.py
│   ├── exception_routing_agent.py
│   └── decision_agent.py
│
├── models/                         # Pydantic schemas
│   ├── invoice.py
│   ├── purchase_order.py
│   └── audit.py
│
├── services/
│   ├── llm_service.py              # OpenAI / Gemini / Anthropic wrapper
│   └── workflow.py                 # LangGraph pipeline
│
├── prompts/
│   └── extraction_prompts.py       # System + user prompts with injection defense
│
├── utils/
│   ├── document_processor.py       # PDF / OCR / DOCX extraction
│   └── helpers.py                  # JSON parsing, sanitization
│
├── audit/
│   └── audit_service.py            # JSONL persistence, stats, duplicate detection
│
├── config/
│   └── settings.py                 # pydantic-settings, typed env loading
│
├── tests/
│   ├── conftest.py                 # Shared fixtures, mock LLM
│   ├── test_scenario_clean_invoice.py
│   ├── test_scenario_price_variance.py
│   ├── test_scenario_missing_approval.py
│   ├── test_scenario_malformed_extraction.py
│   └── test_scenario_prompt_injection.py
│
├── logs/                           # Structured logs (gitignored)
├── audit/                          # audit_log.jsonl (gitignored)
└── data/                           # Optional reference data
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `MODEL` | `google/gemini-2.5-flash` | Model slug from openrouter.ai/models |
| `PRICE_VARIANCE_THRESHOLD` | `5.0` | Max % price variance before exception |
| `QUANTITY_VARIANCE_THRESHOLD` | `2.0` | Max % quantity variance before exception |
| `TESSERACT_CMD` | — | Path to Tesseract binary (for OCR) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Exception Codes

| Code | Trigger |
|---|---|
| `PRICE_VARIANCE` | Invoice price > PO price beyond threshold |
| `QUANTITY_VARIANCE` | Invoice quantity differs from PO quantity |
| `TOTAL_MISMATCH` | Invoice total doesn't match PO total |
| `MISSING_APPROVAL` | Amount exceeds threshold but no approver on PO/contract |
| `UNKNOWN_VENDOR` | Invoice vendor not recognized in PO/contract |
| `OFF_CONTRACT_TERMS` | Price outside contracted tolerance or exceeds max order value |
| `INVALID_EXTRACTION` | Validation failed — required fields missing or invalid |
| `MALFORMED_INVOICE` | Pydantic schema violations |
| `CONTRACT_EXPIRED` | Contract is expired or not ACTIVE status |
| `PO_CLOSED` | PO status is CLOSED or CANCELLED |
| `DUPLICATE_INVOICE` | Same vendor + invoice number within 90 days |
| `PROMPT_INJECTION_DETECTED` | Adversarial instructions found in uploaded document |

---

## Decision Logic

```
ALL of these must be true for STP:
  ✓ Invoice extraction succeeded
  ✓ All required fields present and valid
  ✓ Vendor matches PO / contract
  ✓ PO number matches
  ✓ Quantities within tolerance
  ✓ Unit prices within tolerance
  ✓ Total amount within tolerance
  ✓ Contract is ACTIVE and not expired
  ✓ Approval threshold met (or not applicable)
  ✓ No injection detected

ANY failure → HUMAN_REVIEW → payment BLOCKED
```

---

## Running Tests

```bash
# All tests (no API key required — uses mock LLM)
pytest tests/ -v

# Single scenario
pytest tests/test_scenario_clean_invoice.py -v
pytest tests/test_scenario_prompt_injection.py -v
```

---

## Security

- **API keys** are loaded from `.env` only — never hardcoded
- **Document content** is always placed in the user role with explicit boundary markers (`<DOCUMENT>...</DOCUMENT>`) — the system prompt never includes untrusted data
- **Prompt injection** is detected heuristically; any match adds `PROMPT_INJECTION_DETECTED` and routes to human review — the pipeline still runs all stages
- **Pipeline order** is enforced by the LangGraph graph structure — no embedded instruction can skip a stage
- All LLM outputs are parsed as JSON data, never executed

---

## OCR Support (Optional)

For scanned PDFs and image invoices:

1. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
2. Install [Poppler](https://poppler.freedesktop.org/) (for pdf2image)
3. Set in `.env`:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Text-layer PDFs work without Tesseract.

---

## Audit Log Format

Each line in `audit/audit_log.jsonl` is a complete `AuditRecord`:

```jsonc
{
  "audit_id": "uuid",
  "invoice_id": "INV-XXXXXXXX",
  "session_id": "uuid",
  "processing_started_at": "ISO datetime",
  "extracted_fields": { /* all invoice fields */ },
  "validation_report": { "is_valid": true, "field_results": [...] },
  "matching_report": { "overall_match": true, "exceptions": [] },
  "exception_report": { "exception_codes": [], "priority": "LOW" },
  "decision": { "decision": "STP", "payment_scheduled": true },
  "llm_calls": [
    { "agent_name": "extraction_agent", "prompt_tokens": 400,
      "completion_tokens": 200, "latency_ms": 800, "estimated_cost_usd": 0.0001 }
  ],
  "total_tokens_used": 600,
  "total_cost_usd": 0.0001,
  "total_latency_ms": 800,
  "llm_model": "gpt-4o-mini",
  "llm_provider": "openai"
}
```

---

## Extending the System

**Add a new LLM provider:**
1. Create a new client class in `services/llm_service.py` extending `BaseLLMClient`
2. Add it to `LLMService._build_client()` and `settings.available_models`

**Add a new exception code:**
1. Add to `ExceptionCode` enum in `models/audit.py`
2. Add detection logic in `agents/matching_agent.py` or `agents/validation_agent.py`
3. Add priority to `_PRIORITY_MAP` in `agents/exception_routing_agent.py`

**Add a new validation rule:**
1. Add a check in `agents/validation_agent.py` → `run()`
2. No other files need to change — the pipeline is already wired

---

## License

MIT
