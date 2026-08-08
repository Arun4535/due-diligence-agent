# Due Diligence Agent

An AI-powered startup due-diligence pipeline. Give it a company name or URL and it scrapes the company site, researches team/funding/market/competitor data, assesses risk, and produces a structured investment verdict — plus a downloadable PDF memo.

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/), [Firecrawl](https://firecrawl.dev/), [Tavily](https://tavily.com/), and Claude.

---

## How it works

```
startup input (name or URL)
        │
        ▼
   ┌─────────┐
   │ scrape  │  resolve + scrape the company website
   └────┬────┘  → flags parked/for-sale domains and unresolvable inputs
        ▼
   ┌─────────┐
   │  team   │  find founders/execs
   └────┬────┘
        ▼
   ┌─────────┐
   │ funding │  find funding rounds
   └────┬────┘
        ▼
   ┌─────────┐
   │ market  │  TAM, growth rate, dynamics
   └────┬────┘
        ▼
   ┌─────────┐
   │competitors│  identify and verify competitors
   └────┬────┘
        ▼
   ┌─────────┐
   │  risks  │  synthesize risk factors
   └────┬────┘
        ▼
   ┌─────────┐
   │ verdict │  PASS / WATCH / AVOID + confidence score
   └────┬────┘
        ▼
   ┌─────────┐
   │ report  │  generate PDF memo
   └─────────┘
```

Each step is a LangGraph node operating on a shared `DueDiligenceState`. See `app/graph/builder.py` for the graph definition.

### Non-operating-company handling

If the input resolves to a parked or for-sale domain (or a domain that returns no verifiable content at all), the pipeline sets `is_operating_company: False` on the state. Every downstream node checks this flag and skips analysis that doesn't apply to a non-operating domain (team search, funding search, TAM/market sizing, competitor search) instead of quietly generating a plausible-looking but fabricated startup report. The risk and verdict nodes switch to domain-asset-specific framing (unverified pricing, trademark exposure, illiquidity) rather than VC-memo framing.

---

## Setup

### Prerequisites
- Python 3.11+
- API keys for:
  - `ANTHROPIC_API_KEY` — Claude (analysis + extraction)
  - `TAVILY_API_KEY` — web search
  - `FIRECRAWL_API_KEY` — website scraping

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
FIRECRAWL_API_KEY=fc-...
```

### Run

Start the API:

```bash
uvicorn app.api.routes:app --reload --port 8000
```

Start the UI (in a separate terminal):

```bash
streamlit run streamlit_app.py
```

Or via Docker Compose:

```bash
docker compose up --build
```

---

## API

| Method | Path              | Description                                  |
|--------|-------------------|-----------------------------------------------|
| GET    | `/`                | Health message                                |
| GET    | `/health`          | Health check                                  |
| POST   | `/analyse`         | Run the full pipeline on `{"startup_input": "..."}` |
| GET    | `/report/{filename}` | Download a generated PDF memo               |

Example:

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"startup_input": "Linear"}'
```

Response includes `verdict`, `confidence_score`, `verdict_reasoning`, counts for team/funding/competitors, `risk_factors`, `market_analysis`, `pdf_path`, and `is_operating_company`.

---

## Project structure

```
app/
  api/
    routes.py           FastAPI app + endpoints
  graph/
    builder.py           LangGraph wiring
    state.py              Shared state schema
    nodes/
      scraper.py           Resolve + scrape company site, detect non-operating domains
      team.py               Extract founders/execs
      funding.py            Extract funding history
      market.py             Market sizing & dynamics
      competitors.py        Competitor research
      risks.py               Risk factor synthesis
      verdict.py             Final PASS/WATCH/AVOID verdict
  report/
    generator.py           PDF memo generation (ReportLab)
streamlit_app.py          Streamlit UI
reports/                   Generated PDFs (gitignored — not committed)
```

---

## Known limitations

- Fully automated — no human review of any output. Treat every result as a first pass.
- Web scraping and search results can be incomplete, stale, or wrong; the agent reports what it found, not ground truth.
- No authentication on the API — don't expose it publicly without adding some.
- `reports/` is not cleaned up automatically; PDFs accumulate over time.
