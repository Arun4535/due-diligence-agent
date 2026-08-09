import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from app.graph.builder import build_graph

app = FastAPI(title="Due Diligence Agent", version="1.0.0")
graph = build_graph()

REPORTS_DIR = os.path.abspath("reports")

# ── Simple manual rate limiter — 1 request per 5 minutes per IP ────
RATE_LIMIT_SECONDS = 5 * 60
_last_request_time: dict[str, float] = {}


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    last_time = _last_request_time.get(client_ip)
    if last_time is not None and (now - last_time) < RATE_LIMIT_SECONDS:
        wait_seconds = int(RATE_LIMIT_SECONDS - (now - last_time))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Please wait {wait_seconds} seconds before trying again."
        )
    _last_request_time[client_ip] = now


class DiligenceRequest(BaseModel):
    startup_input: str  # name or URL

class DiligenceResponse(BaseModel):
    company_name: str
    verdict: str
    confidence_score: float
    verdict_reasoning: str
    team_count: int
    funding_rounds: int
    competitor_count: int
    risk_count: int
    pdf_path: str
    market_analysis: str = ""
    risk_factors: List[str] = []
    is_operating_company: bool = True

@app.get("/")
async def root():
    return {"message": "Due Diligence Agent API is running"}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}

@app.post("/analyse", response_model=DiligenceResponse)
async def analyse(http_request: Request, request: DiligenceRequest):
    client_ip = http_request.client.host if http_request.client else "unknown"
    _check_rate_limit(client_ip)

    if not request.startup_input or not request.startup_input.strip():
        raise HTTPException(status_code=400, detail="startup_input must not be empty")

    initial_state = {
        "startup_input": request.startup_input.strip(),
        "website_url": "",
        "raw_website_content": "",
        "is_operating_company": True,
        "team_members": [],
        "funding_history": [],
        "competitors": [],
        "market_analysis": "",
        "risk_factors": [],
        "news_mentions": [],
        "errors": [],
        "completed_agents": []
    }

    try:
        result = await run_in_threadpool(graph.invoke, initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return DiligenceResponse(
        company_name=result["overview"].name,
        verdict=result["verdict"],
        confidence_score=result["confidence_score"],
        verdict_reasoning=result["verdict_reasoning"],
        team_count=len(result["team_members"]),
        funding_rounds=len(result["funding_history"]),
        competitor_count=len(result["competitors"]),
        risk_count=len(result["risk_factors"]),
        pdf_path=result["memo_pdf_path"],
        market_analysis=result.get("market_analysis", ""),
        risk_factors=result.get("risk_factors", []),
        is_operating_company=result.get("is_operating_company", True)
    )

@app.get("/report/{filename}")
async def download_report(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.abspath(os.path.join(REPORTS_DIR, safe_name))
    if not file_path.startswith(REPORTS_DIR + os.sep) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=safe_name
    )