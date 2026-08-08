"""Funding history extraction node for Due Diligence Agent."""
from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from app.graph import state
from app.graph.state import DueDiligenceState, FundingRound

import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def extract_funding(state: DueDiligenceState) -> DueDiligenceState:
    """Search for funding history via Tavily and extract
    structured rounds using Claude."""

    # A parked/for-sale domain has no company behind it, so "funding rounds"
    # is not a meaningful concept — skip straight past it instead of running
    # searches that will pollute the report with noise from unrelated,
    # similarly-named companies.
    if not state.get("is_operating_company", True):
        print("[funding] skipping funding search — not an operating company (parked/for-sale domain)")
        return {
            "funding_history": [],
            "completed_agents": state.get("completed_agents", []) + ["funding"]
        }

    company_name = state["overview"].name if state.get("overview") else "the company"

    queries = [
        f"{company_name} funding round raised investors amount",
        f"{company_name} Crunchbase Series A Seed venture capital",
        f"{company_name} startup investment valuation"
    ]

    all_content: list[str] = []
    for query in queries:
        try:
            results = tavily.search(query=query, max_results=3)
            for r in results.get("results", []):
                if company_name.lower() in r["content"].lower():
                    all_content.append(r["content"][:500])
        except Exception:
            pass

    combined = "\n\n".join(all_content) if all_content else "No funding data found."

    prompt = f"""Extract funding history for {company_name} from this content.

Return ONLY a valid JSON array with no markdown fences, no preamble:
[
  {{
    "round": "Seed / Pre-Seed / Series A / Series B / etc",
    "amount": "$5M or null if unknown",
    "date": "Month Year or null if unknown",
    "investors": ["Investor Name 1", "Investor Name 2"]
  }}
]

Rules:
- Only include rounds you find evidence for in the content
- If investors are not mentioned, use an empty array []
- If no funding information is found at all, return []
- Do NOT invent or hallucinate funding data

Content:
{combined[:4000]}"""

    response = llm.invoke(prompt)

    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        rounds_data = json.loads(clean)
        if not isinstance(rounds_data, list):
            rounds_data = []
        rounds = [FundingRound(**r) for r in rounds_data]
    except (json.JSONDecodeError, TypeError, KeyError):
        rounds = []

    # ── Step 3: Also check website content for funding mentions ───
    if not rounds and state.get("raw_website_content"):
        website_prompt = f"""Does this website content mention any funding, investment,
or backing for {company_name}?

Return ONLY a valid JSON array (or [] if nothing found):
[
  {{
    "round": "round type or Unknown",
    "amount": null,
    "date": null,
    "investors": []
  }}
]

Content: {state['raw_website_content'][:2000]}"""

        website_response = llm.invoke(website_prompt)
        try:
            clean = re.sub(r"```json|```", "", website_response.content).strip()
            extra = json.loads(clean)
            if isinstance(extra, list):
                rounds.extend([FundingRound(**r) for r in extra])
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    return {
        "funding_history": rounds,
        "completed_agents": state.get("completed_agents", []) + ["funding"]
    }
