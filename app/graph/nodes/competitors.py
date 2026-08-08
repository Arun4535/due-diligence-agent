from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from app.graph.state import DueDiligenceState, Competitor

import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def find_competitors(state: DueDiligenceState) -> DueDiligenceState:
    """Find direct and indirect competitors, then score
    how each differs from the startup being analysed."""

    company = state["overview"]

    if not state.get("is_operating_company", True):
        print("[competitors] skipping competitor search — not an operating company (parked/for-sale domain)")
        return {
            "competitors": [],
            "completed_agents": state.get("completed_agents", []) + ["competitors"]
        }

    queries = [
        f"competitors alternatives to {company.name}",
        f"best {company.business_model} companies like {company.name}",
        f"{company.name} vs competitors comparison"
    ]

    all_content: list[str] = []
    for query in queries:
        try:
            results = tavily.search(query=query, max_results=3)
            for r in results.get("results", []):
                content = r.get("content", "")
                if content:
                    all_content.append(content[:500])
        except Exception:
            pass

    combined = "\n\n".join(all_content) if all_content else "No competitor data found."

    prompt = f"""You are a competitive intelligence analyst.
Identify 3-5 direct competitors to {company.name}.

Company: {company.name}
Description: {company.description}
Business model: {company.business_model}

Research data:
{combined[:4000]}

Return ONLY a valid JSON array with no markdown fences, no preamble:
[
  {{
    "name": "Competitor Company Name",
    "website": "https://competitor.com",
    "differentiator": "One sentence: how this competitor differs from {company.name} — what they do better or worse"
  }}
]

Rules:
- Only include real, verifiable companies
- Do NOT include {company.name} itself
- Prefer direct competitors over adjacent ones
- If fewer than 3 found, return what you have — do not invent companies
- Return [] if no competitors found"""

    response = llm.invoke(prompt)

    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        comp_data = json.loads(clean)
        if not isinstance(comp_data, list):
            comp_data = []
        competitors = [Competitor(**c) for c in comp_data]
    except (json.JSONDecodeError, TypeError, KeyError):
        competitors = []

    verified: list[Competitor] = []
    for comp in competitors[:3]:
        try:
            check = tavily.search(
                query=f"{comp.name} official website company",
                max_results=1
            )
            if check.get("results"):
                result_url = check["results"][0].get("url", "")
                if result_url and comp.name.lower().replace(" ", "") in result_url.lower():
                    comp.website = result_url
        except Exception:
            pass
        verified.append(comp)

    verified.extend(competitors[3:])

    return {
        "competitors": verified,
        "completed_agents": state.get("completed_agents", []) + ["competitors"]
    }
