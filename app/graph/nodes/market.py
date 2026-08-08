from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from app.graph.state import DueDiligenceState

import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def analyse_market(state: DueDiligenceState) -> DueDiligenceState:
    """Search for market size, growth rate, and trends
    relevant to the startup's business model and domain."""

    company = state["overview"]

    # TAM / CAGR / "market growth rate" is a startup-analysis concept. It
    # doesn't meaningfully describe a parked/for-sale domain — citing "the
    # global domain market is worth $X" says nothing about whether THIS
    # domain is worth its asking price (see: land value vs. real-estate
    # market size). Skip the search entirely and say so explicitly instead
    # of generating a plausible-looking but irrelevant TAM section.
    if not state.get("is_operating_company", True):
        note = (
            f"{company.name} does not appear to be an operating company — "
            f"{company.website} resolves to a parked or for-sale domain rather than a "
            "live product or business. Standard market-sizing (TAM, CAGR, market dynamics) "
            "does not apply to a domain-resale asset and has been skipped. Valuing this "
            "domain instead requires comparable domain sales (same length/category/TLD), "
            "current traffic and backlink data, and identifiable buyer demand — none of "
            "which this automated pass collects. Treat any asking price as unverified "
            "until that research is done manually."
        )
        return {
            "market_analysis": note,
            "completed_agents": state.get("completed_agents", []) + ["market"]
        }

    # ── Step 1: Build targeted search queries ─────────────────────
    queries = [
        f"{company.business_model} market size TAM 2024 2025 billion",
        f"{company.name} industry market growth rate trends",
        f"{company.business_model} total addressable market forecast"
    ]

    all_content: list[str] = []
    for query in queries:
        try:
            results = tavily.search(query=query, max_results=3)
            for r in results.get("results", []):
                content = r.get("content", "")
                if content:
                    all_content.append(f"Source: {r.get('url', '')}\n{content[:600]}")
        except Exception:
            pass

    combined = "\n\n".join(all_content) if all_content else "No market data found."

    # ── Step 2: Generate market analysis with Claude ───────────────
    prompt = f"""You are a senior VC analyst writing the market analysis section
of an investment memo for {company.name}.

Company description: {company.description}
Business model: {company.business_model}
Location: {company.location}

Market research data:
{combined[:4000]}

Write a structured market analysis covering exactly these 4 sections:

1. Total Addressable Market (TAM)
   - Specific market size in dollars with source year
   - If exact TAM not found, estimate based on adjacent market data

2. Market Growth Rate
   - YoY or CAGR growth rate with numbers
   - Key growth drivers

3. Market Dynamics
   - Is this market growing, mature, or declining?
   - Key trends shaping the space (AI, regulation, consumer behaviour)

4. Opportunity for {company.name}
   - What share of the market is realistically addressable?
   - Why now — what makes this the right time?

Be specific with numbers. Clearly flag estimates vs confirmed data.
Write in professional memo style. 4-6 paragraphs total."""

    response = llm.invoke(prompt)

    return {
        "market_analysis": response.content,
        "completed_agents": state.get("completed_agents", []) + ["market"]
    }
