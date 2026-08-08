from __future__ import annotations

import json
import re

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


def _domain_asset_risks(state: DueDiligenceState) -> DueDiligenceState:
    """Return domain-resale risk factors when the input is not an operating company."""
    company = state["overview"]

    risks = [
        f"{company.name} is not an operating company — {company.website} appears to be a "
        "parked or for-sale domain with no team, product, or funding history to evaluate.",
        "No comparable domain sales were collected for this asset, so the listed asking "
        "price (if any) is unverified against actual market transactions.",
        "No traffic, backlink, or historical-use data was collected — if the domain is "
        "unused, its value is largely speculative rather than demonstrated.",
    ]

    name_slug = re.sub(r'[^a-z0-9]', '', company.name.lower())
    domain_slug = re.sub(r'[^a-z0-9]', '', company.website.lower())
    if name_slug and name_slug in domain_slug:
        risks.append(
            f"The domain name is similar to '{company.name}', which may be a registered "
            "trademark of an unrelated existing company. Commercial use or resale under "
            "that name could carry trademark/legal risk — this can narrow the legitimate "
            "buyer pool rather than expand it, since it makes the asset less useful to "
            "anyone other than the trademark holder."
        )

    risks.append(
        "Any thesis that a specific well-known company 'might want' this domain is "
        "speculative unless there is direct evidence of interest — a single-buyer thesis "
        "deserves a large liquidity discount, not a premium."
    )

    return {
        "risk_factors": risks,
        "news_mentions": [],
        "completed_agents": state.get("completed_agents", []) + ["risks"]
    }


def assess_risks(state: DueDiligenceState) -> DueDiligenceState:
    """Search for negative signals and return risk factors."""

    company = state["overview"]

    if not state.get("is_operating_company", True):
        print("[risks] using domain-asset risk framing — not an operating company (parked/for-sale domain)")
        return _domain_asset_risks(state)

    negative_queries = [
        f"{company.name} controversy lawsuit problem issue",
        f"{company.name} layoffs shutdown pivot funding failed",
        f"{company.name} negative review complaint criticism"
    ]

    news_mentions: list[str] = []
    negative_content: list[str] = []

    for query in negative_queries:
        try:
            results = tavily.search(query=query, max_results=2)
            for r in results.get("results", []):
                content = r.get("content", "")
                if content and company.name.lower() in content.lower():
                    snippet = content[:300]
                    negative_content.append(snippet)
                    news_mentions.append(f"{r.get('url', '')}: {snippet}")
        except Exception:
            pass

    combined_negative = "\n\n".join(negative_content) if negative_content else "No negative signals found publicly."

    team_summary = "No team information found."
    if state.get("team_members"):
        team_summary = f"{len(state['team_members'])} team members identified: " + \
            ", ".join([f"{m.name} ({m.role})" for m in state["team_members"][:3]])

    funding_summary = "No funding information found."
    if state.get("funding_history"):
        rounds = state["funding_history"]
        funding_summary = f"{len(rounds)} funding round(s): " + \
            ", ".join([f"{r.round} {r.amount or 'undisclosed'}" for r in rounds])

    competitor_summary = "No competitor data available."
    if state.get("competitors"):
        competitor_summary = f"{len(state['competitors'])} competitors identified: " + \
            ", ".join([c.name for c in state["competitors"]])

    prompt = f"""You are a senior VC risk analyst performing due diligence on {company.name}.

COMPANY PROFILE:
- Name: {company.name}
- Description: {company.description}
- Business model: {company.business_model}
- Location: {company.location}
- Founded: {company.founded_year or "Unknown"}

INTELLIGENCE GATHERED:
- Team: {team_summary}
- Funding: {funding_summary}
- Competitors: {competitor_summary}
- Market: {state.get("market_analysis", "Not analysed")[:300]}

NEGATIVE SIGNALS FROM NEWS/WEB:
{combined_negative}

Generate 6-8 specific, evidence-based risk factors for investing in {company.name}.

Cover these risk categories (use only the ones relevant):
1. Market risk — is the market too crowded, too niche, or shrinking?
2. Competition risk — are there well-funded incumbents?
3. Team risk — gaps in founding team, high turnover, missing CTO/CEO?
4. Execution risk — is the product unproven, early stage, complex to build?
5. Funding risk — runway concerns, no institutional backing?
6. Regulatory risk — data privacy, financial regulation, sector-specific?
7. Reputational risk — any controversies, negative press?
8. Technology risk — dependent on third-party APIs, model providers?

Return ONLY a valid JSON array of strings, no markdown fences:
[
  "Risk factor 1 — specific and evidence-based",
  "Risk factor 2 — specific and evidence-based"
]

Rules:
- Each risk must be specific to {company.name}, not generic
- Back each risk with evidence where available
- Do NOT invent risks not supported by data
- Keep each risk under 2 sentences"""

    response = llm.invoke(prompt)

    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        risks = json.loads(clean)
        if not isinstance(risks, list):
            risks = []
        risks = [str(r) for r in risks if r]
    except (json.JSONDecodeError, TypeError):
        risks = [
            "Insufficient public data to fully assess risk profile",
            "Limited team transparency — unable to verify founding team credentials",
            "No public funding data — runway and investor backing unclear"
        ]

    return {
        "risk_factors": risks,
        "news_mentions": news_mentions,
        "completed_agents": state.get("completed_agents", []) + ["risks"]
    }
