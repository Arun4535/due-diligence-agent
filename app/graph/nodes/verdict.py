"""Verdict generation node for Due Diligence Agent."""
from __future__ import annotations

import json
import re
from typing import Literal

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

# Valid verdict values
VALID_VERDICTS: set[str] = {"PASS", "WATCH", "AVOID"}


def _summarise_team(state: DueDiligenceState) -> str:
    members = state.get("team_members", [])
    if not members:
        return "No team information found."
    lines = [f"- {m.name} ({m.role}): {m.background[:120]}" for m in members[:4]]
    return f"{len(members)} member(s) identified:\n" + "\n".join(lines)


def _summarise_funding(state: DueDiligenceState) -> str:
    rounds = state.get("funding_history", [])
    if not rounds:
        return "No funding information found publicly."
    lines = [
        f"- {r.round}: {r.amount or 'undisclosed'} "
        f"({r.date or 'date unknown'}) "
        f"— {', '.join(r.investors) if r.investors else 'investors undisclosed'}"
        for r in rounds
    ]
    return f"{len(rounds)} round(s):\n" + "\n".join(lines)


def _summarise_competitors(state: DueDiligenceState) -> str:
    comps = state.get("competitors", [])
    if not comps:
        return "No competitor data available."
    lines = [f"- {c.name} ({c.website}): {c.differentiator}" for c in comps]
    return "\n".join(lines)


def _summarise_risks(state: DueDiligenceState) -> str:
    risks = state.get("risk_factors", [])
    if not risks:
        return "No risk factors identified."
    return "\n".join([f"- {r}" for r in risks])


def _generate_domain_asset_verdict(state: DueDiligenceState) -> DueDiligenceState:
    """Verdict path for inputs that resolve to a parked/for-sale domain rather
    than an operating company. Deliberately uses a domain-investing framing
    instead of the startup-investment memo prompt, so the model isn't nudged
    into producing TAM figures, team/funding assessments, or false-precision
    claims (e.g. exact buyer-pool size, "litigation would be immediate") for
    an asset that has none of that underlying data."""

    company = state["overview"]
    raw_content = state.get("raw_website_content", "")
    risk_summary = _summarise_risks(state)

    prompt = f"""You are a domain-investing analyst, NOT a venture capital analyst.

The input "{state.get('startup_input', company.name)}" resolved to {company.website},
which appears to be a PARKED OR FOR-SALE DOMAIN rather than an operating company —
there is no team, product, revenue, or funding history to evaluate.

Page content found at this domain:
{raw_content[:2000]}

Known risk factors already identified:
{risk_summary}

Evaluate this strictly as a speculative digital asset (domain resale), not a startup.
Do not invent or assume: comparable sale prices, current asking price, live traffic,
specific interested buyers, or the likelihood/timing of legal action. If evidence for
something isn't in the content above, say the automated pass could not verify it —
don't state it as fact.

Return ONLY valid JSON, no markdown fences, no preamble:
{{
  "verdict": "PASS or WATCH or AVOID",
  "reasoning": "3-4 sentences. State plainly that this is a domain asset, not an operating company. Note that no comparable sales, traffic, or buyer-demand evidence was collected in this automated pass, so any asking price is unverified. If the name closely resembles an existing trademarked brand, flag that as risk that could narrow rather than expand the buyer pool — don't assert litigation is imminent. Recommend manual verification (comparable sales, traffic data, trademark scope) before any offer.",
  "strengths": ["at most 2 factual, non-speculative observations — e.g. brand-adjacent name recognition. Do not claim a specific company wants to buy it unless there is direct evidence in the content above."],
  "concerns": ["3-5 specific concerns, e.g.: no comparable sales data, no traffic/revenue evidence, potential trademark risk, illiquid/single-buyer thesis, unverified asking price"],
  "confidence_score": 0.0
}}

confidence_score must be low (0.2-0.4): an automated pass cannot verify comparable
sales, live traffic, or true buyer demand for a domain asset."""

    response = llm.invoke(prompt)

    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        data = json.loads(clean)

        verdict = str(data.get("verdict", "AVOID")).upper().strip()
        if verdict not in VALID_VERDICTS:
            verdict = "AVOID"

        confidence = float(data.get("confidence_score", 0.3))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get("reasoning", ""))
        strengths = [str(s) for s in data.get("strengths", []) if s][:5]
        concerns = [str(c) for c in data.get("concerns", []) if c][:5]

    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        verdict = "AVOID"
        confidence = 0.25
        reasoning = (
            f"'{state.get('startup_input', company.name)}' resolves to {company.website}, "
            "a parked or for-sale domain rather than an operating company. No comparable "
            "sales, traffic, or buyer-demand evidence is available from this automated "
            "pass, so any asking price cannot be validated. Manual domain-valuation "
            "research is required before treating this as an investable asset."
        )
        strengths = []
        concerns = [
            "No comparable domain sales data available",
            "No evidence of traffic, backlinks, or revenue",
            "Potential trademark/brand-similarity risk if the name resembles an existing company",
        ]

    return {
        "verdict": verdict,
        "verdict_reasoning": reasoning,
        "verdict_strengths": strengths,
        "verdict_concerns": concerns,
        "confidence_score": confidence,
        "completed_agents": state.get("completed_agents", []) + ["verdict"]
    }


def generate_verdict(state: DueDiligenceState) -> DueDiligenceState:
    """Synthesise all agent outputs into a final investment
    verdict of PASS, WATCH, or AVOID with structured reasoning."""

    if not state.get("is_operating_company", True):
        print("[verdict] using domain-asset verdict framing — not an operating company (parked/for-sale domain)")
        return _generate_domain_asset_verdict(state)

    company = state["overview"]

    # ── Build full memo context ────────────────────────────────────
    team_summary      = _summarise_team(state)
    funding_summary   = _summarise_funding(state)
    competitor_summary = _summarise_competitors(state)
    risk_summary      = _summarise_risks(state)
    market_analysis   = state.get("market_analysis", "Market analysis not available.")
    news              = state.get("news_mentions", [])
    news_summary      = "\n".join(news[:3]) if news else "No notable news found."

    # ── Scoring heuristics passed to Claude ───────────────────────
    # Give Claude explicit signals to reason about
    data_quality_signals = []
    if state.get("team_members"):
        data_quality_signals.append(f"✓ Team data found ({len(state['team_members'])} members)")
    else:
        data_quality_signals.append("✗ No team data found — opaque founding team")

    if state.get("funding_history"):
        data_quality_signals.append(f"✓ Funding data found ({len(state['funding_history'])} rounds)")
    else:
        data_quality_signals.append("✗ No funding data — bootstrapped or pre-funding")

    if state.get("competitors"):
        data_quality_signals.append(f"⚠ {len(state['competitors'])} competitors identified")
    else:
        data_quality_signals.append("✗ No competitors found — possibly niche or very new market")

    if news:
        data_quality_signals.append(f"⚠ {len(news)} negative signal(s) found in news")
    else:
        data_quality_signals.append("✓ No negative news signals")

    signals_text = "\n".join(data_quality_signals)

    # ── Main verdict prompt ────────────────────────────────────────
    prompt = f"""You are a senior VC partner at a top-tier venture fund.
You are writing the final verdict section of an investment memo for {company.name}.

═══════════════════════════════════════════
COMPANY
═══════════════════════════════════════════
Name:           {company.name}
Description:    {company.description}
Business model: {company.business_model}
Location:       {company.location}
Founded:        {company.founded_year or "Unknown"}

═══════════════════════════════════════════
TEAM
═══════════════════════════════════════════
{team_summary}

═══════════════════════════════════════════
FUNDING HISTORY
═══════════════════════════════════════════
{funding_summary}

═══════════════════════════════════════════
MARKET ANALYSIS
═══════════════════════════════════════════
{market_analysis[:600]}

═══════════════════════════════════════════
COMPETITORS
═══════════════════════════════════════════
{competitor_summary}

═══════════════════════════════════════════
RISK FACTORS
═══════════════════════════════════════════
{risk_summary}

═══════════════════════════════════════════
NEWS & SIGNALS
═══════════════════════════════════════════
{news_summary}

═══════════════════════════════════════════
DATA QUALITY SIGNALS
═══════════════════════════════════════════
{signals_text}

═══════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════
Based on ALL of the above, provide a final investment verdict.

Verdict criteria:
- PASS  → Strong team + large market + clear differentiation + manageable risk
- WATCH → Promising but missing key data, early stage, or crowded market
- AVOID → Weak team signal, declining market, major red flags, or high competition with no moat

Return ONLY valid JSON, no markdown fences, no preamble:
{{
  "verdict": "PASS or WATCH or AVOID",
  "reasoning": "3-4 sentences explaining the verdict. Be specific — reference actual data points from above (team names, funding amounts, competitor names, market size). Do not be generic.",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "concerns": ["concern 1", "concern 2", "concern 3"],
  "confidence_score": 0.0
}}

confidence_score rules:
- 0.8–1.0: strong data across all sections, clear verdict
- 0.6–0.79: moderate data, some gaps but verdict is supportable
- 0.4–0.59: significant data gaps, verdict is tentative
- 0.2–0.39: very limited data, verdict is speculative
- confidence must be a float between 0.0 and 1.0"""

    response = llm.invoke(prompt)

    # ── Parse response ─────────────────────────────────────────────
    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        data = json.loads(clean)

        # Validate verdict
        verdict = str(data.get("verdict", "WATCH")).upper().strip()
        if verdict not in VALID_VERDICTS:
            verdict = "WATCH"

        # Validate confidence
        confidence = float(data.get("confidence_score", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        reasoning = str(data.get("reasoning", "Insufficient data for a confident verdict."))
        strengths = [str(s) for s in data.get("strengths", []) if s][:5]
        concerns  = [str(c) for c in data.get("concerns", []) if c][:5]

    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        # Safe fallback — never crash on verdict node
        verdict    = "WATCH"
        confidence = 0.3
        reasoning  = (
            f"Unable to parse a structured verdict for {company.name}. "
            "Insufficient public data was available across team, funding, "
            "and market sections to make a confident investment recommendation. "
            "Manual due diligence is strongly advised."
        )
        strengths = []
        concerns  = ["Insufficient public data", "Unable to verify team credentials", "No funding history found"]

    return {
        "verdict":          verdict,
        "verdict_reasoning": reasoning,
        "verdict_strengths": strengths,
        "verdict_concerns":  concerns,
        "confidence_score": confidence,
        "completed_agents": state.get("completed_agents", []) + ["verdict"]
    }
