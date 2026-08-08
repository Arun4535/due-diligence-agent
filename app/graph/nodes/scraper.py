"""Scraper node for Due Diligence Agent."""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
load_dotenv()

from firecrawl import FirecrawlApp
from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from app.graph.state import DueDiligenceState, CompanyOverview

llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY"))
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

DOMAIN_SALE_SIGNALS = [
    "domain for sale", "buy this domain", "this domain may be for sale",
    "domain name is for sale", "make an offer", "spaceship.com",
    "godaddy.com/domainfind", "sedo.com", "afternic.com",
    "domain broker", "premium domain",
]

MIN_CONTENT_LENGTH = 300
MAX_CANDIDATES_TO_CHECK = 3
CANDIDATE_TLDS = [".com", ".ai", ".io", ".co", ".app"]

KNOWN_DOMAINS: dict[str, str] = {
    "perplexity": "perplexity.ai",
    "perplexity ai": "perplexity.ai",
    "linear": "linear.app",
    "notion": "notion.so",
    "vercel": "vercel.com",
}

_BARE_DOMAIN_RE = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


def _looks_like_domain_sale(content: str) -> bool:
    lowered = content.lower()
    return any(signal in lowered for signal in DOMAIN_SALE_SIGNALS)


def _scrape_pages(website_url: str) -> str:
    pages = [
        website_url,
        f"{website_url.rstrip('/')}/about",
        f"{website_url.rstrip('/')}/pricing",
        f"{website_url.rstrip('/')}/team",
    ]
    raw: list[str] = []
    for url in pages:
        try:
            result = firecrawl.scrape_url(url, params={"formats": ["markdown"]})
            markdown = result.markdown if hasattr(result, "markdown") else result.get("markdown", "")
            if markdown:
                raw.append(f"## {url}\n{markdown[:3000]}")
        except Exception:
            pass
    return "\n\n".join(raw)


def _tavily_fallback_content(startup_name: str, website_url: str) -> str:
    """When direct scraping is blocked (common for large production sites
    like stripe.com), fall back to Tavily search summaries as a content
    source instead of declaring the company non-operating."""
    try:
        results = tavily.search(
            query=f"{startup_name} company overview product what they do",
            max_results=5
        )
        snippets = [r["content"][:600] for r in results.get("results", []) if r.get("content")]
        if snippets:
            return f"## Tavily search summary for {website_url}\n" + "\n\n".join(snippets)
    except Exception:
        pass
    return ""


def _normalise_input(raw_input: str) -> tuple[str, bool]:
    value = raw_input.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value, True
    check_value = value[4:] if value.lower().startswith("www.") else value
    if _BARE_DOMAIN_RE.match(check_value):
        return f"https://{value}", True
    return value, False


def _generate_tld_candidates(startup_name: str) -> list[str]:
    slug = re.sub(r"[^a-z0-9]", "", startup_name.lower())
    return [f"{slug}{tld}" for tld in CANDIDATE_TLDS] if slug else []


def _search_candidates(query: str) -> list[str]:
    candidates: list[str] = []
    try:
        search_result = firecrawl.search(query)
        data = search_result.data if hasattr(search_result, "data") else search_result.get("data", [])
        for item in data[:MAX_CANDIDATES_TO_CHECK]:
            url = item.url if hasattr(item, "url") else item.get("url", "")
            if url:
                candidates.append(url)
    except Exception:
        pass
    return candidates


def _classify(content: str) -> str:
    """Classify scraped content into one of three states."""
    if _looks_like_domain_sale(content):
        return "confirmed_domain_sale"
    if len(content) >= MIN_CONTENT_LENGTH:
        return "ok"
    return "insufficient_data"


def _resolve(raw_input: str) -> tuple[str, str, str]:
    """Returns (website_url, content, data_availability)."""
    normalised, is_direct = _normalise_input(raw_input)
    website_url = normalised
    startup_name = normalised if not is_direct else normalised.split("//")[-1].split(".")[0]

    if not is_direct:
        known = KNOWN_DOMAINS.get(startup_name.strip().lower())
        if known:
            website_url = f"https://{known}"

    content = _scrape_pages(website_url)
    status = _classify(content)

    # Confirmed domain sale — trust it immediately, no need to search harder
    if status == "confirmed_domain_sale":
        return website_url, content, status

    # Got good content directly — done
    if status == "ok":
        return website_url, content, status

    # ── insufficient_data: try alternatives before giving up ────────
    if not is_direct:
        # Try TLD variants
        for domain in _generate_tld_candidates(startup_name):
            alt_url = f"https://{domain}"
            alt_content = _scrape_pages(alt_url)
            alt_status = _classify(alt_content)
            if alt_status == "ok":
                return alt_url, alt_content, alt_status
            if alt_status == "confirmed_domain_sale":
                return alt_url, alt_content, alt_status

        # Try search
        for url in _search_candidates(f"{startup_name} official company website"):
            alt_content = _scrape_pages(url)
            alt_status = _classify(alt_content)
            if alt_status == "ok":
                return url, alt_content, alt_status
            if alt_status == "confirmed_domain_sale":
                return url, alt_content, alt_status

    # Direct scraping never worked — try Tavily as a content source
    # (this is what saves cases like stripe.com being bot-blocked)
    tavily_content = _tavily_fallback_content(startup_name, website_url)
    if tavily_content and len(tavily_content) >= MIN_CONTENT_LENGTH:
        return website_url, tavily_content, "ok"

    # Genuinely nothing found anywhere — insufficient data, NOT a claim
    # that it's a parked domain
    return website_url, content, "insufficient_data"


def scrape_website(state: DueDiligenceState) -> DueDiligenceState:
    startup_input = state["startup_input"]
    website_url, combined, data_availability = _resolve(startup_input)

    if data_availability == "ok":
        prompt = f"""Extract company information from this website content.
Return ONLY valid JSON, no markdown fences:
{{
    "name": "company name",
    "website": "{website_url}",
    "description": "2-3 sentence description",
    "founded_year": null,
    "location": "city, country",
    "business_model": "B2B SaaS / B2C / Marketplace / etc"
}}

Content:
{combined[:4000]}"""
        response = llm.invoke(prompt)
        try:
            clean = re.sub(r"```json|```", "", response.content).strip()
            overview = CompanyOverview(**json.loads(clean))
        except (json.JSONDecodeError, TypeError, KeyError):
            name = startup_input.split("//")[-1].split(".")[0].title() if startup_input.startswith("http") else startup_input
            overview = CompanyOverview(name=name, website=website_url, description="Could not extract description.", location="Unknown", business_model="Unknown")

    elif data_availability == "confirmed_domain_sale":
        name = startup_input.split("//")[-1].split(".")[0].title() if startup_input.startswith("http") else startup_input
        overview = CompanyOverview(
            name=name, website=website_url,
            description=f"{website_url} is listed as a domain for sale — confirmed by explicit marketplace language on the page.",
            location="Unknown", business_model="Unknown"
        )

    else:  # insufficient_data
        name = startup_input.split("//")[-1].split(".")[0].title() if startup_input.startswith("http") else startup_input
        overview = CompanyOverview(
            name=name, website=website_url,
            description=(
                f"Could not retrieve content from {website_url} — the site may block "
                "automated scraping (common for large production sites), or the "
                "company could not be confidently located. This is NOT evidence "
                "the domain is for sale or unused."
            ),
            location="Unknown", business_model="Unknown"
        )

    return {
        "website_url": website_url,
        "raw_website_content": combined,
        "overview": overview,
        "data_availability": data_availability,
        "completed_agents": state.get("completed_agents", []) + ["scraper"]
    }