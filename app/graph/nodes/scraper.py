# app/graph/nodes/scraper.py
import re
import json
import os
import requests
import concurrent.futures

from firecrawl import FirecrawlApp
from langchain_anthropic import ChatAnthropic
from app.graph.state import DueDiligenceState, CompanyOverview

from dotenv import load_dotenv
load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6")
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

FIRECRAWL_TIMEOUT = 20  # seconds — hard ceiling per Firecrawl call

# Plain `requests` with no headers gets blocked (403/timeout) by a lot of
# real, live sites that run bot protection — including major companies like
# Perplexity. That false-negative was pushing resolution past the real
# domain and into low-confidence TLD guessing. A normal browser UA fixes
# most of these without needing anything fancier.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Signals that a domain is parked/for-sale rather than an operating company.
# Covers the major registrar parking pages AND the major domain-marketplace /
# brokerage sites where a domain can be "for sale" while still rendering a
# normal-looking landing page (these were previously missing and could slip
# through as if they were a real operating company's homepage).
PARKED_DOMAIN_MARKERS = [
    "domain is for sale", "buy this domain", "this domain may be for sale",
    "dynadot", "godaddy", "namecheap marketplace", "sedo", "afternic",
    "domain parking", "make an offer", "domain broker", "this domain name is for sale",
    "backorder this domain", "domain auction",
    # Marketplaces / brokerages
    "dan.com", "hugedomains", "buydomains", "flippa", "squadhelp",
    "brandbucket", "epik marketplace", "atom.com", "domain.com marketplace",
    "porkbun marketplace", "namesilo marketplace", "escrow.com",
    "this website is for sale", "domain name for sale", "premium domain for sale",
    "inquire about this domain", "purchase this domain", "lease to own this domain",
]


def _get_attr(obj, key, default=None):
    """Read a field whether obj is a dict or an SDK response object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _with_timeout(fn, *args, timeout=FIRECRAWL_TIMEOUT, **kwargs):
    """Run a blocking call with a hard timeout so a stalled request can't hang the pipeline."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"[scraper] timed out after {timeout}s calling {getattr(fn, '__name__', fn)}")
            return None
        except Exception as e:
            print(f"[scraper] error calling {getattr(fn, '__name__', fn)}: {e}")
            return None


def _normalize_url(url: str) -> str:
    """Strip trailing slash so downstream path concatenation doesn't double up."""
    return url.rstrip("/")


def _url_is_reachable(url: str) -> bool:
    """Check a candidate URL actually resolves before trusting it."""
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True, headers=_BROWSER_HEADERS)
        if resp.status_code < 400:
            return True
    except Exception:
        pass
    try:
        resp = requests.get(url, timeout=8, allow_redirects=True, headers=_BROWSER_HEADERS)
        return resp.status_code < 400
    except Exception:
        return False


def _is_parked_domain(markdown: str) -> bool:
    """Detect domain-for-sale / parking / marketplace pages so they aren't
    mistaken for a real company site."""
    if not markdown:
        return False
    lowered = markdown.lower()
    return any(marker in lowered for marker in PARKED_DOMAIN_MARKERS)


def _domain_matches_name(url: str, startup_input: str) -> bool:
    """Score whether a URL's domain plausibly belongs to the company name."""
    try:
        domain = url.split("//")[-1].split("/")[0].lower()
        domain = domain.replace("www.", "")
        domain_root = domain.split(".")[0]
    except Exception:
        return False

    name_slug = re.sub(r'[^a-z0-9]', '', startup_input.lower())
    domain_slug = re.sub(r'[^a-z0-9]', '', domain_root)

    if not name_slug or not domain_slug:
        return False

    return name_slug == domain_slug or name_slug in domain_slug or domain_slug in name_slug


def _llm_guess_domains(startup_input: str) -> list[str]:
    """Ask Claude for its best-guess candidate domains for this company."""
    prompt = f"""What is the official website domain for the company/startup "{startup_input}"?

Return ONLY a JSON array of up to 3 candidate domains, ranked most likely first, as bare domains without protocol or path (e.g. "linear.app", not "https://linear.app/").

If you don't know this company or aren't confident, return an empty array [].

Example output: ["linear.app", "linear.com"]"""

    try:
        response = llm.invoke(prompt)
        clean = re.sub(r"```json|```", "", response.content).strip()
        domains = json.loads(clean)
        if isinstance(domains, list):
            return [d.strip().lower() for d in domains if isinstance(d, str) and d.strip()]
    except Exception as e:
        print(f"[scraper] LLM domain guess failed: {e}")
    return []


def _check_candidate(url: str, require_content: bool = True) -> tuple[bool, str]:
    """Verify a candidate is reachable AND not a parked/for-sale page.
    Returns (is_valid, markdown).

    require_content controls what happens when Firecrawl fails to scrape the
    page (e.g. ERR_EMPTY_RESPONSE, timeout). Previously a failed scrape was
    always treated as a "weak pass" — reachable-so-good-enough — which let a
    domain through as a verified operating company with literally zero
    content actually read from it. That content-free "pass" then skipped the
    parked-domain check entirely (nothing to match markers against) and let
    the LLM fabricate a plausible-looking overview from its own training
    knowledge of the name instead of what was actually on the page.

    For name-based resolution (LLM guess, search fallback, TLD guess) we now
    require an actual successful scrape — require_content=True — before
    accepting a candidate; a scrape failure rejects it and resolution moves
    to the next candidate. The one exception is a user-supplied explicit URL
    (require_content=False), where we don't want to silently substitute a
    different domain for what the user typed — we still return a weak pass
    but the caller (_resolve_url) already warns about it, and
    scrape_website() has its own downstream check for effectively-empty
    content as a second safety net.
    """
    if not _url_is_reachable(url):
        return False, ""

    # Retry once — Firecrawl can be transiently flaky even on real, live sites.
    result = _with_timeout(firecrawl.scrape_url, url, timeout=10)
    if result is None:
        result = _with_timeout(firecrawl.scrape_url, url, timeout=10)

    if result is None:
        if require_content:
            print(f"[scraper] rejected {url} — reachable but Firecrawl could not scrape it (no verifiable content)")
            return False, ""
        return True, ""

    markdown = _get_attr(result, "markdown", "") or ""

    if _is_parked_domain(markdown):
        print(f"[scraper] rejected {url} — looks like a parked/for-sale domain")
        return False, ""

    if require_content and not markdown.strip():
        print(f"[scraper] rejected {url} — reachable but returned no page content")
        return False, ""

    return True, markdown


def _resolve_url(startup_input: str) -> tuple[str, str]:
    """
    Resolve a startup name to its website URL: LLM guess -> verify (incl. parked-domain check)
    -> search fallback -> TLD guess.
    Returns (resolved_url, homepage_markdown_if_already_fetched).
    """
    if startup_input.startswith("http"):
        url = _normalize_url(startup_input)
        valid, markdown = _check_candidate(url, require_content=False)
        if not valid:
            print(f"[scraper] WARNING: provided URL {url} appears to be a parked/unreachable domain")
        return url, markdown

    # 1. Ask the LLM directly — it already knows most real company domains
    llm_candidates = _llm_guess_domains(startup_input)
    for domain in llm_candidates:
        candidate = f"https://{domain}"
        valid, markdown = _check_candidate(candidate)
        if valid:
            print(f"[scraper] resolved '{startup_input}' -> {candidate} via LLM knowledge")
            return candidate, markdown
    if llm_candidates:
        print(f"[scraper] LLM suggested {llm_candidates} but none passed verification (unreachable or parked)")

    # 2. Fall back to web search for companies outside the LLM's knowledge
    candidates = []
    search_queries = [f"{startup_input} official website", f"{startup_input} homepage"]
    for query in search_queries:
        try:
            search_result = _with_timeout(firecrawl.search, query, timeout=10)
            if search_result is None:
                continue
            data = _get_attr(search_result, "data", [])
            for item in data[:5]:
                url = _get_attr(item, "url", "")
                if url:
                    candidates.append(_normalize_url(url))
        except Exception as e:
            print(f"[scraper] search failed for '{query}': {e}")

    seen = set()
    unique_candidates = [u for u in candidates if not (u in seen or seen.add(u))]

    for url in unique_candidates:
        if _domain_matches_name(url, startup_input):
            valid, markdown = _check_candidate(url)
            if valid:
                print(f"[scraper] resolved '{startup_input}' -> {url} via search (name match)")
                return url, markdown
    for url in unique_candidates:
        valid, markdown = _check_candidate(url)
        if valid:
            print(f"[scraper] resolved '{startup_input}' -> {url} via search (best effort)")
            return url, markdown

    # 3. Last resort: blind TLD guessing, still parked-checked
    slug = re.sub(r'[^a-z0-9]', '', startup_input.lower())
    for tld in ["com", "io", "app", "co", "ai", "dev", "so", "xyz", "in", "org", "net"]:
        candidate = f"https://{slug}.{tld}"
        valid, markdown = _check_candidate(candidate)
        if valid:
            print(f"[scraper] resolved '{startup_input}' -> {candidate} via TLD guess (low confidence)")
            return candidate, markdown

    fallback = f"https://{slug}.com"
    print(f"[scraper] could not verify any URL for '{startup_input}', using unverified fallback {fallback}")
    return fallback, ""


def _discover_pages(website_url: str) -> list[str]:
    """Try to find real subpages via Firecrawl's map; fall back to common paths."""
    guessed_paths = ["/about", "/about-us", "/pricing", "/team", "/company", "/our-story", "/our-purpose"]
    fallback_pages = [website_url] + [f"{website_url}{p}" for p in guessed_paths]

    map_result = _with_timeout(firecrawl.map_url, website_url)
    if map_result is None:
        print("[scraper] map_url timed out or failed, using guessed paths")
        return fallback_pages

    links = _get_attr(map_result, "links", [])
    if not links or len(links) > 2000:
        print(f"[scraper] map_url returned {len(links)} links, too noisy — using guessed paths")
        return fallback_pages

    keywords = ["about", "team", "pricing", "company", "founder", "leadership", "story", "purpose", "career"]
    matched = [website_url]
    for link in links:
        url = link if isinstance(link, str) else _get_attr(link, "url", "")
        if not url:
            continue
        path_segments = url.lower().replace("-", "/").split("/")
        if any(kw in path_segments for kw in keywords):
            matched.append(url)

    return matched[:6] if len(matched) > 1 else fallback_pages


def _clean_markdown(markdown: str) -> str:
    """Strip image links and excessive markdown noise so real text isn't crowded out."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _scrape_page(url: str) -> str:
    """Scrape a single URL using the installed firecrawl-py v2 SDK signature, with a timeout."""
    result = _with_timeout(firecrawl.scrape_url, url)
    if result is None:
        return ""
    markdown = _get_attr(result, "markdown", "") or ""
    return _clean_markdown(markdown)


def scrape_website(state: DueDiligenceState) -> DueDiligenceState:
    """Scrape homepage and relevant subpages."""
    startup_input = state["startup_input"]

    website_url, homepage_markdown = _resolve_url(startup_input)
    is_parked = _is_parked_domain(homepage_markdown)

    if is_parked:
        print(f"[scraper] WARNING: {website_url} is a parked/for-sale domain — no real company to analyse")
        combined = f"## {website_url}\n[This domain appears to be parked or listed for sale — no operating company content found.]"
        overview = CompanyOverview(
            name=startup_input,
            website=website_url,
            description=f"'{website_url}' appears to be a parked or for-sale domain with no operating business, product, or team.",
            founded_year=None,
            location="Unknown",
            business_model="Unknown — no operating company found at this domain"
        )
        return {
            "website_url": website_url,
            "raw_website_content": combined,
            "overview": overview,
            "is_operating_company": False,
            "completed_agents": state.get("completed_agents", []) + ["scraper"]
        }

    pages_to_scrape = _discover_pages(website_url)

    raw_content: list[str] = []
    # Reuse homepage markdown if we already fetched it during resolution
    if homepage_markdown:
        raw_content.append(f"## {website_url}\n{_clean_markdown(homepage_markdown)[:3000]}")
        pages_to_scrape = [p for p in pages_to_scrape if p != website_url]

    for url in pages_to_scrape:
        markdown = _scrape_page(url)
        if markdown:
            raw_content.append(f"## {url}\n{markdown[:3000]}")
        else:
            print(f"[scraper] no usable content for {url}")

    combined = "\n\n".join(raw_content)

    if not combined.strip():
        # No page anywhere under this domain actually yielded content (every
        # scrape failed, or the site returned nothing usable). Without real
        # page content, asking the LLM to "extract company info" just invites
        # it to fall back on whatever it already knows about a company with
        # this name from training — which may have nothing to do with what's
        # actually (or isn't) at this specific domain. Treat this the same
        # way as a confirmed-parked domain rather than risk a fabricated,
        # confident-looking overview built from zero verified content.
        print(f"[scraper] WARNING: no content could be scraped from {website_url} — treating as unverified, not an operating company")
        combined = f"## {website_url}\n[No page content could be retrieved from this domain — nothing here was verified.]"
        overview = CompanyOverview(
            name=startup_input,
            website=website_url,
            description=(
                f"No content could be scraped from '{website_url}'. This may be a domain that "
                "blocks automated access, an empty/misconfigured site, or a genuinely non-operating "
                "domain — it was not possible to confirm which from this automated pass."
            ),
            founded_year=None,
            location="Unknown",
            business_model="Unknown — no verifiable content found at this domain"
        )
        return {
            "website_url": website_url,
            "raw_website_content": combined,
            "overview": overview,
            "is_operating_company": False,
            "completed_agents": state.get("completed_agents", []) + ["scraper"]
        }

    prompt = f"""Extract company information from this website content.
Return ONLY valid JSON matching this schema:
{{
    "name": "company name",
    "website": "website url",
    "description": "2-3 sentence description",
    "founded_year": null or year as int,
    "location": "city, country",
    "business_model": "B2B SaaS / B2C / Marketplace / etc"
}}

If a field cannot be determined, use "Unknown" as the value — except founded_year, which should be null if unknown.
Never return null for name, description, location, or business_model.

Website content:
{combined[:4000]}"""

    response = llm.invoke(prompt)
    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        data = json.loads(clean)

        data["website"] = website_url
        data["name"] = data.get("name") or startup_input
        data["description"] = data.get("description") or "Could not extract description"
        data["location"] = data.get("location") or "Unknown"
        data["business_model"] = data.get("business_model") or "Unknown"

        overview = CompanyOverview(**data)
    except Exception as e:
        print(f"[scraper] JSON parse failed: {e}")
        overview = CompanyOverview(
            name=startup_input,
            website=website_url,
            description="Could not extract",
            founded_year=None,
            location="Unknown",
            business_model="Unknown"
        )

    return {
        "website_url": website_url,
        "raw_website_content": combined,
        "overview": overview,
        "is_operating_company": True,
        "completed_agents": state.get("completed_agents", []) + ["scraper"]
    }