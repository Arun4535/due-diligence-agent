# app/graph/nodes/team.py
import json
import re
from tavily import TavilyClient
from langchain_anthropic import ChatAnthropic
from app.graph.state import DueDiligenceState, TeamMember

llm = ChatAnthropic(model="claude-sonnet-4-6")

ROLE_KEYWORDS = ["CEO", "COO", "CPO", "CTO", "Co-founder", "Founder"]


def _search_roles(company_name: str, company_domain: str) -> str:
    """Search company name + domain + each exec role keyword, combine snippets."""
    tavily = TavilyClient()
    combined = []

    for role in ROLE_KEYWORDS:
        try:
            results = tavily.search(
                f'"{company_name}" ({company_domain}) {role} name LinkedIn',
                max_results=2
            )
            for r in results.get("results", []):
                combined.append(f"[{role} search] {r['content'][:300]}")
        except Exception:
            continue

    return "\n".join(combined)


def extract_team(state: DueDiligenceState) -> DueDiligenceState:
    """Extract team members from website content + per-role Tavily searches, scoped to this domain."""
    company_name = state["overview"].name
    company_domain = state["overview"].website

    # If the scraper already flagged this as a parked/no-content domain, don't bother
    # searching for a team — it would just pull in an unrelated same-named entity.
    if not state.get("is_operating_company", True):
        print("[team] skipping team search — not an operating company (parked/for-sale domain)")
        return {
            "team_members": [],
            "completed_agents": state.get("completed_agents", []) + ["team"]
        }

    role_search_content = _search_roles(company_name, company_domain)

    combined_content = (
        f"WEBSITE CONTENT:\n{state['raw_website_content'][:2500]}\n\n"
        f"ROLE SEARCH RESULTS:\n{role_search_content[:2500]}"
    )

    prompt = f"""Extract team/founder/executive information for the company at {company_domain} (named "{company_name}") from the content below.
The content includes both website text and web search results targeting specific roles (CEO, COO, CPO, CTO, Co-founder).

IMPORTANT: Only include people you can confirm are associated with the company AT THIS SPECIFIC DOMAIN ({company_domain}).
If a search result mentions a person associated with a similarly-named but different company or domain, DO NOT include them.
Deduplicate — if the same person appears in both sources, merge into one entry.
Only include people you can confidently name — do not invent names.

Return ONLY a JSON array:
[
  {{
    "name": "Full Name",
    "role": "CEO / CTO / Co-founder / etc",
    "background": "previous experience in 1-2 sentences, or empty string if unknown",
    "linkedin_url": null
  }}
]

Content:
{combined_content}

If no team info found, return empty array [].
Return ONLY the JSON array, no other text."""

    response = llm.invoke(prompt)
    try:
        clean = re.sub(r"```json|```", "", response.content).strip()
        members_data = json.loads(clean)
        members = [TeamMember(**m) for m in members_data]
    except Exception:
        members = []

    tavily = TavilyClient()
    enriched = []
    for member in members[:5]:
        try:
            results = tavily.search(
                f'{member.name} "{company_name}" ({company_domain}) {member.role} background LinkedIn',
                max_results=2
            )
            if results.get("results"):
                snippet = results["results"][0]["content"][:200]
                if snippet not in member.background:
                    member.background = (member.background + f" — {snippet}").strip(" —")
        except Exception:
            pass
        enriched.append(member)

    return {
        "team_members": enriched,
        "completed_agents": state.get("completed_agents", []) + ["team"]
    }
