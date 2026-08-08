from typing import TypedDict, Optional
from pydantic import BaseModel


class CompanyOverview(BaseModel):
    name: str
    website: str
    description: str
    founded_year: Optional[int]
    location: str
    business_model: str


class TeamMember(BaseModel):
    name: str
    role: str
    background: str
    linkedin_url: Optional[str] = None


class FundingRound(BaseModel):
    round: str
    amount: Optional[str]
    date: Optional[str]
    investors: list[str]


class Competitor(BaseModel):
    name: str
    website: str
    differentiator: str


class DueDiligenceState(TypedDict):
    startup_input: str
    website_url: str

    # Agent outputs
    overview: Optional[CompanyOverview]
    raw_website_content: str
    team_members: list[TeamMember]
    funding_history: list[FundingRound]
    competitors: list[Competitor]
    market_analysis: str
    risk_factors: list[str]
    news_mentions: list[str]

    # True (default) when the resolved website belongs to a real, operating
    # company. Set to False by the scraper when the input resolves to a
    # parked/for-sale domain (or other non-operating listing). Every
    # downstream node checks this flag and skips startup-specific analysis
    # (team search, funding search, TAM/market sizing, competitor search)
    # that doesn't meaningfully apply to a domain-resale asset — instead of
    # silently producing a VC-memo-shaped report for something that isn't
    # a startup at all.
    is_operating_company: bool

    # Final output
    verdict: str                  # PASS / WATCH / AVOID
    verdict_reasoning: str
    verdict_strengths: list[str]
    verdict_concerns: list[str]
    confidence_score: float
    memo_pdf_path: Optional[str]

    # Control
    errors: list[str]
    completed_agents: list[str]
