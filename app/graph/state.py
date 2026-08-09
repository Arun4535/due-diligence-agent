from typing import TypedDict, Optional
from pydantic import BaseModel


class CompanyOverview(BaseModel):
    name: str
    website: str
    description: str
    founded_year: Optional[int] = None
    location: Optional[str] = "Unknown"
    business_model: Optional[str] = "Unknown"


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

    overview: Optional[CompanyOverview]
    raw_website_content: str
    team_members: list[TeamMember]
    funding_history: list[FundingRound]
    competitors: list[Competitor]
    market_analysis: str
    risk_factors: list[str]
    news_mentions: list[str]

    is_operating_company: bool

    verdict: str
    verdict_reasoning: str
    verdict_strengths: list[str]
    verdict_concerns: list[str]
    confidence_score: float
    memo_pdf_path: Optional[str]

    errors: list[str]
    completed_agents: list[str]
