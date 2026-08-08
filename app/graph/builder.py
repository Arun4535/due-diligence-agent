from langgraph.graph import StateGraph, END
from app.graph.state import DueDiligenceState
from app.graph.nodes.scraper import scrape_website
from app.graph.nodes.team import extract_team
from app.graph.nodes.market import analyse_market
from app.graph.nodes.competitors import find_competitors
from app.graph.nodes.funding import extract_funding
from app.graph.nodes.risks import assess_risks
from app.graph.nodes.verdict import generate_verdict
from app.report.generator import generate_pdf

def generate_report_node(state: DueDiligenceState) -> DueDiligenceState:
    pdf_path = generate_pdf(state)
    return {"memo_pdf_path": pdf_path}

def build_graph():
    workflow = StateGraph(DueDiligenceState)

    workflow.add_node("scrape", scrape_website)
    workflow.add_node("team", extract_team)
    workflow.add_node("market", analyse_market)
    workflow.add_node("competitors", find_competitors)
    workflow.add_node("funding", extract_funding)
    workflow.add_node("risks", assess_risks)
    workflow.add_node("verdict", generate_verdict)
    workflow.add_node("report", generate_report_node)

    workflow.set_entry_point("scrape")
    workflow.add_edge("scrape", "team")
    workflow.add_edge("team", "funding")
    workflow.add_edge("funding", "market")
    workflow.add_edge("market", "competitors")
    workflow.add_edge("competitors", "risks")
    workflow.add_edge("risks", "verdict")
    workflow.add_edge("verdict", "report")
    workflow.add_edge("report", END)

    return workflow.compile()