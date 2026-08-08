"""PDF report generator for Due Diligence Agent — uses ReportLab (pure Python)."""
from __future__ import annotations

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

from app.graph.state import DueDiligenceState

PASS_COLOR  = colors.HexColor("#155724")
WATCH_COLOR = colors.HexColor("#856404")
AVOID_COLOR = colors.HexColor("#721c24")
PASS_BG     = colors.HexColor("#d4edda")
WATCH_BG    = colors.HexColor("#fff3cd")
AVOID_BG    = colors.HexColor("#f8d7da")
DARK        = colors.HexColor("#1a1a1a")
GRAY        = colors.HexColor("#666666")
LIGHT_GRAY  = colors.HexColor("#f0f0f0")
ACCENT      = colors.HexColor("#0c447c")
NOTICE_BG   = colors.HexColor("#e2e3e5")
NOTICE_TEXT = colors.HexColor("#383d41")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=24,
            textColor=DARK,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=GRAY,
            spaceAfter=16,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=14,
            textColor=ACCENT,
            spaceBefore=20,
            spaceAfter=8,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            textColor=DARK,
            leading=15,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontSize=9,
            textColor=GRAY,
            leading=13,
        ),
        "verdict_text": ParagraphStyle(
            "verdict_text",
            parent=base["Normal"],
            fontSize=20,
            fontName="Helvetica-Bold",
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontSize=10,
            textColor=DARK,
            leading=14,
            leftIndent=12,
            spaceAfter=4,
        ),
        "notice": ParagraphStyle(
            "notice",
            parent=base["Normal"],
            fontSize=9.5,
            textColor=NOTICE_TEXT,
            leading=13,
        ),
    }


def _verdict_colors(verdict: str) -> tuple:
    mapping = {
        "PASS":  (PASS_COLOR,  PASS_BG),
        "WATCH": (WATCH_COLOR, WATCH_BG),
        "AVOID": (AVOID_COLOR, AVOID_BG),
    }
    return mapping.get(verdict, (GRAY, LIGHT_GRAY))


def generate_pdf(state: DueDiligenceState) -> str:
    """Generate a PDF investment memo and return its file path."""

    os.makedirs("reports", exist_ok=True)
    import re as _re
    safe_name = _re.sub(r"[^A-Za-z0-9_-]", "_", state["overview"].name.strip())
    safe_name = _re.sub(r"_{2,}", "_", safe_name).strip("_") or "company"
    date_str     = datetime.now().strftime("%Y%m%d")
    pdf_path     = f"reports/{safe_name}_{date_str}.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    s       = _styles()
    story   = []
    company = state["overview"]
    verdict = state.get("verdict", "WATCH")
    v_color, v_bg = _verdict_colors(verdict)
    is_operating_company = state.get("is_operating_company", True)

    story.append(Paragraph("Investment Due Diligence Memo", s["title"]))
    story.append(Paragraph(
        f"{company.name} &nbsp;·&nbsp; {datetime.now().strftime('%B %d, %Y')}",
        s["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=DARK, spaceAfter=16))

    if not is_operating_company:
        notice_table = Table(
            [[Paragraph(
                f"<b>Asset type: Domain / speculative digital asset — not an operating company.</b> "
                f"{company.website} resolves to a parked or for-sale domain, not a live product or "
                "business. Startup-specific sections below (Team, Funding, Market/TAM, Competitors) "
                "are marked not applicable rather than left to imply missing data.",
                s["notice"]
            )]],
            colWidths=[170 * mm],
        )
        notice_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), NOTICE_BG),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ]))
        story.append(notice_table)
        story.append(Spacer(1, 12))

    story.append(Paragraph("Verdict", s["h2"]))
    verdict_table = Table(
        [[Paragraph(verdict, ParagraphStyle(
            "vt",
            fontSize=20,
            fontName="Helvetica-Bold",
            textColor=v_color
        ))]],
        colWidths=[60 * mm],
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), v_bg),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 8))

    confidence = state.get("confidence_score", 0.0)
    story.append(Paragraph(
        f"Confidence: <b>{confidence:.0%}</b>",
        s["small"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(state.get("verdict_reasoning", ""), s["body"]))

    strengths = state.get("verdict_strengths", [])
    concerns  = state.get("verdict_concerns",  [])
    if strengths or concerns:
        story.append(Spacer(1, 8))
        sc_data = [[
            Paragraph("<b>Strengths</b>", s["body"]),
            Paragraph("<b>Concerns</b>",  s["body"])
        ]]
        str_lines = "\n".join([f"✓ {x}" for x in strengths]) or "—"
        con_lines = "\n".join([f"⚠ {x}" for x in concerns])  or "—"
        sc_data.append([
            Paragraph(str_lines, s["bullet"]),
            Paragraph(con_lines, s["bullet"]),
        ])
        sc_table = Table(sc_data, colWidths=[85 * mm, 85 * mm])
        sc_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), LIGHT_GRAY),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(sc_table)

    story.append(Paragraph(
        "Domain Overview" if not is_operating_company else "Company Overview",
        s["h2"]
    ))
    story.append(Paragraph(company.description, s["body"]))
    overview_data = [
        ["Founded",        str(company.founded_year or "Unknown")],
        ["Location",       company.location],
        ["Business Model", company.business_model],
        ["Website",        company.website],
    ]
    overview_table = Table(overview_data, colWidths=[45 * mm, 125 * mm])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), LIGHT_GRAY),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(overview_table)

    story.append(Paragraph("Team", s["h2"]))
    team = state.get("team_members", [])
    if not is_operating_company:
        story.append(Paragraph("Not applicable — no operating company or team behind this domain.", s["small"]))
    elif team:
        team_data = [["Name", "Role", "Background"]]
        for m in team:
            team_data.append([m.name, m.role, m.background[:140]])
        team_table = Table(team_data, colWidths=[40 * mm, 35 * mm, 95 * mm])
        team_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ]))
        story.append(team_table)
    else:
        story.append(Paragraph("No team information found publicly.", s["small"]))

    story.append(Paragraph("Funding History", s["h2"]))
    funding = state.get("funding_history", [])
    if not is_operating_company:
        story.append(Paragraph("Not applicable — no operating company to have raised funding.", s["small"]))
    elif funding:
        fund_data = [["Round", "Amount", "Date", "Investors"]]
        for r in funding:
            fund_data.append([
                r.round,
                r.amount or "Undisclosed",
                r.date   or "—",
                ", ".join(r.investors) if r.investors else "Undisclosed",
            ])
        fund_table = Table(fund_data, colWidths=[30 * mm, 30 * mm, 30 * mm, 80 * mm])
        fund_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ]))
        story.append(fund_table)
    else:
        story.append(Paragraph("No public funding information found.", s["small"]))

    story.append(Paragraph(
        "Domain Valuation Notes" if not is_operating_company else "Market Analysis",
        s["h2"]
    ))
    story.append(Paragraph(
        state.get("market_analysis", "Market analysis not available."),
        s["body"]
    ))

    story.append(Paragraph("Competitor Landscape", s["h2"]))
    competitors = state.get("competitors", [])
    if not is_operating_company:
        story.append(Paragraph("Not applicable — no product or business to have competitors.", s["small"]))
    elif competitors:
        comp_data = [["Company", "Website", "How They Differ"]]
        for c in competitors:
            comp_data.append([c.name, c.website, c.differentiator])
        comp_table = Table(comp_data, colWidths=[40 * mm, 50 * mm, 80 * mm])
        comp_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ]))
        story.append(comp_table)
    else:
        story.append(Paragraph("No competitors identified.", s["small"]))

    story.append(Paragraph("Risk Factors", s["h2"]))
    risks = state.get("risk_factors", [])
    if risks:
        for risk in risks:
            story.append(Paragraph(f"⚠ {risk}", s["bullet"]))
    else:
        story.append(Paragraph("No risk factors identified.", s["small"]))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Due Diligence Agent · {datetime.now().strftime('%B %d, %Y')}",
        s["small"]
    ))

    doc.build(story)
    return pdf_path
