import streamlit as st
import requests
import os
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")

SUGGESTED_STARTUPS = ["Linear", "Notion", "Vercel", "Stripe", "Figma"]

VERDICT_INFO = {
    "PASS": ("green", "Data supports moving forward — strong signals across team, market, and funding."),
    "WATCH": ("orange", "Mixed or incomplete signals — worth monitoring, not yet a clear yes or no."),
    "AVOID": ("red", "Significant red flags or insufficient verifiable data to support investment."),
}


def _escape_markdown_dollars(text: str) -> str:
    """Prevent Streamlit from treating $ as LaTeX math delimiters."""
    return text.replace("$", "\\$")


def render_result(data: dict):
    """Render a full result block — used for both a fresh analysis and a history replay."""
    verdict = data["verdict"]
    color, verdict_meaning = VERDICT_INFO.get(verdict, ("gray", "Unrecognized verdict value."))
    is_operating_company = data.get("is_operating_company", True)

    if not is_operating_company:
        st.warning(
            "🌐 **Not an operating company** — this input resolved to a parked or for-sale "
            "domain, not a live product or business. Team/Funding/Market/Competitor sections "
            "are not applicable and have been skipped; see the memo for domain-asset-specific "
            "risk notes instead."
        )

    st.markdown(f"## :{color}[{verdict}]")
    st.caption(verdict_meaning)
    st.write(_escape_markdown_dollars(data["verdict_reasoning"]))

    confidence = data["confidence_score"]
    if confidence < 0.4:
        st.warning(
            f"Confidence is low ({confidence:.0%}) — this likely reflects sparse available "
            "data rather than a confirmed poor company. Consider manual research before "
            "treating this verdict as final."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{confidence:.0%}")
    col1.caption("Data completeness, not company quality")
    col2.metric("Team members", data["team_count"])
    col3.metric("Funding rounds", data["funding_rounds"])
    col4.metric("Competitors", data["competitor_count"])

    with st.expander("📋 View full findings"):
        market_analysis = data.get("market_analysis") or ""
        risk_factors = data.get("risk_factors") or []

        if market_analysis:
            st.subheader("Domain Valuation Notes" if not is_operating_company else "Market Analysis")
            st.write(_escape_markdown_dollars(market_analysis))
        else:
            st.caption("No market analysis available for this run.")

        if risk_factors:
            st.subheader("Risk Factors")
            for r in risk_factors:
                st.markdown(f"- ⚠️ {_escape_markdown_dollars(r)}")
        else:
            st.caption("No risk factors available for this run.")

        if is_operating_company and data["team_count"] == 0:
            st.info("No team members were identified — verify this manually before trusting the verdict.")
        if is_operating_company and data["funding_rounds"] == 0:
            st.info("No funding history found — could mean bootstrapped, private, or simply not indexed.")

    pdf_filename = data["pdf_path"].split("/")[-1]
    try:
        pdf_response = requests.get(f"{API_URL}/report/{pdf_filename}", timeout=30)
        if pdf_response.status_code == 200:
            st.download_button(
                label="📄 Download Full Memo (PDF)",
                data=pdf_response.content,
                file_name=pdf_filename,
                mime="application/pdf",
                key=f"download_{pdf_filename}_{data.get('_history_key', 'live')}"
            )
    except requests.exceptions.RequestException:
        st.caption("PDF no longer available for download.")

    st.caption("Report generated for the current analysis.")


st.set_page_config(page_title="Due Diligence Agent", page_icon="🔍", layout="centered")

if "startup_input_value" not in st.session_state:
    st.session_state.startup_input_value = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"input": str, "timestamp": str, "data": dict}

st.title("🔍 Due Diligence Agent")
st.caption(
    "Enter a startup name or website. The agent scrapes the company site, searches for team, "
    "funding, market, and competitor data, then generates a verdict and a downloadable investment memo."
)

st.info(
    "**AI-generated analysis — not investment advice.** This tool uses AI to search public "
    "web data and may misidentify companies, miss key facts, or produce incorrect verdicts "
    "(as seen with ambiguous or similarly-named inputs). Always verify findings independently "
    "before making any investment decision.",
    icon="⚠️"
)
st.caption(
    "Enter a startup name or website. The agent scrapes the company site, searches for team, "
    "funding, market, and competitor data, then generates a verdict and a downloadable investment memo."
)

with st.expander("ℹ️ How to read the results"):
    st.markdown("""
**Verdict** — a summary judgment, not financial advice:
- 🟢 **PASS** — data supports moving forward
- 🟠 **WATCH** — mixed or incomplete signals, worth monitoring
- 🔴 **AVOID** — significant red flags or too little verifiable data to evaluate

**Confidence** — how much reliable data the agent actually found, *not* how good the company is.
A low confidence score (e.g. under 40%) usually means the agent couldn't verify much — sparse
website content, no funding data, no team info — rather than a definitively bad company. Treat
low-confidence AVOID verdicts as "needs manual research," not a hard rejection.

**Domain asset warning** — if the input resolves to a parked or for-sale domain rather than a
real company, the agent skips startup-specific analysis (team, funding, TAM, competitors)
instead of guessing.
""")

tab_analyse, tab_history = st.tabs(["🔍 Analyse", "🕘 History"])

with tab_analyse:
    st.write("Try:")
    cols = st.columns(len(SUGGESTED_STARTUPS))
    for col, name in zip(cols, SUGGESTED_STARTUPS):
        if col.button(name, use_container_width=True):
            st.session_state.startup_input_value = name

    with st.form("diligence_form"):
        startup_input = st.text_input(
            "Startup name or website URL",
            value=st.session_state.startup_input_value,
            placeholder="e.g. Linear or https://linear.app"
        )
        submitted = st.form_submit_button("Analyse", type="primary")

    if submitted and startup_input:
        with st.spinner("Running due diligence — this takes 60–90 seconds..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyse",
                    json={"startup_input": startup_input},
                    timeout=300
                )
                response.raise_for_status()
                data = response.json()

                # Save to history
                st.session_state.history.insert(0, {
                    "input": startup_input,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data": data
                })

                render_result(data)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running.")
            except requests.exceptions.Timeout:
                st.error("Analysis timed out. Try again.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")

with tab_history:
    if not st.session_state.history:
        st.caption("No analyses yet this session. Run one from the Analyse tab.")
    else:
        st.caption(f"{len(st.session_state.history)} analysis run(s) this session.")
        for i, entry in enumerate(st.session_state.history):
            verdict = entry["data"]["verdict"]
            color, _ = VERDICT_INFO.get(verdict, ("gray", ""))
            label = f":{color}[{verdict}] — {entry['data']['company_name']} ({entry['timestamp']})"
            with st.expander(label, expanded=False):
                entry_data = dict(entry["data"])
                entry_data["_history_key"] = i  # unique key suffix for download buttons
                render_result(entry_data)
