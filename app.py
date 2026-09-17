"""
app.py

AI News Research & Summarization Assistant
--------------------------------------------
Streamlit UI that ties together Tavily (web search) and Gemini
(analysis / summarization). This file only handles the UI and
application flow — the real work happens inside services/ and utils/.
"""

from datetime import datetime
from pathlib import Path

import streamlit as st

from services.tavily_service import search_news
from services.gemini_service import generate_summary
from utils.text_utils import (
    prepare_sources_for_gemini,
    format_sources_block,
    split_followups,
)
from config import GEMINI_API_KEY, TAVILY_API_KEY, RESEARCH_MODES, DEFAULT_MODE


# ===========================================================
# PAGE CONFIG + CUSTOM CSS
# ===========================================================
st.set_page_config(
    page_title="NewsLens AI — Research & Summarization Assistant",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css_path = Path(__file__).parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

QUICK_TOPICS = {
    "🤖 AI": "Latest AI developments",
    "💻 Technology": "Latest technology developments",
    "🛡️ Cybersecurity": "Latest cybersecurity developments",
    "🚀 Space": "Recent space technology developments",
    "📈 Business": "Latest business and economy news",
}

MAX_HISTORY_ITEMS = 8


# ===========================================================
# SESSION STATE
# ===========================================================
defaults = {
    "query_text": "",
    "mode_select": DEFAULT_MODE,
    "history": [],
    "active": None,
    "auto_run": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def use_topic(topic_query: str):
    st.session_state.query_text = topic_query


def trigger_followup(question: str):
    st.session_state.query_text = question
    st.session_state.auto_run = True


def view_history_item(item: dict):
    st.session_state.active = item


def clear_history():
    st.session_state.history = []
    st.session_state.active = None


# ===========================================================
# CORE RESEARCH PIPELINE (used by both the Research button
# and the follow-up question buttons)
# ===========================================================
def run_research(query: str, mode: str):
    """Runs the full Tavily -> Gemini pipeline with live status feedback."""
    try:
        with st.status("Running research...", expanded=True) as status:
            st.write("🔎 Searching web sources...")
            raw_results = search_news(query)

            if not raw_results:
                status.update(label="No results found", state="error")
                st.warning(
                    "No useful web results were found for this query. "
                    "Try rephrasing it or using a broader topic."
                )
                return None

            st.write(f"✅ Collected {len(raw_results)} relevant sources")

            prepared = prepare_sources_for_gemini(raw_results)
            sources_block = format_sources_block(prepared)

            st.write("🤖 Analyzing sources with Gemini...")
            raw_analysis = generate_summary(query, sources_block, mode=mode)
            main_text, followups = split_followups(raw_analysis)

            st.write("📝 Preparing report...")
            status.update(label="Research complete", state="complete")

    except ValueError as ve:
        st.error(str(ve))
        return None
    except RuntimeError as re_err:
        st.error(str(re_err))
        return None
    except Exception:
        st.error("Something unexpected went wrong. Please try again.")
        return None

    return {
        "query": query,
        "mode": mode,
        "sources": prepared,
        "analysis": main_text,
        "followups": followups,
        "timestamp": datetime.now().strftime("%I:%M %p"),
    }


def build_report_markdown(result: dict) -> str:
    """Builds the downloadable .md report: query + analysis + sources."""
    mode_label = RESEARCH_MODES.get(result["mode"], {}).get("label", result["mode"])
    lines = [
        f"# Research Report: {result['query']}",
        f"_Mode: {mode_label} · Generated at {result['timestamp']}_",
        "",
        result["analysis"],
        "",
        "## Sources",
    ]
    for source in result["sources"]:
        lines.append(f"{source['number']}. {source['title']} — {source['url']}")
    return "\n".join(lines)


# ===========================================================
# SIDEBAR — RESEARCH HISTORY (session only, no database)
# ===========================================================
with st.sidebar:
    st.markdown("### 🕘 Recent Research")
    if not st.session_state.history:
        st.caption("Your past searches in this session will appear here.")
    else:
        for idx, item in enumerate(st.session_state.history):
            with st.container(border=True):
                st.caption(f"{item['timestamp']} · {RESEARCH_MODES[item['mode']]['icon']} {RESEARCH_MODES[item['mode']]['label']}")
                st.markdown(f"**{item['query'][:55]}**{'…' if len(item['query']) > 55 else ''}")
                st.button(
                    "View",
                    key=f"hist_view_{idx}",
                    on_click=view_history_item,
                    args=(item,),
                    use_container_width=True,
                )
        st.button(
            "🗑️ Clear history",
            key="clear_history_btn",
            use_container_width=True,
            on_click=clear_history,
        )

# ===========================================================
# HERO SECTION
# ===========================================================
st.markdown('<div class="hero-title">📰 NewsLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-tagline">Research the web. Understand the story.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-description">Ask a question about current events. '
    "NewsLens searches live web sources with Tavily, then Gemini reads, "
    "cross-checks, and organizes what it finds — with every claim traceable "
    "back to a real source.</div>",
    unsafe_allow_html=True,
)
st.write("")

# API key check (friendly warning, no crash)
if not GEMINI_API_KEY or not TAVILY_API_KEY:
    missing = []
    if not GEMINI_API_KEY:
        missing.append("`GEMINI_API_KEY`")
    if not TAVILY_API_KEY:
        missing.append("`TAVILY_API_KEY`")
    st.warning(
        f"⚠️ Missing configuration: {', '.join(missing)}. "
        "Please add the missing key(s) to your `.env` file before searching."
    )

st.write("")

# ===========================================================
# SEARCH CARD
# ===========================================================
with st.container(border=True):
    st.markdown(
        '<span class="live-badge"><span class="live-dot"></span> Live Web Research</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown("**Quick topics**")
    topic_cols = st.columns(len(QUICK_TOPICS))
    for col, (label, topic_query) in zip(topic_cols, QUICK_TOPICS.items()):
        with col:
            st.button(
                label,
                key=f"topic_{label}",
                on_click=use_topic,
                args=(topic_query,),
                use_container_width=True,
            )

    st.write("")
    st.text_area(
        "Research question",
        placeholder="Example: Latest developments in artificial intelligence",
        key="query_text",
        height=100,
        label_visibility="collapsed",
    )

    mode_keys = list(RESEARCH_MODES.keys())
    st.radio(
        "Research mode",
        options=mode_keys,
        format_func=lambda k: f"{RESEARCH_MODES[k]['icon']} {RESEARCH_MODES[k]['label']}",
        key="mode_select",
        horizontal=True,
    )

    research_clicked = st.button("🔍 Research", type="primary", use_container_width=True)

# ===========================================================
# HANDLE SEARCH (manual click or an auto-triggered follow-up)
# ===========================================================
should_run = research_clicked or st.session_state.auto_run
if should_run:
    st.session_state.auto_run = False
    query = (st.session_state.query_text or "").strip()

    if not query:
        st.warning("Please enter a news topic.")
    elif not TAVILY_API_KEY:
        st.error("Tavily API key is missing. Please add it to your .env file.")
    elif not GEMINI_API_KEY:
        st.error("Gemini API key is missing. Please add it to your .env file.")
    else:
        result = run_research(query, st.session_state.mode_select)
        if result:
            st.session_state.history.insert(0, result)
            st.session_state.history = st.session_state.history[:MAX_HISTORY_ITEMS]
            st.session_state.active = result

# ===========================================================
# RESULTS
# ===========================================================
active = st.session_state.active

if active:
    st.divider()

    mode_info = RESEARCH_MODES.get(active["mode"], {})

    # ---------------- STAT CARDS ----------------
    stat_cols = st.columns(3)
    stats = [
        ("Sources Used", str(len(active["sources"]))),
        ("Research Mode", f"{mode_info.get('icon', '')} {mode_info.get('label', active['mode'])}"),
        ("Status", "✅ Complete"),
    ]
    for col, (label, value) in zip(stat_cols, stats):
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="stat-value">{value}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-label">{label}</div>', unsafe_allow_html=True)

    st.write("")

    # ---------------- MAIN ANALYSIS ----------------
    with st.container(border=True):
        st.markdown(f"#### Results for: _{active['query']}_")
        st.markdown(active["analysis"])

    st.write("")

    # ---------------- EXPORT ----------------
    export_cols = st.columns(2)
    with export_cols[0]:
        with st.expander("📋 Copy-friendly view"):
            st.code(active["analysis"], language=None)
    with export_cols[1]:
        st.download_button(
            "⬇️ Download Report (.md)",
            data=build_report_markdown(active),
            file_name=f"newslens_report_{active['timestamp'].replace(':', '').replace(' ', '')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ---------------- FOLLOW-UP QUESTIONS ----------------
    if active["followups"]:
        st.write("")
        st.markdown("**🔗 Continue researching**")
        fu_cols = st.columns(len(active["followups"]))
        for col, question in zip(fu_cols, active["followups"]):
            with col:
                st.button(
                    question,
                    key=f"followup_{hash(question)}",
                    on_click=trigger_followup,
                    args=(question,),
                    use_container_width=True,
                )

    # ---------------- SOURCES ----------------
    st.divider()
    st.subheader("Sources")

    for source in active["sources"]:
        domain = source["url"].split("/")[2] if "//" in source["url"] else source["url"]
        with st.container(border=True):
            card_cols = st.columns([5, 1])
            with card_cols[0]:
                st.markdown(f"**{source['number']}. {source['title']}**")
                st.markdown(f'<span class="source-domain">{domain}</span>', unsafe_allow_html=True)
                snippet = source["content"]
                st.caption(snippet[:220] + ("..." if len(snippet) > 220 else ""))
            with card_cols[1]:
                st.link_button("Open ↗", source["url"], use_container_width=True)

    # ---------------- TRANSPARENCY ----------------
    st.divider()
    with st.container(border=True):
        st.markdown("**🛡️ Research Transparency**")
        st.caption(
            "- All information above comes from the web sources retrieved for this query.\n"
            "- Gemini analyzes and organizes only that retrieved information — it does not "
            "browse the web on its own.\n"
            "- Sources are listed above so you can verify any claim yourself.\n"
            "- Where sources disagreed, the analysis calls that out rather than picking a side."
        )

else:
    st.caption("Enter a topic above, choose a research mode, and click **Research** to get started.")