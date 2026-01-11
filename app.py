import streamlit as st
import pandas as pd
import logging
from pathlib import Path

from models.registry import ModelRegistry
from ui.discovery_page import render_discovery_page, render_constraints_summary
from ui.filtering_page import render_filtering_page, render_filtering_summary
from ui.scoring_page import render_scoring_page, render_scoring_summary
from ui.recommendation_page import render_recommendation_page

from agents.llm_adapter import LLMConfig, LLMProvider, LLMAdapterFactory

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="LLM Decision Agent",
    page_icon="assets/favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# REQUIRED: Session State Initialization
# -------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

if "constraints" not in st.session_state:
    st.session_state.constraints = None

if "viable_models" not in st.session_state:
    st.session_state.viable_models = None

if "model_scores" not in st.session_state:
    st.session_state.model_scores = None

if "trade_off_analysis" not in st.session_state:
    st.session_state.trade_off_analysis = None

if "llm_enabled" not in st.session_state:
    st.session_state.llm_enabled = False

if "llm_config" not in st.session_state:
    st.session_state.llm_config = None


# -------------------------------------------------
# Load CSS (UI ONLY – NO LOGIC)
# -------------------------------------------------
css = Path("assets/styles.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# REQUIRED: LLM CONFIGURATION
# -------------------------------------------------
def render_llm_configuration():
    st.sidebar.header("🧠 AI Settings")

    if "llm_enabled" not in st.session_state:
        st.session_state.llm_enabled = False
        st.session_state.llm_config = None

    enable_llm = st.sidebar.checkbox(
        "Enable AI-powered analysis",
        value=st.session_state.llm_enabled
    )

    if not enable_llm:
        st.session_state.llm_enabled = False
        st.session_state.llm_config = None
        return None, False

    provider = st.sidebar.selectbox(
        "Provider",
        [p.value for p in LLMProvider]
    )

    model = st.sidebar.text_input("Model name", "mock-model")
    api_key = st.sidebar.text_input("API key", type="password")

    llm_config = LLMConfig(
        provider=LLMProvider(provider),
        model=model,
        api_key=api_key or None,
        temperature=0.3,
        max_tokens=800
    )

    st.session_state.llm_enabled = True
    st.session_state.llm_config = llm_config

    return llm_config, True


# -------------------------------------------------
# Sidebar branding + navigation (SAFE)
# -------------------------------------------------
with st.sidebar:
    st.image("assets/logo.svg", width=160)
    st.caption("Context-aware LLM selection")

    st.markdown("### Workflow")

    steps = [
        ("🔍 Discovery", 1),
        ("🔧 Filtering", 2),
        ("⚖️ Scoring", 3),
        ("💡 Recommendation", 4),
    ]

    for label, step in steps:
        if st.session_state.step == step:
            st.markdown(f"**➡️ {label}**")
        elif st.session_state.step > step:
            st.markdown(f"✅ {label}")
        else:
            st.markdown(f"⬜ {label}")

# -------------------------------------------------
# Load registry
# -------------------------------------------------
@st.cache_data
def load_registry():
    return ModelRegistry()

registry = load_registry()

# -------------------------------------------------
# Header
# -------------------------------------------------
header_left, header_right = st.columns([5, 2], vertical_alignment="center")

with header_left:
    st.markdown(
    """
    <h2 class="compact-title">Context-Aware LLM Model Comparison</h2>
    """,
    unsafe_allow_html=True,
)


with header_right:
    mode = "AI Mode" if st.session_state.get("llm_enabled") else "Rule-based Mode"
    st.markdown(
        f"<span class='mode-badge'>{mode}</span>",
        unsafe_allow_html=True,
    )

st.divider()

# -------------------------------------------------
# REQUIRED: LLM CONFIG RENDER
# -------------------------------------------------
llm_config, llm_enabled = render_llm_configuration()

# -------------------------------------------------
# ROUTING (ORIGINAL LOGIC PRESERVED)
# -------------------------------------------------
if st.session_state.step == 1:
    render_discovery_page(llm_config=llm_config, llm_enabled=llm_enabled)

elif st.session_state.step == 2:
    render_constraints_summary(st.session_state.constraints)
    render_filtering_page(registry, st.session_state.constraints)

elif st.session_state.step == 3:
    render_constraints_summary(st.session_state.constraints)
    render_filtering_summary()
    render_scoring_page(
        registry,
        st.session_state.constraints,
        st.session_state.viable_models
    )

elif st.session_state.step == 4:
    render_constraints_summary(st.session_state.constraints)
    render_filtering_summary()
    render_scoring_summary()
    render_recommendation_page(
        registry,
        st.session_state.constraints,
        st.session_state.model_scores,
        st.session_state.trade_off_analysis
    )
