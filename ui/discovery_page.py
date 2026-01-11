import streamlit as st
import logging
from typing import Optional

from agents.discovery import create_discovery_agent
from agents.llm_adapter import LLMConfig
from models.schemas import TaskType, LatencyCategory
from utils.ui_helpers import show_spinner, toast_success, toast_error, toast_warning


def render_discovery_page(
    llm_config: Optional[LLMConfig] = None,
    llm_enabled: bool = False,
):
    """Render the discovery page (UI-enhanced, logic unchanged)"""

    st.markdown(
    """
    <h3 class="compact-step">🔍 Step 1: Use Case Discovery</h3>
    <p class="compact-caption">
        Describe your use case and priorities to identify the right LLM
    </p>
    """,
    unsafe_allow_html=True,
)


    # =========================================================
    # AI STATUS (CLEAR BUT NOT LOUD)
    # =========================================================
    if llm_enabled and llm_config:
        st.success(f"🤖 AI-enhanced analysis active ({llm_config.provider.value})")
    else:
        st.info("🔧 Rule-based analysis")

    # =========================================================
    # INITIALIZE DISCOVERY AGENT (UNCHANGED)
    # =========================================================
    if (
        "discovery_agent" not in st.session_state
        or st.session_state.get("llm_config_changed", False)
    ):
        try:
            st.session_state.discovery_agent = create_discovery_agent(
                enable_llm=llm_enabled,
                llm_config=llm_config,
            )
            st.session_state.llm_config_changed = False
            st.session_state.discovery_method = (
                "AI-Enhanced" if llm_enabled else "Rule-Based"
            )
        except Exception as e:
            logging.error(f"Discovery agent init failed: {e}")
            toast_error("Failed to initialize AI agent. Falling back.")
            st.session_state.discovery_agent = create_discovery_agent(enable_llm=False)
            st.session_state.discovery_method = "Rule-Based (Fallback)"

    discovery_agent = st.session_state.discovery_agent

    # =========================================================
    # SECTION 1 — PRIMARY INPUT (MOST IMPORTANT)
    # =========================================================
    st.markdown("### 📝 Describe your use case")

    user_input = st.text_area(
        label="What do you want to use an LLM for?",
        placeholder=(
            "Example:\n"
            "Analyze customer feedback to identify trends and generate summaries. "
            "Accuracy is important. ~1,000 reviews/day."
        ),
        height=100,
    )

    # =========================================================
    # SECTION 2 — STRUCTURED REFINEMENTS
    # =========================================================
    st.markdown("### 🎯 Refine your requirements")

    c1, c2 = st.columns(2)

    with c1:
        task_type = st.selectbox(
            "Primary Task",
            ["Auto-detect"] + [t.value for t in TaskType],
        )

        latency = st.selectbox(
            "Latency Requirement",
            ["Auto-detect"] + [l.value for l in LatencyCategory],
        )

    with c2:
        budget_input = st.text_input(
            "Budget sensitivity",
            placeholder="e.g. budget-conscious / $100 per month",
        )

        context_input = st.text_input(
            "Context window needs",
            placeholder="e.g. 32k tokens / long documents",
        )

    # =========================================================
    # SECTION 3 — ADVANCED OPTIONS (OPTIONAL)
    # =========================================================
    with st.expander("⚙️ Advanced: Adjust priority weights"):
        st.caption("If not set, weights are inferred automatically")

        c3, c4 = st.columns(2)

        with c3:
            reasoning_w = st.slider("Reasoning / Accuracy", 0.0, 1.0, 0.25, 0.05)
            latency_w = st.slider("Latency / Speed", 0.0, 1.0, 0.25, 0.05)

        with c4:
            cost_w = st.slider("Cost Efficiency", 0.0, 1.0, 0.25, 0.05)
            reliability_w = st.slider("Reliability", 0.0, 1.0, 0.25, 0.05)

        total = reasoning_w + latency_w + cost_w + reliability_w
        custom_weights = (
            {
                "reasoning": reasoning_w / total,
                "latency": latency_w / total,
                "cost": cost_w / total,
                "reliability": reliability_w / total,
            }
            if total > 0
            else None
        )

    # =========================================================
    # PRIMARY ACTION
    # =========================================================
    st.markdown("---")

    if st.button("🔍 Analyze requirements", type="primary"):
        if not user_input.strip():
            st.error("Please describe your use case to continue.")
            return

        with show_spinner(
            f"Analyzing using {st.session_state.discovery_method} approach..."
        ):
            try:
                constraints = discovery_agent.extract_constraints(
                    user_input=user_input,
                    use_case_type=None if task_type == "Auto-detect" else task_type,
                    budget_input=budget_input or None,
                    latency_input=None if latency == "Auto-detect" else latency,
                    context_input=context_input or None,
                )

                if custom_weights:
                    constraints.priority_weights = custom_weights

                st.session_state.constraints = constraints
                st.session_state.step = 2

                toast_success("Requirements analyzed successfully")

            except Exception as e:
                logging.error(f"Constraint extraction failed: {e}")
                toast_error("Analysis failed")

                if llm_enabled:
                    toast_warning("Retrying with rule-based fallback")
                    try:
                        fallback = create_discovery_agent(enable_llm=False)
                        constraints = fallback.extract_constraints(
                            user_input=user_input,
                            use_case_type=None,
                            budget_input=budget_input or None,
                            latency_input=None,
                            context_input=context_input or None,
                        )
                        if custom_weights:
                            constraints.priority_weights = custom_weights
                        st.session_state.constraints = constraints
                        st.session_state.step = 2
                        toast_success("Fallback analysis successful")
                    except Exception as fe:
                        toast_error(f"Fallback failed: {fe}")
                        return
                else:
                    return

        st.rerun()

    # =========================================================
    # SECTION 4 — EXTRACTED CONSTRAINTS (RESULT)
    # =========================================================
    if st.session_state.get("constraints"):
        c = st.session_state.constraints

        st.markdown("### 📊 Extracted constraints")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="card">
                    <b>Task</b><br/>
                    {c.task_type.value if c.task_type else "N/A"}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="card">
                    <b>Latency</b><br/>
                    {c.latency_tolerance.value if c.latency_tolerance else "N/A"}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            top = max(c.priority_weights.items(), key=lambda x: x[1])
            st.markdown(
                f"""
                <div class="card">
                    <b>Top Priority</b><br/>
                    {top[0].title()} ({top[1]:.0%})
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🔧 View full constraint details"):
            st.json(c.model_dump())

        questions = discovery_agent.ask_clarifying_questions(c)
        if questions:
            st.warning("Additional clarification may improve results:")
            for q in questions:
                st.write(f"• {q}")

        if discovery_agent.has_sufficient_information(c):
            if st.button("➡️ Proceed to Model Filtering", type="primary"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.error("Please provide more information to continue.")


def render_constraints_summary(constraints):
    """Compact constraint summary for later steps"""

    if not constraints:
        return

    with st.expander("📋 Current Constraints"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.write(f"**Task:** {constraints.task_type.value if constraints.task_type else 'N/A'}")
            st.write(f"**Latency:** {constraints.latency_tolerance.value if constraints.latency_tolerance else 'N/A'}")

        with c2:
            st.write(
                f"**Budget:** ${constraints.max_cost_per_token:.6f}"
                if constraints.max_cost_per_token
                else "**Budget:** N/A"
            )
            st.write(
                f"**Context:** {constraints.min_context_window:,}"
                if constraints.min_context_window
                else "**Context:** N/A"
            )

        with c3:
            top = max(constraints.priority_weights.items(), key=lambda x: x[1])
            st.write(f"**Top Priority:** {top[0]} ({top[1]:.0%})")
