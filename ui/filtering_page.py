import streamlit as st
import pandas as pd

from agents.constraint_filter import ConstraintFilterAgent
from models.registry import ModelRegistry
from models.schemas import UserConstraints


def render_filtering_page(registry: ModelRegistry, constraints: UserConstraints):
    """Render the constraint filtering page (UI-enhanced, logic unchanged)"""

    st.markdown("## 🔧 Step 2: Model Filtering")
    st.caption("Applying your constraints to eliminate incompatible models")

    # -----------------------------
    # Safety guard
    # -----------------------------
    if constraints is None:
        st.error("No constraints found. Please complete Discovery first.")
        return

    # -----------------------------
    # Initialize agent (unchanged)
    # -----------------------------
    if "filter_agent" not in st.session_state:
        st.session_state.filter_agent = ConstraintFilterAgent(registry)

    filter_agent = st.session_state.filter_agent

    # -----------------------------
    # Run filtering (unchanged)
    # -----------------------------
    with st.spinner("Filtering models based on your constraints..."):
        viable_models, elimination_reasons = filter_agent.filter_models(constraints)
        filtering_summary = filter_agent.get_filtering_summary(
            viable_models, elimination_reasons
        )
        constraint_impacts = filter_agent.get_constraint_impact_analysis(constraints)

    # Persist state (unchanged)
    st.session_state.viable_models = viable_models
    st.session_state.elimination_reasons = elimination_reasons
    st.session_state.filtering_summary = filtering_summary

    # =========================================================
    # SECTION 1 — EXECUTIVE SUMMARY (CARDS)
    # =========================================================
    st.markdown("### 📌 Filtering Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <h4>Total Models</h4>
                <h2>{filtering_summary["total_models"]}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
                <h4>Eliminated</h4>
                <h2 style="color:#ef4444">{filtering_summary["eliminated_count"]}</h2>
                <small>{filtering_summary["elimination_rate"]:.0%} removed</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="card">
                <h4>Viable Models</h4>
                <h2 style="color:#22c55e">{filtering_summary["viable_count"]}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================
    # SECTION 2 — WHY MODELS WERE ELIMINATED (VISUAL)
    # =========================================================
    st.markdown("### 📊 Why models were eliminated")

    impact_rows = []
    for constraint, count in constraint_impacts.items():
        if count > 0:
            impact_rows.append(
                {
                    "Constraint": constraint.replace("_", " ").title(),
                    "Models Eliminated": count,
                }
            )

    if impact_rows:
        impact_df = pd.DataFrame(impact_rows)

        with st.expander("Constraint Impact Breakdown", expanded=True):
            st.bar_chart(
                impact_df.set_index("Constraint")["Models Eliminated"],
                use_container_width=True,
            )
            st.dataframe(impact_df, use_container_width=True)
    else:
        st.success("🎉 No models were eliminated by your constraints.")

    # =========================================================
    # SECTION 3 — VIABLE MODELS (SURVIVORS)
    # =========================================================
    st.markdown("### ✅ Models that passed filtering")

    if viable_models:
        for model in filtering_summary["viable_models"]:
            st.markdown(
                f"""
                <div class="card">
                    <b>{model["name"]}</b><br/>
                    Provider: {model["provider"]}<br/>
                    Context Window: {model["context_window"]:,} tokens<br/>
                    Latency: {model["latency"]}<br/>
                    Cost Tier: {model["cost_tier"]}<br/>
                    Reasoning Strength: {model["reasoning_strength"]}/10
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.error("❌ No models meet your requirements.")
        st.markdown(
            """
            **Suggestions**
            - Relax budget constraints
            - Increase latency tolerance
            - Reduce context window requirements
            - Review required capabilities
            """
        )

    # =========================================================
    # SECTION 4 — DEEP DIVE (OPTIONAL DETAILS)
    # =========================================================
    if viable_models:
        with st.expander("🔍 Detailed Viable Model Comparison"):
            viable_df = pd.DataFrame(
                [
                    {
                        "Model": m["name"],
                        "Provider": m["provider"],
                        "Cost Tier": m["cost_tier"],
                        "Latency": m["latency"],
                        "Context Window": f'{m["context_window"]:,}',
                        "Reasoning": f'{m["reasoning_strength"]}/10',
                    }
                    for m in filtering_summary["viable_models"]
                ]
            )
            st.dataframe(viable_df, use_container_width=True)

    if elimination_reasons:
        with st.expander("❌ Eliminated Models (Detailed Reasons)"):
            elimination_by_constraint = filtering_summary["elimination_by_constraint"]

            for constraint_type, reasons in elimination_by_constraint.items():
                st.markdown(f"**{constraint_type.replace('_', ' ').title()}**")
                for r in reasons:
                    st.write(f"- **{r.model_name}**: {r.reason}")

    # =========================================================
    # NAVIGATION
    # =========================================================
    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("⬅️ Back to Discovery"):
            st.session_state.step = 1
            st.rerun()

    with c2:
        if viable_models:
            if st.button("➡️ Proceed to Scoring", type="primary"):
                st.session_state.step = 3
                st.rerun()
        else:
            st.button(
                "➡️ Proceed to Scoring",
                disabled=True,
                help="No viable models available",
            )


def render_filtering_summary():
    """Compact summary used in later steps (unchanged, but cleaner)"""

    if "filtering_summary" not in st.session_state:
        return

    summary = st.session_state.filtering_summary

    with st.expander("🔧 Filtering Results"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Models", summary["total_models"])
        c2.metric("Eliminated", summary["eliminated_count"])
        c3.metric("Viable", summary["viable_count"])

        if summary["viable_count"] > 0:
            names = ", ".join(m["name"] for m in summary["viable_models"])
            st.success(f"✅ Viable Models: {names}")
        else:
            st.error("❌ No models passed filtering")
