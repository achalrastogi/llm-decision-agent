import streamlit as st

from agents.recommendation import RecommendationAgent
from models.registry import ModelRegistry
from models.schemas import UserConstraints


def render_recommendation_page(
    registry: ModelRegistry,
    constraints: UserConstraints,
    model_scores: list,
    trade_off_analysis: dict,
):
    """Render the final recommendation page (UI-enhanced, logic unchanged)"""

    st.markdown("## 💡 Step 4: Final Recommendation")
    st.caption("Clear, actionable guidance based on your priorities")

    # -----------------------------
    # Safety guards
    # -----------------------------
    if not model_scores:
        st.error("No scoring data found. Please complete scoring first.")
        return

    # -----------------------------
    # Initialize agent
    # -----------------------------
    if "recommendation_agent" not in st.session_state:
        st.session_state.recommendation_agent = RecommendationAgent(registry)

    agent = st.session_state.recommendation_agent

    # -----------------------------
    # Generate recommendation (unchanged)
    # -----------------------------
    with st.spinner("Generating final recommendation..."):
        recommendation = agent.generate_recommendation(
            model_scores, trade_off_analysis, constraints
        )

    st.session_state.recommendation = recommendation

    # =========================================================
    # NO SUITABLE MODEL CASE
    # =========================================================
    if not recommendation.get("primary_model"):
        render_no_models_recommendation(recommendation)
        render_navigation_actions(recommendation)
        return

    primary = recommendation["primary_model"]

    # =========================================================
    # SECTION 1 — FINAL DECISION (NON-NEGOTIABLE)
    # =========================================================
    st.markdown(
        f"""
        <div class="card">
            <h2>🏆 Final Recommendation</h2>
            <h3>{primary["name"]}</h3>
            <b>Overall Score:</b> {primary["score"]:.3f}<br/>
            <b>Confidence:</b> {primary["confidence"].title()}<br/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # SECTION 2 — WHY THIS MODEL
    # =========================================================
    st.markdown("### 🎯 Why this model")

    st.markdown(recommendation["decision_rationale"])

    with st.expander("📋 Context considered"):
        st.markdown(recommendation["context"])

    # =========================================================
    # SECTION 3 — RISKS & MITIGATIONS
    # =========================================================
    mitigations = recommendation.get("mitigation_strategies", [])

    if mitigations:
        st.markdown("### ⚠️ Risks & Mitigations")

        for m in mitigations:
            st.warning(m)

    # =========================================================
    # SECTION 4 — TRADE-OFFS (ONLY IF THEY MATTER)
    # =========================================================
    trade_offs = recommendation.get("trade_offs", [])

    if trade_offs:
        st.markdown("### ⚖️ Trade-offs vs alternatives")

        for t in trade_offs:
            with st.expander(
                f"{t['alternative_model']} "
                f"(−{t['score_difference']:.3f} vs {primary['name']})"
            ):
                st.markdown(t["trade_off_summary"])
                st.info(t["when_to_consider"])

    # =========================================================
    # SECTION 5 — ALTERNATIVES (SECONDARY OPTIONS)
    # =========================================================
    alternatives = recommendation.get("alternatives", [])

    if alternatives:
        st.markdown("### 🔄 Secondary Options")

        for alt in alternatives:
            with st.expander(f"{alt['name']} — {alt['key_strength']}"):
                st.write(f"**Score:** {alt['score']:.3f}")
                st.write(f"**Best for:** {alt['use_case']}")

    # =========================================================
    # SECTION 6 — IMPLEMENTATION & FUTURE-PROOFING
    # =========================================================
    guidance = recommendation.get("implementation_guidance", [])
    future = recommendation.get("future_proofing", [])

    if guidance or future:
        st.markdown("### 🚀 Implementation & Next Steps")

        if guidance:
            st.markdown("**Implementation Guidance**")
            for g in guidance:
                st.write(f"• {g}")

        if future:
            st.markdown("**Future-proofing Considerations**")
            for f in future:
                st.write(f"• {f}")

    # =========================================================
    # SECTION 7 — EXECUTIVE SUMMARY (ONE SCREEN)
    # =========================================================
    st.markdown("### 📊 Executive Summary")

    st.markdown(
        f"""
        <div class="card">
            <b>Chosen Model:</b> {primary["name"]}<br/>
            <b>Confidence:</b> {primary["confidence"].title()}<br/>
            <b>Key Reason:</b> Highest weighted score aligned with your priorities<br/>
            <b>Alternatives Considered:</b> {len(alternatives)}<br/>
            <b>Mitigations Identified:</b> {len(mitigations)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # NAVIGATION & ACTIONS
    # =========================================================
    render_navigation_actions(recommendation)


# ------------------------------------------------------------------
# FALLBACK CASE
# ------------------------------------------------------------------
def render_no_models_recommendation(recommendation: dict):
    st.error("❌ No Suitable Model Found")

    st.markdown("### Situation")
    st.markdown(recommendation.get("context", "Constraints were too restrictive."))

    st.markdown("### Suggested Actions")
    for s in recommendation.get("mitigation_strategies", []):
        st.write(f"• {s}")


# ------------------------------------------------------------------
# NAVIGATION & EXPORT
# ------------------------------------------------------------------
def render_navigation_actions(recommendation: dict):
    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("⬅️ Back to Scoring"):
            st.session_state.step = 3
            st.rerun()

    with c2:
        if st.button("🔄 Start Over"):
            for key in [
                "constraints",
                "viable_models",
                "model_scores",
                "trade_off_analysis",
                "recommendation",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()

    with c3:
        if recommendation.get("primary_model"):
            if st.button("📄 Export Recommendation"):
                export_recommendation(recommendation)


def export_recommendation(recommendation: dict):
    primary = recommendation["primary_model"]

    text = f"""
# LLM Model Recommendation

## Final Choice
{primary["name"]}

Score: {primary["score"]:.3f}
Confidence: {primary["confidence"].title()}

## Rationale
{recommendation["decision_rationale"]}

## Risks & Mitigations
"""

    for m in recommendation.get("mitigation_strategies", []):
        text += f"- {m}\n"

    st.download_button(
        "📥 Download Decision Report",
        text,
        file_name=f"llm_recommendation_{primary['name'].lower().replace(' ', '_')}.md",
        mime="text/markdown",
    )


def render_recommendation_summary():
    """Compact summary for previous steps"""

    if "recommendation" not in st.session_state:
        return

    rec = st.session_state.recommendation

    with st.expander("💡 Final Recommendation"):
        if rec.get("primary_model"):
            p = rec["primary_model"]
            st.success(f"✅ {p['name']} — {p['confidence'].title()}")
            st.write(f"Score: {p['score']:.3f}")
        else:
            st.error("No recommendation available")
