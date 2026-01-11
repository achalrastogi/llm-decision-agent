import streamlit as st
from agents.recommendation import create_recommendation_agent
from models.registry import ModelRegistry


# =====================================================
# MAIN PAGE RENDER
# =====================================================
def render_recommendation_page(
    registry,
    constraints,
    model_scores,
    trade_off_analysis,
):
    import streamlit as st
    from agents.recommendation import create_recommendation_agent

    st.header("💡 Step 4: Recommendation")

    # Decide LLM usage centrally
    enable_llm = st.session_state.get("llm_enabled", False)
    llm_config = st.session_state.get("llm_config")

    agent = create_recommendation_agent(
        registry=registry,
        enable_llm=enable_llm,
        llm_config=llm_config,
    )

    with st.spinner("Generating recommendation..."):
        recommendation = agent.generate_recommendation(
            model_scores=model_scores,
            trade_off_analysis=trade_off_analysis,
            constraints=constraints,
        )

    st.session_state.recommendation = recommendation
    render_recommendation_summary(recommendation)

# =====================================================
# SUMMARY RENDER (USED BY app.py)
# =====================================================
def render_recommendation_summary(recommendation: dict):
    if not recommendation:
        st.error("No recommendation available.")
        return

    primary = recommendation.get("primary_model")

    if not primary:
        st.error(recommendation.get("decision_rationale", "No viable models found."))
        return

    # -----------------------------
    # Primary recommendation
    # -----------------------------
    st.subheader("✅ Recommended Model")

    st.markdown(
        f"""
        **Model:** {primary['name']}  
        **Score:** {primary['score']:.3f}  
        **Confidence:** {primary['confidence']}  
        _{primary['confidence_reason']}_
        """
    )

    # -----------------------------
    # Decision rationale
    # -----------------------------
    st.subheader("📌 Why this model?")
    st.write(recommendation.get("decision_rationale", ""))

    # -----------------------------
    # Trade-offs
    # -----------------------------
    if recommendation.get("key_trade_offs"):
        st.subheader("⚖️ Key Trade-offs")
        for t in recommendation["key_trade_offs"]:
            st.write(f"• {t}")

    # -----------------------------
    # Mitigation strategies
    # -----------------------------
    if recommendation.get("mitigation_strategies"):
        st.subheader("🛠 Mitigation Strategies")
        for m in recommendation["mitigation_strategies"]:
            st.write(f"• {m}")

    # -----------------------------
    # Implementation guidance
    # -----------------------------
    if recommendation.get("implementation_guidance"):
        st.subheader("🚀 Implementation Guidance")
        for g in recommendation["implementation_guidance"]:
            st.write(f"• {g}")

    # -----------------------------
    # Future-proofing
    # -----------------------------
    if recommendation.get("future_proofing"):
        st.subheader("🔮 Future-proofing")
        for f in recommendation["future_proofing"]:
            st.write(f"• {f}")

    # -----------------------------
    # Alternatives
    # -----------------------------
    if recommendation.get("alternatives"):
        st.subheader("🔁 Alternatives")
        for alt in recommendation["alternatives"]:
            st.markdown(
                f"""
                **{alt['name']}**  
                Score: {alt['score']:.3f}  
                Key strength: {alt['key_strength']}
                """
            )
