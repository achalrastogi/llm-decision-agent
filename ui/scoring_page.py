import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agents.decision_engine import DecisionEngineAgent
from models.registry import ModelRegistry
from models.schemas import UserConstraints


def render_scoring_page(
    registry: ModelRegistry,
    constraints: UserConstraints,
    viable_models: list,
):
    """Render the model scoring page (UI-enhanced, logic unchanged)"""

    st.markdown("## ⚖️ Step 3: Model Scoring")
    st.caption("Ranking viable models using your priority weights")

    # -----------------------------
    # Safety guards
    # -----------------------------
    if not viable_models:
        st.error("No viable models found. Please complete filtering first.")
        return

    # -----------------------------
    # Initialize decision engine
    # -----------------------------
    if "decision_engine" not in st.session_state:
        st.session_state.decision_engine = DecisionEngineAgent(registry)

    engine = st.session_state.decision_engine

    # -----------------------------
    # Run scoring (unchanged)
    # -----------------------------
    with st.spinner("Scoring models based on your priorities..."):
        model_scores = engine.score_models(viable_models, constraints)
        trade_off_analysis = engine.generate_trade_off_analysis(model_scores)

    st.session_state.model_scores = model_scores
    st.session_state.trade_off_analysis = trade_off_analysis

    if not model_scores:
        st.error("❌ No models could be scored.")
        return

    # =========================================================
    # SECTION 1 — WINNER SPOTLIGHT
    # =========================================================
    top = model_scores[0]

    st.markdown("### 🏆 Current Leader")

    st.markdown(
        f"""
        <div class="card">
            <h3>{top.model_name}</h3>
            <b>Overall Score:</b> {top.overall_score:.3f}<br/>
            <b>Top Strength:</b> {max(top.dimension_scores, key=top.dimension_scores.get).title()}<br/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # SECTION 2 — LEADERBOARD (BAR CHART)
    # =========================================================
    st.markdown("### 📊 Score Comparison")

    fig_bar = px.bar(
        x=[s.model_name for s in model_scores],
        y=[s.overall_score for s in model_scores],
        labels={"x": "Model", "y": "Overall Score"},
        color=[s.overall_score for s in model_scores],
        color_continuous_scale="Blues",
    )

    fig_bar.update_layout(
        showlegend=False,
        template="plotly_dark",
        margin=dict(t=20, b=40),
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================
    # SECTION 3 — MULTI-DIMENSION COMPARISON
    # =========================================================
    if len(model_scores) > 1:
        st.markdown("### 🎯 Strengths Across Dimensions")

        top_models = model_scores[: min(3, len(model_scores))]
        dimensions = list(top_models[0].dimension_scores.keys())

        fig_radar = go.Figure()

        for m in top_models:
            values = [m.dimension_scores[d] for d in dimensions]
            values.append(values[0])

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=dimensions + [dimensions[0]],
                    fill="toself",
                    name=m.model_name,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_dark",
            showlegend=True,
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    # =========================================================
    # SECTION 4 — DETAILED BREAKDOWN (OPTIONAL)
    # =========================================================
    with st.expander("🔍 Detailed Scores & Explanations"):
        scoring_df = pd.DataFrame(
            [
                {
                    "Rank": i + 1,
                    "Model": s.model_name,
                    "Overall": f"{s.overall_score:.3f}",
                    **{k.title(): f"{v:.3f}" for k, v in s.dimension_scores.items()},
                }
                for i, s in enumerate(model_scores)
            ]
        )
        st.dataframe(scoring_df, use_container_width=True)

        selected = st.selectbox(
            "Inspect a model",
            [s.model_name for s in model_scores],
        )

        score = next(s for s in model_scores if s.model_name == selected)

        st.markdown(f"**{selected} — Dimension Explanations**")
        for dim, explanation in score.explanations.items():
            with st.expander(dim.title()):
                st.write(explanation)

    # =========================================================
    # SECTION 5 — TRADE-OFF ANALYSIS
    # =========================================================
    trade_offs = trade_off_analysis.get("trade_offs", [])

    if trade_offs:
        st.markdown("### ⚖️ Trade-offs vs Top Choice")

        for t in trade_offs:
            with st.expander(
                f"{t['alternative_model']} vs {t['primary_model']} "
                f"(−{t['score_difference']:.3f})"
            ):
                if t.get("advantages"):
                    st.markdown("**Where this model is stronger**")
                    for a in t["advantages"]:
                        st.write(f"• {a['advantage']} (+{a['score_difference']:.3f})")

                if t.get("disadvantages"):
                    st.markdown("**Where it is weaker**")
                    for d in t["disadvantages"]:
                        st.write(f"• {d['disadvantage']} (−{d['score_difference']:.3f})")

    # =========================================================
    # SECTION 6 — PRIORITY WEIGHTS (CONTEXT)
    # =========================================================
    st.markdown("### 🎚️ Your Priority Weights")

    weights_df = pd.DataFrame(
        [
            {
                "Dimension": k.title(),
                "Weight": f"{v:.1%}",
                "Impact": "High" if v > 0.3 else "Medium" if v > 0.2 else "Low",
            }
            for k, v in constraints.priority_weights.items()
        ]
    )

    st.dataframe(weights_df, use_container_width=True)

    # =========================================================
    # NAVIGATION
    # =========================================================
    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("⬅️ Back to Filtering"):
            st.session_state.step = 2
            st.rerun()

    with c2:
        if st.button("➡️ Get Recommendation", type="primary"):
            st.session_state.step = 4
            st.rerun()


def render_scoring_summary():
    """Compact summary for later steps"""

    if "model_scores" not in st.session_state:
        return

    scores = st.session_state.model_scores

    with st.expander("⚖️ Scoring Results"):
        if scores:
            for i, s in enumerate(scores[:3]):
                st.write(f"**#{i+1} {s.model_name}** — {s.overall_score:.3f}")

            if len(scores) > 1:
                gap = scores[0].overall_score - scores[1].overall_score
                st.write(f"Score gap (1st vs 2nd): {gap:.3f}")
        else:
            st.error("No scoring data available")
