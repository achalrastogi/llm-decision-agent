"""
Scoring page UI components
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agents.decision_engine import DecisionEngineAgent
from models.registry import ModelRegistry
from models.schemas import UserConstraints, ModelScore


def render_scoring_page(registry: ModelRegistry, constraints: UserConstraints, viable_models: list):
    """Render the model scoring and trade-off analysis page"""
    st.header("⚖️ Step 3: Model Scoring & Trade-off Analysis")
    st.markdown("Analyzing viable models using your priority weights...")
    
    # Initialize decision engine
    if 'decision_engine' not in st.session_state:
        st.session_state.decision_engine = DecisionEngineAgent(registry)
    
    decision_engine = st.session_state.decision_engine
    
    # Score models
    with st.spinner("Scoring models based on your priorities..."):
        model_scores = decision_engine.score_models(viable_models, constraints)
        trade_off_analysis = decision_engine.generate_trade_off_analysis(model_scores)
    
    # Store results in session state
    st.session_state.model_scores = model_scores
    st.session_state.trade_off_analysis = trade_off_analysis
    
    if not model_scores:
        st.error("❌ No models to score!")
        return
    
    # Display scoring results
    st.subheader("🏆 Model Rankings")
    
    # Create scoring summary table
    scoring_data = []
    for i, score in enumerate(model_scores):
        scoring_data.append({
            "Rank": i + 1,
            "Model": score.model_name,
            "Overall Score": f"{score.overall_score:.3f}",
            "Reasoning": f"{score.dimension_scores.get('reasoning', 0):.3f}",
            "Latency": f"{score.dimension_scores.get('latency', 0):.3f}",
            "Cost": f"{score.dimension_scores.get('cost', 0):.3f}",
            "Reliability": f"{score.dimension_scores.get('reliability', 0):.3f}"
        })
    
    scoring_df = pd.DataFrame(scoring_data)
    st.dataframe(scoring_df, use_container_width=True)
    
    # Visualization: Overall scores
    st.subheader("📊 Score Visualization")
    
    # Bar chart of overall scores
    fig_bar = px.bar(
        x=[score.model_name for score in model_scores],
        y=[score.overall_score for score in model_scores],
        title="Overall Model Scores",
        labels={'x': 'Model', 'y': 'Overall Score'},
        color=[score.overall_score for score in model_scores],
        color_continuous_scale='viridis'
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Radar chart for top 3 models
    if len(model_scores) >= 2:
        st.subheader("🎯 Multi-Dimensional Comparison")
        
        top_models = model_scores[:min(3, len(model_scores))]
        
        fig_radar = go.Figure()
        
        dimensions = ['reasoning', 'latency', 'cost', 'reliability']
        
        for model_score in top_models:
            values = [model_score.dimension_scores.get(dim, 0) for dim in dimensions]
            values.append(values[0])  # Close the radar chart
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=dimensions + [dimensions[0]],
                fill='toself',
                name=model_score.model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(score.dimension_scores.values()) for score in top_models)]
                )),
            showlegend=True,
            title="Top Models - Multi-Dimensional Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Detailed model analysis
    st.subheader("🔍 Detailed Model Analysis")
    
    selected_model_name = st.selectbox(
        "Select a model for detailed analysis:",
        options=[score.model_name for score in model_scores],
        key="detailed_analysis_selector"
    )
    
    if selected_model_name:
        selected_score = next(score for score in model_scores if score.model_name == selected_model_name)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Overall Score:** {selected_score.overall_score:.3f}")
            st.markdown("**Dimension Scores:**")
            for dim, score in selected_score.dimension_scores.items():
                st.write(f"• **{dim.title()}:** {score:.3f}")
        
        with col2:
            # Get model info for additional details
            model_id = None
            for mid, model_info in registry.get_all_models().items():
                if model_info.name == selected_model_name:
                    model_id = mid
                    break
            
            if model_id:
                model_info = registry.get_model(model_id)
                st.markdown("**Model Specifications:**")
                st.write(f"• **Provider:** {model_info.provider}")
                st.write(f"• **Context Window:** {model_info.context_window:,}")
                st.write(f"• **Cost Tier:** {model_info.cost_tier.value}")
                st.write(f"• **Latency Category:** {model_info.latency_category.value}")
        
        # Show explanations
        st.markdown("**Score Explanations:**")
        for dimension, explanation in selected_score.explanations.items():
            with st.expander(f"{dimension.title()} Score Explanation"):
                st.write(explanation)
    
    # Trade-off analysis
    if len(model_scores) >= 2:
        st.subheader("⚖️ Trade-off Analysis")
        
        trade_offs = trade_off_analysis.get('trade_offs', [])
        
        if trade_offs:
            st.markdown(f"**Comparing alternatives to top choice: {model_scores[0].model_name}**")
            
            for trade_off in trade_offs:
                with st.expander(f"#{trade_off['alternative_rank']} {trade_off['alternative_model']} vs {trade_off['primary_model']}"):
                    
                    score_diff = trade_off['score_difference']
                    st.write(f"**Score Difference:** {score_diff:.3f} (lower)")
                    
                    if trade_off['advantages']:
                        st.markdown("**Where it's stronger:**")
                        for adv in trade_off['advantages']:
                            st.write(f"• {adv['advantage']} (+{adv['score_difference']:.3f})")
                    
                    if trade_off['disadvantages']:
                        st.markdown("**Where it's weaker:**")
                        for dis in trade_off['disadvantages']:
                            st.write(f"• {dis['disadvantage']} (-{dis['score_difference']:.3f})")
        
        # Recommendations
        recommendations = trade_off_analysis.get('recommendations', [])
        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.info(rec)
        
        # Score gap analysis
        gap_analysis = trade_off_analysis.get('score_gap_analysis', {})
        if gap_analysis:
            st.subheader("📈 Score Gap Analysis")
            
            gaps = gap_analysis.get('gaps', [])
            if gaps:
                gap_data = []
                for gap in gaps:
                    gap_data.append({
                        "Rank Comparison": f"#{gap['rank_from']} vs #{gap['rank_to']}",
                        "Score Gap": f"{gap['score_gap']:.3f}",
                        "Significance": gap['gap_significance']
                    })
                
                gap_df = pd.DataFrame(gap_data)
                st.dataframe(gap_df, use_container_width=True)
                
                if gap_analysis.get('close_competition'):
                    st.warning("⚠️ Close competition detected! Consider running evaluation tests.")
    
    # Priority weights impact
    st.subheader("🎚️ Priority Weights Impact")
    
    weights_data = []
    for dimension, weight in constraints.priority_weights.items():
        weights_data.append({
            "Dimension": dimension.title(),
            "Your Weight": f"{weight:.1%}",
            "Impact": "High" if weight > 0.3 else "Medium" if weight > 0.2 else "Low"
        })
    
    weights_df = pd.DataFrame(weights_data)
    st.dataframe(weights_df, use_container_width=True)
    
    # Show scoring methodology
    with st.expander("📚 Scoring Methodology"):
        methodology = decision_engine.explain_scoring_methodology()
        st.markdown(methodology)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Filtering"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("➡️ Get Recommendation", type="primary"):
            st.session_state.step = 4
            st.rerun()


def render_scoring_summary():
    """Render a compact summary of scoring results for other pages"""
    if not hasattr(st.session_state, 'model_scores'):
        return
    
    model_scores = st.session_state.model_scores
    
    with st.expander("⚖️ Scoring Results"):
        if model_scores:
            st.success(f"✅ {len(model_scores)} models scored and ranked")
            
            # Show top 3
            top_3 = model_scores[:3]
            for i, score in enumerate(top_3):
                st.write(f"**#{i+1} {score.model_name}:** {score.overall_score:.3f}")
            
            # Show score gap
            if len(model_scores) >= 2:
                gap = model_scores[0].overall_score - model_scores[1].overall_score
                st.write(f"**Score Gap (1st-2nd):** {gap:.3f}")
        else:
            st.error("❌ No scoring results available")