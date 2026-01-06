"""
Recommendation page UI components
"""

import streamlit as st
from agents.recommendation import RecommendationAgent
from models.registry import ModelRegistry
from models.schemas import UserConstraints, ModelScore


def render_recommendation_page(registry: ModelRegistry, constraints: UserConstraints, 
                             model_scores: list, trade_off_analysis: dict):
    """Render the final recommendation page"""
    st.header("💡 Step 4: Final Recommendation")
    st.markdown("Your personalized LLM model recommendation with implementation guidance...")
    
    # Initialize recommendation agent
    if 'recommendation_agent' not in st.session_state:
        st.session_state.recommendation_agent = RecommendationAgent(registry)
    
    recommendation_agent = st.session_state.recommendation_agent
    
    # Generate recommendation
    with st.spinner("Generating your personalized recommendation..."):
        recommendation = recommendation_agent.generate_recommendation(
            model_scores, trade_off_analysis, constraints
        )
    
    # Store recommendation in session state
    st.session_state.recommendation = recommendation
    
    # Display recommendation
    if recommendation['primary_model']:
        render_primary_recommendation(recommendation)
        render_trade_offs(recommendation)
        render_mitigation_strategies(recommendation)
        render_implementation_guidance(recommendation)
        render_future_proofing(recommendation)
        render_alternatives(recommendation)
        render_decision_summary(recommendation)
    else:
        render_no_models_recommendation(recommendation)
    
    # Navigation and actions
    render_navigation_actions(recommendation)


def render_primary_recommendation(recommendation: dict):
    """Render the primary recommendation section"""
    primary = recommendation['primary_model']
    
    # Header with confidence indicator
    confidence_colors = {
        'high': '🟢',
        'medium': '🟡', 
        'low': '🟠'
    }
    
    confidence_icon = confidence_colors.get(primary['confidence'], '⚪')
    
    st.subheader(f"🏆 Recommended Model: {primary['name']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Score", f"{primary['score']:.3f}")
    with col2:
        st.metric("Confidence", primary['confidence'].title(), 
                 help=primary['confidence_reason'])
    with col3:
        st.write(f"{confidence_icon} **{primary['confidence'].title()} Confidence**")
    
    # Context and rationale
    st.markdown("### 📋 Recommendation Context")
    st.markdown(recommendation['context'])
    
    st.markdown("### 🎯 Decision Rationale")
    st.markdown(recommendation['decision_rationale'])


def render_trade_offs(recommendation: dict):
    """Render trade-offs section"""
    trade_offs = recommendation.get('trade_offs', [])
    
    if not trade_offs:
        return
    
    st.markdown("### ⚖️ Trade-offs with Alternatives")
    
    for i, trade_off in enumerate(trade_offs):
        with st.expander(f"Alternative #{i+2}: {trade_off['alternative_model']}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Score Difference", 
                         f"-{trade_off['score_difference']:.3f}",
                         delta=f"{trade_off['score_difference']:.3f}",
                         delta_color="inverse")
            
            with col2:
                st.markdown("**Trade-off Summary:**")
                st.write(trade_off['trade_off_summary'])
            
            st.markdown("**When to Consider:**")
            st.info(trade_off['when_to_consider'])


def render_mitigation_strategies(recommendation: dict):
    """Render mitigation strategies section"""
    strategies = recommendation.get('mitigation_strategies', [])
    
    if not strategies:
        return
    
    st.markdown("### 🛠️ Mitigation Strategies")
    st.markdown("Address potential weaknesses with these strategies:")
    
    for strategy in strategies:
        st.markdown(f"• {strategy}")


def render_implementation_guidance(recommendation: dict):
    """Render implementation guidance section"""
    guidance = recommendation.get('implementation_guidance', [])
    
    if not guidance:
        return
    
    st.markdown("### 🚀 Implementation Guidance")
    st.markdown("Practical steps to get started:")
    
    for guide in guidance:
        st.markdown(f"• {guide}")


def render_future_proofing(recommendation: dict):
    """Render future-proofing guidance section"""
    future_proofing = recommendation.get('future_proofing', [])
    
    if not future_proofing:
        return
    
    st.markdown("### 🔮 Future-Proofing Strategies")
    st.markdown("Prepare for scaling and evolution:")
    
    for strategy in future_proofing:
        st.markdown(f"• {strategy}")


def render_alternatives(recommendation: dict):
    """Render alternatives summary section"""
    alternatives = recommendation.get('alternatives', [])
    
    if not alternatives:
        return
    
    st.markdown("### 🔄 Alternative Options")
    st.markdown("Keep these models in mind for specific scenarios:")
    
    for alt in alternatives:
        with st.expander(f"{alt['name']} - {alt['key_strength']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Score", f"{alt['score']:.3f}")
                st.write(f"**Key Strength:** {alt['key_strength']}")
            
            with col2:
                st.write(f"**Best For:** {alt['use_case']}")


def render_decision_summary(recommendation: dict):
    """Render decision summary section"""
    st.markdown("### 📊 Decision Summary")
    
    primary = recommendation['primary_model']
    
    # Create summary card
    with st.container():
        st.markdown(f"""
        **Final Recommendation:** {primary['name']}
        
        **Confidence Level:** {primary['confidence'].title()} ({primary['confidence_reason']})
        
        **Key Reasons:**
        - Highest overall score ({primary['score']:.3f}) based on your priorities
        - {len(recommendation.get('mitigation_strategies', []))} mitigation strategies identified
        - {len(recommendation.get('alternatives', []))} alternative options available
        
        **Next Steps:** Review implementation guidance and mitigation strategies above
        """)


def render_no_models_recommendation(recommendation: dict):
    """Render recommendation when no models are available"""
    st.error("❌ No Suitable Models Found")
    
    st.markdown("### 📋 Situation")
    st.markdown(recommendation['context'])
    
    st.markdown("### 🛠️ Suggested Actions")
    strategies = recommendation.get('mitigation_strategies', [])
    for strategy in strategies:
        st.markdown(f"• {strategy}")
    
    st.markdown("### 🔮 Future Options")
    future_proofing = recommendation.get('future_proofing', [])
    for strategy in future_proofing:
        st.markdown(f"• {strategy}")
    
    st.markdown("### 🚀 Implementation Alternatives")
    guidance = recommendation.get('implementation_guidance', [])
    for guide in guidance:
        st.markdown(f"• {guide}")


def render_navigation_actions(recommendation: dict):
    """Render navigation and action buttons"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back to Scoring"):
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if recommendation['primary_model']:
            if st.button("🔄 Start Over", help="Begin with new requirements"):
                # Clear session state
                for key in ['constraints', 'viable_models', 'model_scores', 
                           'trade_off_analysis', 'recommendation']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.step = 1
                st.rerun()
    
    with col3:
        if recommendation['primary_model']:
            # Export recommendation
            if st.button("📄 Export Recommendation"):
                export_recommendation(recommendation)


def export_recommendation(recommendation: dict):
    """Export recommendation as downloadable text"""
    primary = recommendation['primary_model']
    
    export_text = f"""
# LLM Model Recommendation Report

## Primary Recommendation: {primary['name']}
- **Overall Score:** {primary['score']:.3f}
- **Confidence:** {primary['confidence'].title()}
- **Reason:** {primary['confidence_reason']}

## Context
{recommendation['context']}

## Decision Rationale
{recommendation['decision_rationale']}

## Trade-offs with Alternatives
"""
    
    for i, trade_off in enumerate(recommendation.get('trade_offs', [])):
        export_text += f"""
### Alternative #{i+2}: {trade_off['alternative_model']}
- **Score Difference:** -{trade_off['score_difference']:.3f}
- **Trade-off:** {trade_off['trade_off_summary']}
- **When to Consider:** {trade_off['when_to_consider']}
"""
    
    export_text += f"""
## Mitigation Strategies
"""
    for strategy in recommendation.get('mitigation_strategies', []):
        export_text += f"- {strategy}\n"
    
    export_text += f"""
## Implementation Guidance
"""
    for guide in recommendation.get('implementation_guidance', []):
        export_text += f"- {guide}\n"
    
    export_text += f"""
## Future-Proofing Strategies
"""
    for strategy in recommendation.get('future_proofing', []):
        export_text += f"- {strategy}\n"
    
    st.download_button(
        label="📥 Download Report",
        data=export_text,
        file_name=f"llm_recommendation_{primary['name'].lower().replace(' ', '_')}.md",
        mime="text/markdown"
    )


def render_recommendation_summary():
    """Render a compact summary of recommendation for other pages"""
    if not hasattr(st.session_state, 'recommendation'):
        return
    
    recommendation = st.session_state.recommendation
    
    with st.expander("💡 Final Recommendation"):
        if recommendation.get('primary_model'):
            primary = recommendation['primary_model']
            st.success(f"✅ Recommended: **{primary['name']}**")
            st.write(f"**Score:** {primary['score']:.3f}")
            st.write(f"**Confidence:** {primary['confidence'].title()}")
            
            # Show key context
            context_preview = recommendation['context'][:150] + "..." if len(recommendation['context']) > 150 else recommendation['context']
            st.write(f"**Context:** {context_preview}")
        else:
            st.error("❌ No suitable models found")
            st.write("Consider adjusting your constraints")