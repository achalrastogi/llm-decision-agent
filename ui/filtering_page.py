"""
Filtering page UI components
"""

import streamlit as st
import pandas as pd
from agents.constraint_filter import ConstraintFilterAgent
from models.registry import ModelRegistry
from models.schemas import UserConstraints


def render_filtering_page(registry: ModelRegistry, constraints: UserConstraints):
    """Render the constraint filtering page"""
    st.header("🔧 Step 2: Model Filtering")
    st.markdown("Applying your constraints to eliminate incompatible models...")
    
    # Initialize filter agent
    if 'filter_agent' not in st.session_state:
        st.session_state.filter_agent = ConstraintFilterAgent(registry)
    
    filter_agent = st.session_state.filter_agent
    
    # Apply filtering
    with st.spinner("Filtering models based on your constraints..."):
        viable_models, elimination_reasons = filter_agent.filter_models(constraints)
        filtering_summary = filter_agent.get_filtering_summary(viable_models, elimination_reasons)
        constraint_impacts = filter_agent.get_constraint_impact_analysis(constraints)
    
    # Store results in session state
    st.session_state.viable_models = viable_models
    st.session_state.elimination_reasons = elimination_reasons
    st.session_state.filtering_summary = filtering_summary
    
    # Display filtering results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", filtering_summary['total_models'])
    with col2:
        st.metric("Eliminated", filtering_summary['eliminated_count'], 
                 delta=f"-{filtering_summary['elimination_rate']:.1%}")
    with col3:
        st.metric("Viable Models", filtering_summary['viable_count'],
                 delta=f"+{filtering_summary['viable_count']/filtering_summary['total_models']:.1%}")
    
    # Show constraint impact analysis
    st.subheader("📊 Constraint Impact Analysis")
    st.markdown("How many models each constraint eliminates:")
    
    impact_data = []
    for constraint, count in constraint_impacts.items():
        if count > 0:  # Only show constraints that eliminate models
            impact_data.append({
                "Constraint": constraint.replace('_', ' ').title(),
                "Models Eliminated": count,
                "Impact": f"{count/filtering_summary['total_models']:.1%}"
            })
    
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)
        
        # Visualization
        st.bar_chart(impact_df.set_index('Constraint')['Models Eliminated'])
    else:
        st.success("🎉 No models were eliminated by your constraints!")
    
    # Display viable models
    if viable_models:
        st.subheader("✅ Viable Models")
        st.success(f"Found {len(viable_models)} models that meet your requirements:")
        
        viable_data = []
        for model_detail in filtering_summary['viable_models']:
            viable_data.append({
                "Model": model_detail['name'],
                "Provider": model_detail['provider'],
                "Cost Tier": model_detail['cost_tier'],
                "Latency": model_detail['latency'],
                "Context Window": f"{model_detail['context_window']:,}",
                "Reasoning": f"{model_detail['reasoning_strength']}/10"
            })
        
        viable_df = pd.DataFrame(viable_data)
        st.dataframe(viable_df, use_container_width=True)
        
        # Model details expander
        with st.expander("🔍 Detailed Model Information"):
            selected_viable = st.selectbox(
                "Select a viable model for details:",
                options=viable_models,
                format_func=lambda x: registry.get_model(x).name
            )
            
            if selected_viable:
                model_info = registry.get_model(selected_viable)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {model_info.name}")
                    st.write(f"**Provider:** {model_info.provider}")
                    st.write(f"**Context Window:** {model_info.context_window:,}")
                    st.write(f"**Latency Category:** {model_info.latency_category.value}")
                    st.write(f"**Cost Tier:** {model_info.cost_tier.value}")
                
                with col2:
                    st.write(f"**Reasoning Strength:** {model_info.reasoning_strength}/10")
                    st.write(f"**Tool Reliability:** {model_info.tool_reliability}/10")
                    st.write(f"**Input Cost:** ${model_info.cost_per_token['input']:.6f}")
                    st.write(f"**Output Cost:** ${model_info.cost_per_token['output']:.6f}")
                
                st.write("**Capabilities:**")
                for cap in model_info.capabilities:
                    st.write(f"• {cap}")
    else:
        st.error("❌ No models meet your requirements!")
        st.markdown("**Suggestions:**")
        st.markdown("• Consider relaxing your budget constraints")
        st.markdown("• Try a different latency tolerance")
        st.markdown("• Reduce context window requirements")
        st.markdown("• Review required capabilities")
    
    # Display eliminated models
    if elimination_reasons:
        st.subheader("❌ Eliminated Models")
        
        # Group by constraint type
        elimination_by_constraint = filtering_summary['elimination_by_constraint']
        
        for constraint_type, reasons in elimination_by_constraint.items():
            with st.expander(f"Eliminated by {constraint_type.replace('_', ' ').title()} ({len(reasons)} models)"):
                for reason in reasons:
                    st.write(f"**{reason.model_name}:** {reason.reason}")
        
        # Detailed elimination analysis
        with st.expander("🔍 Detailed Elimination Analysis"):
            selected_eliminated = st.selectbox(
                "Select an eliminated model for detailed explanation:",
                options=[r.model_name for r in elimination_reasons],
                key="eliminated_selector"
            )
            
            if selected_eliminated:
                # Find the model ID for the selected name
                eliminated_model_id = None
                for model_id, model_info in registry.get_all_models().items():
                    if model_info.name == selected_eliminated:
                        eliminated_model_id = model_id
                        break
                
                if eliminated_model_id:
                    explanation = filter_agent.explain_elimination(eliminated_model_id, constraints)
                    st.markdown(explanation)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Discovery"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if viable_models and st.button("➡️ Proceed to Scoring", type="primary"):
            st.session_state.step = 3
            st.rerun()
        elif not viable_models:
            st.button("➡️ Proceed to Scoring", disabled=True, 
                     help="No viable models to score. Please adjust your constraints.")


def render_filtering_summary():
    """Render a compact summary of filtering results for other pages"""
    if not hasattr(st.session_state, 'filtering_summary'):
        return
    
    summary = st.session_state.filtering_summary
    
    with st.expander("🔧 Filtering Results"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Models", summary['total_models'])
        with col2:
            st.metric("Eliminated", summary['eliminated_count'])
        with col3:
            st.metric("Viable", summary['viable_count'])
        
        if summary['viable_count'] > 0:
            st.success(f"✅ {summary['viable_count']} models passed filtering")
            viable_names = [model['name'] for model in summary['viable_models']]
            st.write("**Viable Models:** " + ", ".join(viable_names))
        else:
            st.error("❌ No models passed filtering")