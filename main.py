"""
Context-Aware LLM Model Comparison & Decision Agent
Main Streamlit application entry point
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from models.registry import ModelRegistry
from models.schemas import ModelInfo
from ui.discovery_page import render_discovery_page, render_constraints_summary
from ui.filtering_page import render_filtering_page, render_filtering_summary
from ui.scoring_page import render_scoring_page, render_scoring_summary
from ui.recommendation_page import render_recommendation_page, render_recommendation_summary

# Configure Streamlit page
st.set_page_config(
    page_title="LLM Decision Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_model_registry():
    """Load model registry with caching"""
    try:
        registry = ModelRegistry()
        return registry
    except Exception as e:
        st.error(f"Error loading model registry: {e}")
        return None

def render_progress_indicator(current_step: int):
    """Render progress indicator for the wizard"""
    steps = [
        "🔍 Discovery",
        "🔧 Filtering", 
        "⚖️ Scoring",
        "💡 Recommendation"
    ]
    
    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current_step:
                st.success(f"✅ {step_name}")
            elif i + 1 == current_step:
                st.info(f"🔄 {step_name}")
            else:
                st.write(f"⏳ {step_name}")

def display_model_registry(registry: ModelRegistry):
    """Display the model registry in a user-friendly format"""
    st.header("📊 Available LLM Models")
    
    models = registry.get_all_models()
    
    if not models:
        st.warning("No models found in registry")
        return
    
    # Create a summary table
    model_data = []
    for model_id, model_info in models.items():
        model_data.append({
            "Model ID": model_id,
            "Name": model_info.name,
            "Provider": model_info.provider,
            "Context Window": f"{model_info.context_window:,}",
            "Latency": model_info.latency_category.value,
            "Cost Tier": model_info.cost_tier.value,
            "Reasoning": f"{model_info.reasoning_strength}/10",
            "Reliability": f"{model_info.tool_reliability}/10"
        })
    
    df = pd.DataFrame(model_data)
    st.dataframe(df, use_container_width=True)
    
    # Show detailed model information
    st.subheader("🔍 Model Details")
    selected_model = st.selectbox(
        "Select a model to view details:",
        options=list(models.keys()),
        format_func=lambda x: f"{models[x].name} ({models[x].provider})"
    )
    
    if selected_model:
        model_info = models[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            st.write(f"**Name:** {model_info.name}")
            st.write(f"**Provider:** {model_info.provider}")
            st.write(f"**Context Window:** {model_info.context_window:,} tokens")
            st.write(f"**Latency Category:** {model_info.latency_category.value}")
            st.write(f"**Cost Tier:** {model_info.cost_tier.value}")
            
            st.markdown("**Capabilities**")
            for capability in model_info.capabilities:
                st.write(f"• {capability}")
        
        with col2:
            st.markdown("**Performance Metrics**")
            st.write(f"**Reasoning Strength:** {model_info.reasoning_strength}/10")
            st.write(f"**Tool Reliability:** {model_info.tool_reliability}/10")
            
            st.markdown("**Cost per Token**")
            st.write(f"**Input:** ${model_info.cost_per_token['input']:.6f}")
            st.write(f"**Output:** ${model_info.cost_per_token['output']:.6f}")
            
            st.markdown("**Benchmark Scores**")
            for benchmark, score in model_info.benchmark_scores.items():
                st.write(f"**{benchmark.upper()}:** {score}%")
            
            st.markdown("**Deployment Options**")
            for option in model_info.deployment_options:
                st.write(f"• {option}")

def main():
    """Main application entry point"""
    st.title("🤖 Context-Aware LLM Model Comparison & Decision Agent")
    st.markdown("---")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
        st.session_state.constraints = None
        st.session_state.results = {}
    
    # Load model registry
    registry = load_model_registry()
    
    if registry is None:
        st.error("❌ Failed to load model registry")
        return
    
    # Show progress indicator
    render_progress_indicator(st.session_state.step)
    st.markdown("---")
    
    # Route to appropriate page based on step
    if st.session_state.step == 1:
        # Discovery phase
        render_discovery_page()
        
        # Show model registry in sidebar or expander for reference
        with st.expander("📚 Available Models Reference"):
            st.info(f"📋 Found {len(registry.get_all_models())} models from {len(registry.get_providers())} providers")
            
            # Quick model summary
            models = registry.get_all_models()
            model_summary = []
            for model_id, model_info in models.items():
                model_summary.append({
                    "Model": model_info.name,
                    "Provider": model_info.provider,
                    "Context": f"{model_info.context_window:,}",
                    "Latency": model_info.latency_category.value,
                    "Cost": model_info.cost_tier.value
                })
            
            df = pd.DataFrame(model_summary)
            st.dataframe(df, use_container_width=True)
    
    elif st.session_state.step == 2:
        # Filtering phase
        st.header("🔧 Step 2: Model Filtering")
        render_constraints_summary(st.session_state.constraints)
        
        if st.session_state.constraints:
            render_filtering_page(registry, st.session_state.constraints)
        else:
            st.error("❌ No constraints found. Please go back to Discovery.")
            if st.button("⬅️ Back to Discovery"):
                st.session_state.step = 1
                st.rerun()
    
    elif st.session_state.step == 3:
        # Scoring phase
        st.header("⚖️ Step 3: Model Scoring")
        render_constraints_summary(st.session_state.constraints)
        render_filtering_summary()
        
        if (hasattr(st.session_state, 'viable_models') and 
            st.session_state.viable_models and 
            st.session_state.constraints):
            render_scoring_page(registry, st.session_state.constraints, st.session_state.viable_models)
        else:
            st.error("❌ No viable models to score. Please go back to filtering.")
            if st.button("⬅️ Back to Filtering"):
                st.session_state.step = 2
                st.rerun()
    
    elif st.session_state.step == 4:
        # Recommendation phase
        st.header("💡 Step 4: Final Recommendation")
        render_constraints_summary(st.session_state.constraints)
        render_filtering_summary()
        render_scoring_summary()
        
        if (hasattr(st.session_state, 'model_scores') and 
            hasattr(st.session_state, 'trade_off_analysis') and
            st.session_state.model_scores and 
            st.session_state.constraints):
            render_recommendation_page(
                registry, 
                st.session_state.constraints, 
                st.session_state.model_scores,
                st.session_state.trade_off_analysis
            )
        else:
            st.error("❌ Missing scoring data. Please go back to scoring.")
            if st.button("⬅️ Back to Scoring"):
                st.session_state.step = 3
                st.rerun()
    
    else:
        # Other phases (placeholders)
        st.header(f"🚧 Step {st.session_state.step}: Coming Soon")
        render_constraints_summary(st.session_state.constraints)
        render_filtering_summary()
        render_scoring_summary()
        render_recommendation_summary()
        
        if st.button("⬅️ Back to Discovery"):
            st.session_state.step = 1
            st.rerun()

if __name__ == "__main__":
    main()