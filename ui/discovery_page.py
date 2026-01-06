"""
Discovery page UI components
"""

import streamlit as st
import json
from agents.discovery import DiscoveryAgent
from models.schemas import TaskType, LatencyCategory


def render_discovery_page():
    """Render the discovery/constraint extraction page"""
    st.header("🔍 Step 1: Use Case Discovery")
    st.markdown("Tell us about your LLM use case so we can find the best model for your needs.")
    
    # Initialize discovery agent
    if 'discovery_agent' not in st.session_state:
        st.session_state.discovery_agent = DiscoveryAgent()
    
    discovery_agent = st.session_state.discovery_agent
    
    # Main use case input
    st.subheader("📝 Describe Your Use Case")
    user_input = st.text_area(
        "Describe what you want to use the LLM for:",
        placeholder="e.g., I need to analyze customer feedback data to identify trends and generate summary reports for my team. The analysis needs to be accurate and I'll be processing about 1000 reviews per day.",
        height=100,
        help="Provide as much detail as possible about your intended use case, performance requirements, and constraints."
    )
    
    # Structured input form
    st.subheader("🎯 Specific Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Task type selection
        task_type_options = ["Auto-detect"] + [t.value for t in TaskType]
        selected_task_type = st.selectbox(
            "Task Type:",
            options=task_type_options,
            help="What type of task will the LLM primarily perform?"
        )
        
        # Latency requirements
        latency_options = ["Auto-detect"] + [l.value for l in LatencyCategory]
        selected_latency = st.selectbox(
            "Latency Requirements:",
            options=latency_options,
            help="How quickly do you need responses?"
        )
    
    with col2:
        # Budget constraints
        budget_input = st.text_input(
            "Budget Constraint:",
            placeholder="e.g., $100/month or budget-conscious",
            help="Monthly budget or cost sensitivity (budget/standard/premium)"
        )
        
        # Context window
        context_input = st.text_input(
            "Context Window Needs:",
            placeholder="e.g., 32k tokens or long documents",
            help="How much text do you need to process at once?"
        )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        st.markdown("**Priority Weights** (will be auto-inferred if not specified)")
        
        col3, col4 = st.columns(2)
        with col3:
            reasoning_weight = st.slider("Reasoning/Accuracy", 0.0, 1.0, 0.25, 0.05)
            latency_weight = st.slider("Speed/Latency", 0.0, 1.0, 0.25, 0.05)
        with col4:
            cost_weight = st.slider("Cost Efficiency", 0.0, 1.0, 0.25, 0.05)
            reliability_weight = st.slider("Reliability", 0.0, 1.0, 0.25, 0.05)
        
        # Normalize weights
        total_weight = reasoning_weight + latency_weight + cost_weight + reliability_weight
        if total_weight > 0:
            custom_weights = {
                "reasoning": reasoning_weight / total_weight,
                "latency": latency_weight / total_weight,
                "cost": cost_weight / total_weight,
                "reliability": reliability_weight / total_weight
            }
        else:
            custom_weights = None
    
    # Extract constraints button
    if st.button("🔍 Analyze Requirements", type="primary"):
        if not user_input.strip():
            st.error("Please provide a description of your use case.")
            return
        
        with st.spinner("Analyzing your requirements..."):
            # Extract constraints
            task_type_param = None if selected_task_type == "Auto-detect" else selected_task_type
            latency_param = None if selected_latency == "Auto-detect" else selected_latency
            
            constraints = discovery_agent.extract_constraints(
                user_input=user_input,
                use_case_type=task_type_param,
                budget_input=budget_input if budget_input.strip() else None,
                latency_input=latency_param,
                context_input=context_input if context_input.strip() else None
            )
            
            # Override with custom weights if provided
            if custom_weights:
                constraints.priority_weights = custom_weights
            
            # Store in session state
            st.session_state.constraints = constraints
            st.session_state.step = 2  # Move to next step
        
        st.success("✅ Requirements analyzed successfully!")
        st.rerun()
    
    # Display extracted constraints if available
    if hasattr(st.session_state, 'constraints') and st.session_state.constraints:
        st.subheader("📊 Extracted Constraints")
        
        constraints = st.session_state.constraints
        
        # Display in a nice format
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Task Information**")
            st.write(f"**Task Type:** {constraints.task_type.value if constraints.task_type else 'Not specified'}")
            st.write(f"**Latency Tolerance:** {constraints.latency_tolerance.value if constraints.latency_tolerance else 'Not specified'}")
            st.write(f"**Max Cost per Token:** ${constraints.max_cost_per_token:.6f}" if constraints.max_cost_per_token else "**Max Cost per Token:** Not specified")
            st.write(f"**Min Context Window:** {constraints.min_context_window:,} tokens" if constraints.min_context_window else "**Min Context Window:** Not specified")
        
        with col2:
            st.markdown("**Requirements**")
            if constraints.required_capabilities:
                st.write("**Required Capabilities:**")
                for capability in constraints.required_capabilities:
                    st.write(f"• {capability}")
            
            if constraints.deployment_preferences:
                st.write("**Deployment Preferences:**")
                for pref in constraints.deployment_preferences:
                    st.write(f"• {pref}")
        
        # Priority weights visualization
        st.markdown("**Priority Weights**")
        weights_df = {
            "Criterion": list(constraints.priority_weights.keys()),
            "Weight": [f"{v:.1%}" for v in constraints.priority_weights.values()]
        }
        st.bar_chart(constraints.priority_weights)
        
        # Show raw JSON for debugging
        with st.expander("🔧 Raw Constraints (JSON)"):
            st.json(constraints.model_dump())
        
        # Check if we need clarifying questions
        questions = discovery_agent.ask_clarifying_questions(constraints)
        if questions:
            st.warning("**Clarifying Questions:**")
            for question in questions:
                st.write(f"• {question}")
        
        # Navigation
        if discovery_agent.has_sufficient_information(constraints):
            if st.button("➡️ Proceed to Model Filtering", type="primary"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.error("Please provide more information to proceed.")


def render_constraints_summary(constraints):
    """Render a compact summary of constraints for other pages"""
    if not constraints:
        return
    
    with st.expander("📋 Current Constraints"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Task:** {constraints.task_type.value if constraints.task_type else 'N/A'}")
            st.write(f"**Latency:** {constraints.latency_tolerance.value if constraints.latency_tolerance else 'N/A'}")
        
        with col2:
            st.write(f"**Budget:** ${constraints.max_cost_per_token:.6f}" if constraints.max_cost_per_token else "**Budget:** N/A")
            st.write(f"**Context:** {constraints.min_context_window:,}" if constraints.min_context_window else "**Context:** N/A")
        
        with col3:
            st.write(f"**Capabilities:** {len(constraints.required_capabilities)}")
            top_priority = max(constraints.priority_weights.items(), key=lambda x: x[1])
            st.write(f"**Top Priority:** {top_priority[0]} ({top_priority[1]:.1%})")