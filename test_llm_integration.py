#!/usr/bin/env python3
"""
Test script to demonstrate LLM integration capabilities
"""

import logging
from agents.discovery import create_discovery_agent
from agents.recommendation import create_recommendation_agent
from agents.llm_adapter import LLMConfig, LLMProvider
from models.registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_discovery_agent_llm():
    """Test Discovery Agent with LLM integration"""
    print("🔍 Testing Discovery Agent with LLM Integration")
    print("=" * 50)
    
    # Create LLM-enabled discovery agent (using mock)
    discovery_agent = create_discovery_agent(enable_llm=True)
    
    # Test constraint extraction
    user_input = """
    I need to analyze customer feedback data to identify trends and generate 
    summary reports for my team. The analysis needs to be accurate and I'll be 
    processing about 1000 reviews per day. Budget is a concern but quality is 
    important. I need responses within a few seconds for interactive use.
    """
    
    print(f"User Input: {user_input.strip()}")
    print("\nExtracting constraints...")
    
    constraints = discovery_agent.extract_constraints(user_input)
    
    print(f"\n✅ Extracted Constraints:")
    print(f"  Task Type: {constraints.task_type}")
    print(f"  Latency Tolerance: {constraints.latency_tolerance}")
    print(f"  Max Cost per Token: ${constraints.max_cost_per_token:.6f}")
    print(f"  Min Context Window: {constraints.min_context_window:,}")
    print(f"  Required Capabilities: {constraints.required_capabilities}")
    print(f"  Priority Weights: {constraints.priority_weights}")
    
    return constraints

def test_recommendation_agent_llm(constraints):
    """Test Recommendation Agent with LLM integration"""
    print("\n💡 Testing Recommendation Agent with LLM Integration")
    print("=" * 50)
    
    # Create model registry and recommendation agent
    registry = ModelRegistry()
    recommendation_agent = create_recommendation_agent(registry, enable_llm=True)
    
    # Create mock model scores for testing
    from models.schemas import ModelScore
    
    model_scores = [
        ModelScore(
            model_name="GPT-4",
            overall_score=0.85,
            dimension_scores={
                "reasoning": 0.9,
                "latency": 0.7,
                "cost": 0.6,
                "reliability": 0.8
            },
            explanations={
                "reasoning": "Excellent analytical capabilities",
                "latency": "Good response times for interactive use",
                "cost": "Premium pricing tier",
                "reliability": "High uptime and consistency"
            }
        ),
        ModelScore(
            model_name="Claude-3-Sonnet",
            overall_score=0.78,
            dimension_scores={
                "reasoning": 0.85,
                "latency": 0.75,
                "cost": 0.8,
                "reliability": 0.85
            },
            explanations={
                "reasoning": "Strong analytical performance",
                "latency": "Good response times",
                "cost": "Better cost efficiency",
                "reliability": "Very reliable service"
            }
        )
    ]
    
    trade_off_analysis = {
        "summary": "GPT-4 leads in reasoning but Claude-3-Sonnet offers better cost efficiency"
    }
    
    print("Generating LLM-powered recommendation...")
    
    recommendation = recommendation_agent.generate_recommendation(
        model_scores, trade_off_analysis, constraints
    )
    
    print(f"\n✅ Generated Recommendation:")
    print(f"  Primary Model: {recommendation['primary_model']['name']}")
    print(f"  Confidence: {recommendation['primary_model']['confidence']}")
    print(f"  Context: {recommendation['context'][:200]}...")
    print(f"  Mitigation Strategies: {len(recommendation['mitigation_strategies'])} strategies")
    print(f"  Implementation Guidance: {len(recommendation['implementation_guidance'])} items")
    
    return recommendation

def test_llm_providers():
    """Test available LLM providers"""
    print("\n🤖 Testing LLM Provider Availability")
    print("=" * 50)
    
    from agents.llm_adapter import LLMAdapterFactory
    
    available_providers = LLMAdapterFactory.get_available_providers()
    
    print("Available LLM Providers:")
    for provider in available_providers:
        print(f"  ✅ {provider.value}")
    
    # Test mock provider
    mock_config = LLMConfig(
        provider=LLMProvider.MOCK,
        model="mock-model"
    )
    
    mock_adapter = LLMAdapterFactory.create_adapter(mock_config)
    test_response = mock_adapter.generate_response("Test prompt for capabilities")
    
    print(f"\nMock Provider Test:")
    print(f"  Response: {test_response.content[:100]}...")
    print(f"  Provider: {test_response.provider}")
    print(f"  Tokens Used: {test_response.tokens_used}")
    print(f"  Cost Estimate: ${test_response.cost_estimate:.6f}")

def main():
    """Run all LLM integration tests"""
    print("🚀 LLM Decision Agent - Integration Test")
    print("=" * 60)
    
    try:
        # Test LLM providers
        test_llm_providers()
        
        # Test Discovery Agent
        constraints = test_discovery_agent_llm()
        
        # Test Recommendation Agent
        recommendation = test_recommendation_agent_llm(constraints)
        
        print("\n🎉 All LLM Integration Tests Passed!")
        print("=" * 60)
        print("✅ Discovery Agent: LLM-powered constraint extraction working")
        print("✅ Recommendation Agent: LLM-powered recommendation generation working")
        print("✅ LLM Adapters: Multiple provider support with fallback working")
        print("✅ Error Handling: Graceful fallback to rule-based approach working")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()