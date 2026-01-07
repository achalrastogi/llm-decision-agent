"""
Recommendation Agent - Generates explainable recommendations with trade-offs and mitigation strategies
Enhanced with LLM integration for dynamic recommendation generation
"""

import logging
from typing import Dict, List, Optional, Tuple
from models.schemas import UserConstraints, ModelScore
from models.registry import ModelRegistry
from agents.llm_adapter import LLMManager, LLMConfig


class EnhancedRecommendationAgent:
    """
    Converts analysis into human-readable advice with explicit trade-offs,
    mitigation strategies, and future-proofing guidance.
    """
    
    def __init__(self, registry: ModelRegistry, llm_manager: Optional[LLMManager] = None, use_llm: bool = True):
        """
        Initialize Recommendation Agent with optional LLM integration
        
        Args:
            registry: Model registry for model information
            llm_manager: LLM manager for intelligent recommendation generation
            use_llm: Whether to use LLM or fall back to rule-based approach
        """
        self.registry = registry
        self.llm_manager = llm_manager
        self.use_llm = use_llm and llm_manager is not None
        self.logger = logging.getLogger("recommendation_agent")
        
        # Confidence thresholds for recommendations
        self.confidence_thresholds = {
            'high': 0.15,      # Score gap > 0.15 = high confidence
            'medium': 0.05,    # Score gap > 0.05 = medium confidence
            'low': 0.0         # Any gap = low confidence
        }
        
        # Mitigation strategies by weakness type
        self.mitigation_strategies = {
            'reasoning': [
                "Consider using prompt engineering techniques to improve output quality",
                "Implement multi-step reasoning workflows to compensate for lower cognitive ability",
                "Use ensemble approaches combining this model with stronger reasoning models",
                "Add human review for complex analytical tasks"
            ],
            'latency': [
                "Implement caching strategies to reduce repeated processing",
                "Use asynchronous processing for non-critical tasks",
                "Consider batch processing to improve throughput",
                "Optimize prompts to reduce token count and processing time"
            ],
            'cost': [
                "Implement intelligent prompt optimization to reduce token usage",
                "Use tiered processing (cheaper models for simple tasks, premium for complex)",
                "Implement result caching to avoid redundant API calls",
                "Consider fine-tuning to improve efficiency on specific tasks"
            ],
            'reliability': [
                "Implement robust error handling and retry mechanisms",
                "Use multiple API endpoints or providers for redundancy",
                "Add monitoring and alerting for service availability",
                "Consider hybrid approaches with backup models"
            ]
        }
    
    def generate_recommendation(self, model_scores: List[ModelScore], 
                              trade_off_analysis: Dict,
                              constraints: UserConstraints) -> Dict:
        """
        Generate comprehensive recommendation with context, trade-offs, and mitigation.
        Uses LLM for intelligent recommendation generation with rule-based fallback.
        
        Args:
            model_scores: Scored and ranked models
            trade_off_analysis: Trade-off analysis from Decision Engine
            constraints: User constraints and priorities
            
        Returns:
            Dictionary with complete recommendation
        """
        if not model_scores:
            return self._generate_no_models_recommendation()
        
        if self.use_llm:
            try:
                return self._generate_recommendation_with_llm(
                    model_scores, trade_off_analysis, constraints
                )
            except Exception as e:
                self.logger.warning(f"LLM recommendation generation failed, falling back to rule-based: {e}")
        
        # Fallback to rule-based recommendation
        return self._generate_recommendation_rule_based(
            model_scores, trade_off_analysis, constraints
        )
    
    def _generate_recommendation_with_llm(self, model_scores: List[ModelScore], 
                                        trade_off_analysis: Dict,
                                        constraints: UserConstraints) -> Dict:
        """Generate recommendation using LLM for intelligent analysis"""
        
        # Build comprehensive prompt with all context
        prompt = self._build_recommendation_prompt(
            model_scores, trade_off_analysis, constraints
        )
        
        # Get LLM response
        response = self.llm_manager.generate_response(
            prompt, max_tokens=1500, temperature=0.4
        )
        
        # Parse LLM response and combine with structured data
        try:
            llm_recommendation = self._parse_llm_recommendation(response.content)
            
            # Enhance with structured data
            return self._enhance_llm_recommendation(
                llm_recommendation, model_scores, trade_off_analysis, constraints
            )
        except Exception as e:
            self.logger.error(f"Failed to parse LLM recommendation: {e}")
            # Fall back to rule-based if parsing fails
            return self._generate_recommendation_rule_based(
                model_scores, trade_off_analysis, constraints
            )
    
    def _build_recommendation_prompt(self, model_scores: List[ModelScore], 
                                   trade_off_analysis: Dict,
                                   constraints: UserConstraints) -> str:
        """Build comprehensive prompt for LLM recommendation generation"""
        
        # Get model details
        model_details = {}
        for score in model_scores[:3]:  # Top 3 models
            for model_id, model_info in self.registry.get_all_models().items():
                if model_info.name == score.model_name:
                    model_details[score.model_name] = model_info
                    break
        
        prompt = f"""You are an expert AI consultant providing LLM model recommendations. 
Analyze the following data and provide a comprehensive, personalized recommendation.

## User Requirements:
- Task Type: {constraints.task_type.value if constraints.task_type else 'Not specified'}
- Latency Tolerance: {constraints.latency_tolerance.value if constraints.latency_tolerance else 'Not specified'}
- Budget Constraint: ${constraints.max_cost_per_token:.6f} per token
- Context Window: {constraints.min_context_window:,} tokens minimum
- Required Capabilities: {', '.join(constraints.required_capabilities)}
- Priority Weights: {', '.join([f'{k}: {v:.1%}' for k, v in constraints.priority_weights.items()])}

## Top Model Candidates:
"""
        
        # Add model details
        for i, score in enumerate(model_scores[:3], 1):
            model_info = model_details.get(score.model_name)
            prompt += f"""
### {i}. {score.model_name} (Score: {score.overall_score:.3f})
- Provider: {model_info.provider if model_info else 'Unknown'}
- Context Window: {model_info.context_window:,} tokens
- Latency: {model_info.latency_category.value}
- Cost: Input ${model_info.cost_per_token['input']:.6f}, Output ${model_info.cost_per_token['output']:.6f}
- Dimension Scores: {', '.join([f'{k}: {v:.3f}' for k, v in score.dimension_scores.items()])}
- Capabilities: {', '.join(model_info.capabilities) if model_info else 'Not specified'}
"""
        
        prompt += f"""
## Trade-off Analysis:
{trade_off_analysis.get('summary', 'No specific trade-offs identified')}

## Instructions:
Provide a comprehensive recommendation that includes:

1. **Primary Recommendation**: Which model to choose and why
2. **Context & Rationale**: Detailed explanation of why this model fits the user's needs
3. **Key Trade-offs**: What the user gains and potentially gives up with this choice
4. **Mitigation Strategies**: Specific, actionable strategies to address any weaknesses
5. **Implementation Guidance**: Practical next steps for deployment
6. **Future-Proofing**: Advice for scaling and adapting over time

Make your recommendation:
- Specific to the user's stated requirements and priorities
- Honest about trade-offs and limitations
- Actionable with concrete next steps
- Forward-looking for long-term success

Write in a professional but accessible tone, as if advising a technical decision-maker.
"""
        
        return prompt
    
    def _parse_llm_recommendation(self, response_content: str) -> Dict:
        """Parse LLM response into structured recommendation components"""
        # Simple parsing - look for key sections
        sections = {
            'context': '',
            'trade_offs': '',
            'mitigation_strategies': '',
            'implementation_guidance': '',
            'future_proofing': '',
            'decision_rationale': ''
        }
        
        # Extract content (simplified parsing)
        lines = response_content.split('\n')
        current_section = 'context'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            line_lower = line.lower()
            if 'trade-off' in line_lower or 'tradeoff' in line_lower:
                current_section = 'trade_offs'
            elif 'mitigation' in line_lower or 'strategy' in line_lower:
                current_section = 'mitigation_strategies'
            elif 'implementation' in line_lower or 'deployment' in line_lower:
                current_section = 'implementation_guidance'
            elif 'future' in line_lower or 'scaling' in line_lower:
                current_section = 'future_proofing'
            elif 'rationale' in line_lower or 'decision' in line_lower:
                current_section = 'decision_rationale'
            else:
                # Add content to current section
                if sections[current_section]:
                    sections[current_section] += ' '
                sections[current_section] += line
        
        # If no clear sections found, put everything in context
        if not any(sections.values()):
            sections['context'] = response_content
            sections['decision_rationale'] = "LLM-generated recommendation based on comprehensive analysis."
        
        return sections
    
    def _enhance_llm_recommendation(self, llm_recommendation: Dict, 
                                  model_scores: List[ModelScore],
                                  trade_off_analysis: Dict,
                                  constraints: UserConstraints) -> Dict:
        """Enhance LLM recommendation with structured data"""
        
        primary_recommendation = model_scores[0]
        confidence = self._assess_confidence(model_scores, trade_off_analysis)
        
        return {
            'primary_model': {
                'name': primary_recommendation.model_name,
                'score': primary_recommendation.overall_score,
                'confidence': confidence['level'],
                'confidence_reason': confidence['reason']
            },
            'context': llm_recommendation.get('context', 'LLM-generated recommendation'),
            'trade_offs': self._parse_trade_offs_from_llm(llm_recommendation.get('trade_offs', '')),
            'mitigation_strategies': self._parse_strategies_from_llm(
                llm_recommendation.get('mitigation_strategies', '')
            ),
            'future_proofing': self._parse_guidance_from_llm(
                llm_recommendation.get('future_proofing', '')
            ),
            'implementation_guidance': self._parse_guidance_from_llm(
                llm_recommendation.get('implementation_guidance', '')
            ),
            'alternatives': self._generate_alternatives_summary(model_scores),
            'decision_rationale': llm_recommendation.get('decision_rationale', 
                                                       'Comprehensive analysis of requirements and model capabilities.')
        }
    
    def _parse_trade_offs_from_llm(self, trade_offs_text: str) -> List[Dict]:
        """Parse trade-offs from LLM text into structured format"""
        if not trade_offs_text:
            return []
        
        # Simple parsing - split by sentences and create trade-off objects
        sentences = [s.strip() for s in trade_offs_text.split('.') if s.strip()]
        trade_offs = []
        
        for sentence in sentences[:3]:  # Limit to 3 trade-offs
            if len(sentence) > 20:  # Filter out very short fragments
                trade_offs.append({
                    'alternative_model': 'Alternative option',
                    'score_difference': 0.0,
                    'trade_off_summary': sentence,
                    'when_to_consider': 'Consider based on specific requirements'
                })
        
        return trade_offs
    
    def _parse_strategies_from_llm(self, strategies_text: str) -> List[str]:
        """Parse mitigation strategies from LLM text"""
        if not strategies_text:
            return ["Implement monitoring and optimization practices for best performance."]
        
        # Split by bullet points, numbers, or sentences
        strategies = []
        
        # Look for bullet points or numbered lists
        lines = strategies_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or 
                        line.startswith('*') or any(line.startswith(f'{i}.') for i in range(1, 10))):
                # Clean up formatting
                clean_line = line.lstrip('-•*0123456789. ')
                if len(clean_line) > 10:
                    strategies.append(clean_line)
        
        # If no bullet points found, split by sentences
        if not strategies:
            sentences = [s.strip() for s in strategies_text.split('.') if s.strip()]
            strategies = [s for s in sentences if len(s) > 20][:4]  # Limit to 4
        
        return strategies if strategies else ["Implement best practices for optimal performance."]
    
    def _parse_guidance_from_llm(self, guidance_text: str) -> List[str]:
        """Parse guidance from LLM text into list format"""
        if not guidance_text:
            return []
        
        return self._parse_strategies_from_llm(guidance_text)  # Same parsing logic
    
    def _generate_recommendation_rule_based(self, model_scores: List[ModelScore], 
                                          trade_off_analysis: Dict,
                                          constraints: UserConstraints) -> Dict:
        """
        Rule-based recommendation generation (original implementation)
        Used as fallback when LLM is unavailable
        """
        primary_recommendation = model_scores[0]
        
        # Determine confidence level
        confidence = self._assess_confidence(model_scores, trade_off_analysis)
        
        # Generate recommendation components
        recommendation = {
            'primary_model': {
                'name': primary_recommendation.model_name,
                'score': primary_recommendation.overall_score,
                'confidence': confidence['level'],
                'confidence_reason': confidence['reason']
            },
            'context': self._generate_context(primary_recommendation, constraints),
            'trade_offs': self._generate_trade_offs(model_scores, trade_off_analysis),
            'mitigation_strategies': self._generate_mitigation_strategies(
                primary_recommendation, model_scores
            ),
            'future_proofing': self._generate_future_proofing_guidance(
                model_scores, constraints
            ),
            'implementation_guidance': self._generate_implementation_guidance(
                primary_recommendation, constraints
            ),
            'alternatives': self._generate_alternatives_summary(model_scores),
            'decision_rationale': self._generate_decision_rationale(
                primary_recommendation, constraints, confidence
            )
        }
        
        # Validate recommendation has supporting context
        self._validate_recommendation_context(recommendation)
        
        return recommendation
    
    def _assess_confidence(self, model_scores: List[ModelScore], 
                          trade_off_analysis: Dict) -> Dict:
        """Assess confidence level in the primary recommendation"""
        if len(model_scores) < 2:
            return {
                'level': 'high',
                'reason': 'Only one viable model available'
            }
        
        # Calculate score gap between top two models
        score_gap = model_scores[0].overall_score - model_scores[1].overall_score
        
        if score_gap >= self.confidence_thresholds['high']:
            return {
                'level': 'high',
                'reason': f'Clear leader with {score_gap:.3f} score advantage'
            }
        elif score_gap >= self.confidence_thresholds['medium']:
            return {
                'level': 'medium',
                'reason': f'Moderate advantage with {score_gap:.3f} score gap'
            }
        else:
            return {
                'level': 'low',
                'reason': f'Very close competition with only {score_gap:.3f} score difference'
            }
    
    def _generate_context(self, primary_model: ModelScore, 
                         constraints: UserConstraints) -> str:
        """Generate contextual explanation for the recommendation"""
        # Get model details
        model_info = None
        for model_id, info in self.registry.get_all_models().items():
            if info.name == primary_model.model_name:
                model_info = info
                break
        
        if not model_info:
            return f"Recommended {primary_model.model_name} based on scoring analysis."
        
        context_parts = []
        
        # Primary strengths
        top_dimensions = sorted(
            primary_model.dimension_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        strengths = [dim.replace('_', ' ') for dim, _ in top_dimensions]
        context_parts.append(
            f"**{primary_model.model_name}** is recommended primarily for its "
            f"strong {' and '.join(strengths)} performance."
        )
        
        # Task alignment
        if constraints.task_type:
            task_alignment = self._assess_task_alignment(model_info, constraints.task_type)
            context_parts.append(task_alignment)
        
        # Key specifications
        context_parts.append(
            f"This model offers {model_info.context_window:,} tokens of context, "
            f"{model_info.latency_category.value} latency, and "
            f"{model_info.cost_tier.value}-tier pricing."
        )
        
        return " ".join(context_parts)
    
    def _assess_task_alignment(self, model_info, task_type) -> str:
        """Assess how well the model aligns with the task type"""
        alignments = {
            'analytical': "Its strong reasoning capabilities make it well-suited for analytical tasks.",
            'generative': "Its balanced performance across dimensions supports diverse generative use cases.",
            'agentic': "Its reliability and tool integration capabilities support autonomous workflows."
        }
        
        return alignments.get(task_type.value, "It provides balanced capabilities for your use case.")
    
    def _generate_trade_offs(self, model_scores: List[ModelScore], 
                           trade_off_analysis: Dict) -> List[Dict]:
        """Generate explicit trade-offs between top options"""
        if len(model_scores) < 2:
            return []
        
        trade_offs = []
        primary_model = model_scores[0]
        
        for alternative in model_scores[1:3]:  # Top 2 alternatives
            trade_off = {
                'alternative_model': alternative.model_name,
                'score_difference': primary_model.overall_score - alternative.overall_score,
                'trade_off_summary': self._summarize_trade_off(primary_model, alternative),
                'when_to_consider': self._generate_when_to_consider(primary_model, alternative)
            }
            trade_offs.append(trade_off)
        
        return trade_offs
    
    def _summarize_trade_off(self, primary: ModelScore, alternative: ModelScore) -> str:
        """Summarize the key trade-off between two models"""
        # Find the dimension where alternative is strongest relative to primary
        best_alternative_dim = None
        best_advantage = 0
        
        for dim in primary.dimension_scores.keys():
            advantage = alternative.dimension_scores.get(dim, 0) - primary.dimension_scores.get(dim, 0)
            if advantage > best_advantage:
                best_advantage = advantage
                best_alternative_dim = dim
        
        if best_alternative_dim and best_advantage > 0.1:
            return (
                f"Trade {primary.model_name}'s overall advantage for "
                f"{alternative.model_name}'s superior {best_alternative_dim.replace('_', ' ')} performance."
            )
        else:
            return f"{alternative.model_name} offers similar capabilities with minor trade-offs."
    
    def _generate_when_to_consider(self, primary: ModelScore, alternative: ModelScore) -> str:
        """Generate guidance on when to consider the alternative"""
        # Find alternative's strongest dimension
        alt_strengths = sorted(
            alternative.dimension_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if alt_strengths:
            strongest_dim = alt_strengths[0][0].replace('_', ' ')
            return f"Consider if {strongest_dim} is your absolute top priority."
        
        return "Consider for specific use cases where its characteristics align better."
    
    def _generate_mitigation_strategies(self, primary_model: ModelScore, 
                                      all_models: List[ModelScore]) -> List[str]:
        """Generate mitigation strategies for identified weaknesses"""
        strategies = []
        
        # Identify weaknesses (dimensions where score is below 0.5)
        weaknesses = []
        for dim, score in primary_model.dimension_scores.items():
            if score < 0.5:
                weaknesses.append(dim)
        
        # Generate strategies for each weakness
        for weakness in weaknesses:
            if weakness in self.mitigation_strategies:
                # Pick the most relevant strategy
                strategy = self.mitigation_strategies[weakness][0]
                strategies.append(f"**{weakness.replace('_', ' ').title()}:** {strategy}")
        
        # Add general strategies if no specific weaknesses
        if not strategies:
            strategies.append(
                "**Optimization:** Implement monitoring and optimization practices "
                "to maximize the model's performance in your specific use case."
            )
        
        return strategies
    
    def _generate_future_proofing_guidance(self, model_scores: List[ModelScore], 
                                         constraints: UserConstraints) -> List[str]:
        """Generate future-proofing guidance including multi-model approaches"""
        guidance = []
        
        # Multi-model routing recommendation
        if len(model_scores) >= 2:
            top_two = model_scores[:2]
            guidance.append(
                f"**Multi-Model Routing:** Consider implementing intelligent routing "
                f"between {top_two[0].model_name} and {top_two[1].model_name} "
                f"based on task complexity or requirements."
            )
        
        # Scaling considerations
        if constraints.priority_weights.get('cost', 0) > 0.3:
            guidance.append(
                "**Cost Scaling:** Plan for tiered model usage as volume grows - "
                "use premium models for complex tasks, budget models for simple ones."
            )
        
        # Technology evolution
        guidance.append(
            "**Technology Evolution:** Regularly reassess model options as "
            "new models are released and pricing changes."
        )
        
        # Vendor diversification
        providers = set()
        for score in model_scores[:3]:
            for model_id, model_info in self.registry.get_all_models().items():
                if model_info.name == score.model_name:
                    providers.add(model_info.provider)
        
        if len(providers) > 1:
            guidance.append(
                "**Vendor Diversification:** Consider maintaining relationships "
                "with multiple providers to reduce dependency risk."
            )
        
        return guidance
    
    def _generate_implementation_guidance(self, primary_model: ModelScore, 
                                        constraints: UserConstraints) -> List[str]:
        """Generate practical implementation guidance"""
        guidance = []
        
        # Get model info
        model_info = None
        for model_id, info in self.registry.get_all_models().items():
            if info.name == primary_model.model_name:
                model_info = info
                break
        
        if not model_info:
            return guidance
        
        # Deployment guidance
        if 'api' in model_info.deployment_options:
            guidance.append(
                "**API Integration:** Start with API integration for fastest deployment "
                "and lowest infrastructure overhead."
            )
        
        # Context window optimization
        if model_info.context_window > 50000:
            guidance.append(
                "**Context Management:** Leverage the large context window for "
                "document analysis and long conversation maintenance."
            )
        elif model_info.context_window < 10000:
            guidance.append(
                "**Context Optimization:** Implement efficient prompt design "
                "to work within the context window limitations."
            )
        
        # Cost management
        if model_info.cost_tier.value == 'premium':
            guidance.append(
                "**Cost Management:** Implement usage monitoring and optimization "
                "strategies given the premium pricing tier."
            )
        
        return guidance
    
    def _generate_alternatives_summary(self, model_scores: List[ModelScore]) -> List[Dict]:
        """Generate summary of alternative options"""
        if len(model_scores) <= 1:
            return []
        
        alternatives = []
        for score in model_scores[1:4]:  # Top 3 alternatives
            alternatives.append({
                'name': score.model_name,
                'score': score.overall_score,
                'key_strength': self._identify_key_strength(score),
                'use_case': self._suggest_use_case(score)
            })
        
        return alternatives
    
    def _identify_key_strength(self, model_score: ModelScore) -> str:
        """Identify the model's key strength"""
        top_dimension = max(
            model_score.dimension_scores.items(),
            key=lambda x: x[1]
        )
        return top_dimension[0].replace('_', ' ').title()
    
    def _suggest_use_case(self, model_score: ModelScore) -> str:
        """Suggest when this model might be preferred"""
        top_dimension = max(
            model_score.dimension_scores.items(),
            key=lambda x: x[1]
        )[0]
        
        use_cases = {
            'reasoning': 'Complex analytical tasks requiring deep thinking',
            'latency': 'Real-time applications requiring immediate responses',
            'cost': 'High-volume applications with budget constraints',
            'reliability': 'Production systems requiring maximum uptime'
        }
        
        return use_cases.get(top_dimension, 'Specialized use cases')
    
    def _generate_decision_rationale(self, primary_model: ModelScore, 
                                   constraints: UserConstraints, 
                                   confidence: Dict) -> str:
        """Generate the overall decision rationale"""
        rationale_parts = []
        
        # Score-based rationale
        rationale_parts.append(
            f"{primary_model.model_name} achieved the highest overall score "
            f"({primary_model.overall_score:.3f}) based on your priority weights."
        )
        
        # Priority alignment
        top_priority = max(constraints.priority_weights.items(), key=lambda x: x[1])
        top_dimension_score = primary_model.dimension_scores.get(top_priority[0], 0)
        
        rationale_parts.append(
            f"It aligns well with your top priority ({top_priority[0].replace('_', ' ')}: "
            f"{top_priority[1]:.1%} weight) with a {top_dimension_score:.3f} score in this area."
        )
        
        # Confidence rationale
        rationale_parts.append(confidence['reason'] + ".")
        
        return " ".join(rationale_parts)
    
    def _validate_recommendation_context(self, recommendation: Dict) -> None:
        """Validate that recommendation includes supporting context"""
        required_fields = ['context', 'decision_rationale']
        
        for field in required_fields:
            if not recommendation.get(field):
                raise ValueError(f"Recommendation missing required context: {field}")
        
        # Ensure we never have bare assertions
        context = recommendation['context']
        rationale = recommendation['decision_rationale']
        
        if len(context) < 50 or len(rationale) < 50:
            raise ValueError("Recommendation context insufficient - must provide detailed reasoning")
    
    def _generate_no_models_recommendation(self) -> Dict:
        """Generate recommendation when no models are available"""
        return {
            'primary_model': None,
            'context': "No models meet your specified constraints.",
            'trade_offs': [],
            'mitigation_strategies': [
                "**Constraint Relaxation:** Consider relaxing budget, latency, or context window requirements",
                "**Alternative Approaches:** Explore different task decomposition or workflow strategies",
                "**Future Options:** Monitor for new model releases that might meet your requirements"
            ],
            'future_proofing': [
                "**Requirement Review:** Regularly reassess whether your constraints remain necessary",
                "**Market Monitoring:** Stay informed about new model releases and pricing changes"
            ],
            'implementation_guidance': [
                "**Constraint Analysis:** Review which constraints are eliminating models",
                "**Phased Approach:** Consider implementing with relaxed constraints initially"
            ],
            'alternatives': [],
            'decision_rationale': "No viable models found matching the specified constraints. Consider adjusting requirements or exploring alternative approaches."
        }

# Backward compatibility wrapper
class RecommendationAgent(EnhancedRecommendationAgent):
    """
    Backward compatibility wrapper for the original RecommendationAgent.
    Maintains the same interface while providing enhanced LLM capabilities.
    """
    
    def __init__(self, registry: ModelRegistry, llm_manager: Optional[LLMManager] = None):
        """Initialize with optional LLM manager for enhanced capabilities"""
        super().__init__(registry=registry, llm_manager=llm_manager, use_llm=llm_manager is not None)


def create_recommendation_agent(registry: ModelRegistry, enable_llm: bool = True, 
                              llm_config: Optional[LLMConfig] = None) -> RecommendationAgent:
    """
    Factory function to create Recommendation Agent with optional LLM integration
    
    Args:
        registry: Model registry for model information
        enable_llm: Whether to enable LLM capabilities
        llm_config: LLM configuration (uses mock if None and enable_llm=True)
    
    Returns:
        RecommendationAgent instance with appropriate configuration
    """
    if not enable_llm:
        return RecommendationAgent(registry)
    
    try:
        from agents.llm_adapter import LLMManager, LLMProvider
        
        # Use provided config or default to mock for testing
        if llm_config is None:
            llm_config = LLMConfig(
                provider=LLMProvider.MOCK,
                model="mock-model",
                max_tokens=1500,
                temperature=0.4
            )
        
        # Create LLM manager with fallback to mock
        fallback_config = LLMConfig(
            provider=LLMProvider.MOCK,
            model="mock-model",
            max_tokens=1500,
            temperature=0.4
        )
        
        llm_manager = LLMManager(
            primary_config=llm_config,
            fallback_configs=[fallback_config] if llm_config.provider != LLMProvider.MOCK else None
        )
        
        return RecommendationAgent(registry, llm_manager=llm_manager)
        
    except Exception as e:
        logging.warning(f"Failed to create LLM-enabled Recommendation Agent: {e}")
        return RecommendationAgent(registry)  # Fall back to rule-based only