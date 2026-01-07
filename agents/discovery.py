"""
Discovery Agent - Extracts user constraints and requirements
Enhanced with LLM integration for intelligent constraint extraction
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from models.schemas import UserConstraints, TaskType, LatencyCategory
from agents.llm_adapter import LLMManager, LLMConfig, LLMProvider


class EnhancedDiscoveryAgent:
    """
    Extracts user intent, constraints, and priorities from input.
    Uses LLM for intelligent understanding with rule-based fallback.
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None, use_llm: bool = True):
        """
        Initialize Discovery Agent with optional LLM integration
        
        Args:
            llm_manager: LLM manager for intelligent extraction
            use_llm: Whether to use LLM or fall back to rule-based approach
        """
        self.llm_manager = llm_manager
        self.use_llm = use_llm and llm_manager is not None
        self.logger = logging.getLogger("discovery_agent")
        
        # Rule-based fallback keywords (same as original)
        self.task_type_keywords = {
            TaskType.ANALYTICAL: [
                "analyze", "analysis", "research", "investigate", "examine", 
                "study", "evaluate", "assess", "review", "compare", "summarize"
            ],
            TaskType.GENERATIVE: [
                "generate", "create", "write", "compose", "draft", "produce",
                "build", "make", "design", "craft", "author", "content"
            ],
            TaskType.AGENTIC: [
                "automate", "agent", "autonomous", "workflow", "process",
                "execute", "perform", "handle", "manage", "orchestrate", "tool"
            ]
        }
        
        self.latency_keywords = {
            LatencyCategory.REAL_TIME: [
                "real-time", "instant", "immediate", "live", "streaming",
                "milliseconds", "ms", "fast", "quick", "responsive"
            ],
            LatencyCategory.INTERACTIVE: [
                "interactive", "chat", "conversation", "dialogue", "seconds",
                "responsive", "user-facing", "interface", "ui"
            ],
            LatencyCategory.BATCH: [
                "batch", "bulk", "background", "offline", "scheduled",
                "minutes", "hours", "processing", "large-scale"
            ]
        }
        
        self.capability_keywords = {
            "text_generation": ["text", "writing", "content", "generate"],
            "code_generation": ["code", "programming", "development", "script"],
            "analysis": ["analyze", "analysis", "research", "study"],
            "tool_calling": ["tool", "function", "api", "integration"],
            "multimodal": ["image", "vision", "multimodal", "visual"]
        }
    
    def extract_constraints(self, user_input: str, use_case_type: str = None, 
                          budget_input: str = None, latency_input: str = None,
                          context_input: str = None) -> UserConstraints:
        """
        Extract constraints from user input and form data.
        Uses LLM for intelligent extraction with rule-based fallback.
        
        Args:
            user_input: Free-form description of the use case
            use_case_type: Selected task type from form
            budget_input: Budget constraint input
            latency_input: Latency requirement input
            context_input: Context window requirement
        
        Returns:
            UserConstraints object with extracted information
        """
        if self.use_llm:
            try:
                return self._extract_constraints_with_llm(
                    user_input, use_case_type, budget_input, latency_input, context_input
                )
            except Exception as e:
                self.logger.warning(f"LLM extraction failed, falling back to rule-based: {e}")
        
        # Fallback to rule-based extraction
        return self._extract_constraints_rule_based(
            user_input, use_case_type, budget_input, latency_input, context_input
        )
    
    def _extract_constraints_with_llm(self, user_input: str, use_case_type: str = None, 
                                    budget_input: str = None, latency_input: str = None,
                                    context_input: str = None) -> UserConstraints:
        """Extract constraints using LLM for intelligent understanding"""
        
        # Construct prompt for constraint extraction
        prompt = self._build_extraction_prompt(
            user_input, use_case_type, budget_input, latency_input, context_input
        )
        
        # Get LLM response
        response = self.llm_manager.generate_response(prompt, max_tokens=800, temperature=0.3)
        
        # Parse LLM response into structured constraints
        try:
            constraints_data = self._parse_llm_response(response.content)
            return self._build_constraints_from_llm_data(constraints_data)
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Fall back to rule-based if parsing fails
            return self._extract_constraints_rule_based(
                user_input, use_case_type, budget_input, latency_input, context_input
            )
    
    def _build_extraction_prompt(self, user_input: str, use_case_type: str = None, 
                               budget_input: str = None, latency_input: str = None,
                               context_input: str = None) -> str:
        """Build prompt for LLM constraint extraction"""
        
        prompt = f"""You are an expert AI consultant helping users choose the right LLM for their needs. 
Analyze the following user input and extract structured constraints and requirements.

User Description: "{user_input}"
"""
        
        if use_case_type:
            prompt += f"Explicit Task Type: {use_case_type}\n"
        if budget_input:
            prompt += f"Budget Input: {budget_input}\n"
        if latency_input:
            prompt += f"Latency Input: {latency_input}\n"
        if context_input:
            prompt += f"Context Window Input: {context_input}\n"
        
        prompt += """
Please extract and return a JSON object with the following structure:
{
    "task_type": "analytical|generative|agentic",
    "latency_tolerance": "real-time|interactive|batch",
    "budget_sensitivity": "low|medium|high",
    "max_cost_per_token": 0.00001,
    "min_context_window": 8000,
    "required_capabilities": ["text_generation", "analysis"],
    "deployment_preferences": ["api", "local"],
    "priority_weights": {
        "reasoning": 0.3,
        "latency": 0.25,
        "cost": 0.25,
        "reliability": 0.2
    },
    "confidence": 0.8,
    "reasoning": "Brief explanation of the analysis"
}

Guidelines:
- task_type: analytical (research, analysis), generative (content creation), agentic (automation, tools)
- latency_tolerance: real-time (<100ms), interactive (<5s), batch (minutes/hours)
- budget_sensitivity: low (premium OK), medium (balanced), high (cost-conscious)
- max_cost_per_token: estimated acceptable cost per token in USD
- min_context_window: minimum tokens needed for context
- required_capabilities: list from [text_generation, code_generation, analysis, tool_calling, multimodal]
- deployment_preferences: list from [api, local, azure]
- priority_weights: importance weights (must sum to 1.0)
- confidence: your confidence in the extraction (0.0-1.0)
- reasoning: brief explanation of your analysis

Return only the JSON object, no additional text.
"""
        
        return prompt
    
    def _parse_llm_response(self, response_content: str) -> Dict:
        """Parse LLM response into structured data"""
        # Clean up response to extract JSON
        response_clean = response_content.strip()
        
        # Find JSON object in response
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in LLM response")
        
        json_str = response_clean[start_idx:end_idx]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"Response content: {response_content}")
            raise
    
    def _build_constraints_from_llm_data(self, data: Dict) -> UserConstraints:
        """Build UserConstraints object from LLM-extracted data"""
        constraints = UserConstraints()
        
        # Map task type
        if data.get("task_type"):
            try:
                constraints.task_type = TaskType(data["task_type"].lower())
            except ValueError:
                constraints.task_type = TaskType.GENERATIVE
        
        # Map latency tolerance
        if data.get("latency_tolerance"):
            try:
                latency_str = data["latency_tolerance"].lower().replace("-", "_")
                constraints.latency_tolerance = LatencyCategory(latency_str)
            except ValueError:
                constraints.latency_tolerance = LatencyCategory.INTERACTIVE
        
        # Set budget constraints
        constraints.max_cost_per_token = data.get("max_cost_per_token", 0.00005)
        
        # Set context window
        constraints.min_context_window = data.get("min_context_window", 8000)
        
        # Set capabilities
        constraints.required_capabilities = data.get("required_capabilities", ["text_generation"])
        
        # Set deployment preferences
        constraints.deployment_preferences = data.get("deployment_preferences", ["api"])
        
        # Set priority weights
        weights = data.get("priority_weights", {})
        if weights and abs(sum(weights.values()) - 1.0) < 0.1:  # Allow small rounding errors
            constraints.priority_weights = weights
        else:
            # Use default weights if LLM weights don't sum to 1
            constraints.priority_weights = self._infer_priority_weights(constraints.task_type, "")
        
        return constraints
    
    def _extract_constraints_rule_based(self, user_input: str, use_case_type: str = None, 
                                      budget_input: str = None, latency_input: str = None,
                                      context_input: str = None) -> UserConstraints:
        """
        Rule-based constraint extraction (original implementation)
        Used as fallback when LLM is unavailable
        """
        constraints = UserConstraints()
        
        # Extract task type
        constraints.task_type = self._extract_task_type(user_input, use_case_type)
        
        # Extract latency tolerance
        constraints.latency_tolerance = self._extract_latency_tolerance(user_input, latency_input)
        
        # Extract budget constraints
        constraints.max_cost_per_token = self._extract_budget_constraint(user_input, budget_input)
        
        # Extract context window requirements
        constraints.min_context_window = self._extract_context_window(user_input, context_input)
        
        # Extract required capabilities
        constraints.required_capabilities = self._extract_capabilities(user_input)
        
        # Extract deployment preferences
        constraints.deployment_preferences = self._extract_deployment_preferences(user_input)
        
        # Set priority weights based on task type and user input
        constraints.priority_weights = self._infer_priority_weights(constraints.task_type, user_input)
        
        return constraints
    
    def _extract_task_type(self, user_input: str, explicit_type: str = None) -> TaskType:
        """Extract or infer task type from input"""
        if explicit_type:
            try:
                return TaskType(explicit_type.lower())
            except ValueError:
                pass
        
        user_input_lower = user_input.lower()
        scores = {}
        
        for task_type, keywords in self.task_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            scores[task_type] = score
        
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        # Default to generative if unclear
        return TaskType.GENERATIVE
    
    def _extract_latency_tolerance(self, user_input: str, explicit_latency: str = None) -> LatencyCategory:
        """Extract latency requirements from input"""
        if explicit_latency:
            try:
                return LatencyCategory(explicit_latency.lower().replace(" ", "-"))
            except ValueError:
                pass
        
        user_input_lower = user_input.lower()
        scores = {}
        
        for latency_cat, keywords in self.latency_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            scores[latency_cat] = score
        
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        # Default to interactive for most use cases
        return LatencyCategory.INTERACTIVE
    
    def _extract_budget_constraint(self, user_input: str, explicit_budget: str = None) -> Optional[float]:
        """Extract budget constraints from input"""
        if explicit_budget:
            try:
                # Handle various budget formats
                budget_clean = re.sub(r'[^\d.]', '', explicit_budget)
                if budget_clean:
                    budget_value = float(budget_clean)
                    # Convert to per-token cost if it seems like a monthly budget
                    if budget_value > 1:  # Assume monthly budget
                        # Rough estimate: $100/month ≈ $0.00005 per token for moderate usage
                        return budget_value * 0.0000005
                    return budget_value
            except ValueError:
                pass
        
        # Look for budget indicators in text
        user_input_lower = user_input.lower()
        if any(word in user_input_lower for word in ["cheap", "budget", "low cost", "affordable"]):
            return 0.00001  # Budget tier
        elif any(word in user_input_lower for word in ["expensive", "premium", "high quality", "best"]):
            return 0.0001   # Premium tier
        
        # Default to standard tier
        return 0.00005
    
    def _extract_context_window(self, user_input: str, explicit_context: str = None) -> Optional[int]:
        """Extract context window requirements"""
        if explicit_context:
            try:
                context_clean = re.sub(r'[^\d]', '', explicit_context)
                if context_clean:
                    return int(context_clean)
            except ValueError:
                pass
        
        # Look for context indicators in text
        user_input_lower = user_input.lower()
        
        # Extract numbers that might be context window sizes
        numbers = re.findall(r'\b(\d+)k?\b', user_input_lower)
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1000 <= num <= 1000000:  # Reasonable context window range
                    return num
                elif 1 <= num <= 1000 and 'k' in user_input_lower:  # Handle "32k" format
                    return num * 1000
            except ValueError:
                continue
        
        # Infer from use case type
        if any(word in user_input_lower for word in ["long", "large", "document", "book", "extensive"]):
            return 32000
        elif any(word in user_input_lower for word in ["short", "brief", "quick", "simple"]):
            return 4000
        
        # Default to moderate context window
        return 8000
    
    def _extract_capabilities(self, user_input: str) -> List[str]:
        """Extract required capabilities from input"""
        user_input_lower = user_input.lower()
        required_capabilities = []
        
        for capability, keywords in self.capability_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                required_capabilities.append(capability)
        
        # Default to text generation if no specific capabilities found
        if not required_capabilities:
            required_capabilities.append("text_generation")
        
        return required_capabilities
    
    def _extract_deployment_preferences(self, user_input: str) -> List[str]:
        """Extract deployment preferences from input"""
        user_input_lower = user_input.lower()
        preferences = []
        
        if any(word in user_input_lower for word in ["local", "on-premise", "private"]):
            preferences.append("local")
        if any(word in user_input_lower for word in ["cloud", "api", "hosted"]):
            preferences.append("api")
        if any(word in user_input_lower for word in ["azure", "microsoft"]):
            preferences.append("azure")
        
        # Default to API if no preference specified
        if not preferences:
            preferences.append("api")
        
        return preferences
    
    def _infer_priority_weights(self, task_type: TaskType, user_input: str) -> Dict[str, float]:
        """Infer priority weights based on task type and user input"""
        user_input_lower = user_input.lower()
        
        # Base weights by task type
        if task_type == TaskType.ANALYTICAL:
            weights = {"reasoning": 0.4, "latency": 0.2, "cost": 0.2, "reliability": 0.2}
        elif task_type == TaskType.GENERATIVE:
            weights = {"reasoning": 0.3, "latency": 0.3, "cost": 0.2, "reliability": 0.2}
        elif task_type == TaskType.AGENTIC:
            weights = {"reasoning": 0.25, "latency": 0.25, "cost": 0.2, "reliability": 0.3}
        else:
            weights = {"reasoning": 0.25, "latency": 0.25, "cost": 0.25, "reliability": 0.25}
        
        # Adjust based on user input keywords
        if any(word in user_input_lower for word in ["fast", "quick", "speed", "real-time"]):
            weights["latency"] += 0.1
            weights["reasoning"] -= 0.05
            weights["cost"] -= 0.05
        
        if any(word in user_input_lower for word in ["cheap", "budget", "cost", "affordable"]):
            weights["cost"] += 0.1
            weights["reasoning"] -= 0.05
            weights["latency"] -= 0.05
        
        if any(word in user_input_lower for word in ["accurate", "quality", "smart", "intelligent"]):
            weights["reasoning"] += 0.1
            weights["cost"] -= 0.05
            weights["latency"] -= 0.05
        
        if any(word in user_input_lower for word in ["reliable", "stable", "production", "critical"]):
            weights["reliability"] += 0.1
            weights["cost"] -= 0.05
            weights["latency"] -= 0.05
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def ask_clarifying_questions(self, constraints: UserConstraints) -> List[str]:
        """
        Generate clarifying questions for incomplete constraints.
        Only ask when necessary and no safe defaults exist.
        """
        questions = []
        
        # Only ask about task type if completely unclear
        if not constraints.task_type:
            questions.append("What is the primary purpose of your LLM usage? (analysis, content generation, or automation)")
        
        # Only ask about budget if no indicators found
        if constraints.max_cost_per_token is None:
            questions.append("What is your budget constraint? (monthly budget or cost sensitivity)")
        
        # Only ask about context if use case suggests it's critical
        if constraints.min_context_window is None and len(constraints.required_capabilities) > 2:
            questions.append("Do you need to process long documents or maintain long conversations?")
        
        return questions
    
    def has_sufficient_information(self, constraints: UserConstraints) -> bool:
        """Check if we have enough information to proceed"""
        # We can proceed if we have task type and at least one other constraint
        essential_fields = [
            constraints.task_type is not None,
            constraints.latency_tolerance is not None,
            constraints.max_cost_per_token is not None
        ]
        
        return sum(essential_fields) >= 2


# Backward compatibility wrapper
class DiscoveryAgent(EnhancedDiscoveryAgent):
    """
    Backward compatibility wrapper for the original DiscoveryAgent.
    Maintains the same interface while providing enhanced LLM capabilities.
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """Initialize with optional LLM manager for enhanced capabilities"""
        super().__init__(llm_manager=llm_manager, use_llm=llm_manager is not None)


def create_discovery_agent(enable_llm: bool = True, llm_config: Optional[LLMConfig] = None) -> DiscoveryAgent:
    """
    Factory function to create Discovery Agent with optional LLM integration
    
    Args:
        enable_llm: Whether to enable LLM capabilities
        llm_config: LLM configuration (uses mock if None and enable_llm=True)
    
    Returns:
        DiscoveryAgent instance with appropriate configuration
    """
    if not enable_llm:
        return DiscoveryAgent()
    
    try:
        # Use provided config or default to mock for testing
        if llm_config is None:
            llm_config = LLMConfig(
                provider=LLMProvider.MOCK,
                model="mock-model",
                max_tokens=800,
                temperature=0.3
            )
        
        # Create LLM manager with fallback to mock
        fallback_config = LLMConfig(
            provider=LLMProvider.MOCK,
            model="mock-model",
            max_tokens=800,
            temperature=0.3
        )
        
        llm_manager = LLMManager(
            primary_config=llm_config,
            fallback_configs=[fallback_config] if llm_config.provider != LLMProvider.MOCK else None
        )
        
        return DiscoveryAgent(llm_manager=llm_manager)
        
    except Exception as e:
        logging.warning(f"Failed to create LLM-enabled Discovery Agent: {e}")
        return DiscoveryAgent()  # Fall back to rule-based only