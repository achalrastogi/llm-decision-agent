"""
Discovery Agent
---------------
Extracts user constraints and priorities from a free-form use case.
Supports:
- Rule-based deterministic extraction
- AI-assisted extraction using structured LLM prompts
"""

import json
import logging
from typing import Dict, List, Optional

from models.schemas import UserConstraints, TaskType, LatencyCategory
from agents.llm_adapter import LLMManager, LLMConfig, LLMProvider

logger = logging.getLogger("discovery_agent")

# =========================================================
# PROMPT ENGINEERING (JUDGE-GRADE)
# =========================================================

DISCOVERY_SYSTEM_PROMPT = """
You are a principal AI systems architect specializing in large language model
selection, cost–latency trade-offs, and enterprise deployment.

Your task is to extract structured technical requirements from a user’s
use-case description.

Rules:
- Be conservative and precise
- Do NOT infer unstated requirements
- If information is missing or ambiguous, use null
- Prefer correctness over completeness
- Do NOT recommend any model
- Respond in valid JSON only
"""

DISCOVERY_OUTPUT_SCHEMA = {
    "task_type": "analytical | generative | agentic | null",
    "latency_tolerance": "real_time | interactive | batch | null",
    "max_cost_per_token": "number | null",
    "min_context_window": "number | null",
    "required_capabilities": ["text_generation | code_generation | analysis | tool_calling | multimodal"],
    "deployment_preferences": ["api | local | azure"],
    "priority_weights": {
        "reasoning": "number",
        "latency": "number",
        "cost": "number",
        "reliability": "number"
    }
}

# =========================================================
# DISCOVERY AGENT
# =========================================================

class DiscoveryAgent:
    """
    Extracts user intent and constraints.
    Uses LLM for reasoning, rules for enforcement.
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.llm_manager = llm_manager
        self.use_llm = llm_manager is not None
        self.logger = logger

    # -----------------------------------------------------
    # PUBLIC API (USED BY UI)
    # -----------------------------------------------------
    def extract_constraints(
        self,
        user_input: str,
        use_case_type: Optional[str] = None,
        budget_input: Optional[str] = None,
        latency_input: Optional[str] = None,
        context_input: Optional[str] = None,
    ) -> UserConstraints:

        if self.use_llm:
            try:
                return self._extract_with_llm(
                    user_input,
                    use_case_type,
                    budget_input,
                    latency_input,
                    context_input,
                )
            except Exception as e:
                self.logger.warning(f"LLM extraction failed, falling back: {e}")

        return self._extract_rule_based(
            user_input,
            use_case_type,
            budget_input,
            latency_input,
            context_input,
        )

    # -----------------------------------------------------
    # LLM-BASED EXTRACTION (SINGLE CALL)
    # -----------------------------------------------------
    def _extract_with_llm(
        self,
        user_input: str,
        use_case_type: Optional[str],
        budget_input: Optional[str],
        latency_input: Optional[str],
        context_input: Optional[str],
    ) -> UserConstraints:

        user_prompt = self._build_user_prompt(
            user_input,
            use_case_type,
            budget_input,
            latency_input,
            context_input,
        )

        response = self.llm_manager.generate(
            system_prompt=DISCOVERY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_schema=DISCOVERY_OUTPUT_SCHEMA,
            temperature=0.2,
            max_tokens=700,
        )

        data = json.loads(response.content)
        return self._build_constraints_from_llm(data)

    # -----------------------------------------------------
    # USER PROMPT (DATA ONLY)
    # -----------------------------------------------------
    def _build_user_prompt(
        self,
        user_input: str,
        use_case_type: Optional[str],
        budget_input: Optional[str],
        latency_input: Optional[str],
        context_input: Optional[str],
    ) -> str:

        parts = [
            f"Use case description:\n{user_input}"
        ]

        if use_case_type:
            parts.append(f"Explicit task type: {use_case_type}")
        if budget_input:
            parts.append(f"Budget input: {budget_input}")
        if latency_input:
            parts.append(f"Latency input: {latency_input}")
        if context_input:
            parts.append(f"Context window input: {context_input}")

        return "\n\n".join(parts)

    # -----------------------------------------------------
    # BUILD CONSTRAINT OBJECT (SAFE)
    # -----------------------------------------------------
    def _build_constraints_from_llm(self, data: Dict) -> UserConstraints:
        c = UserConstraints()

        # Task type
        if data.get("task_type"):
            c.task_type = TaskType(data["task_type"])

        # Latency
        if data.get("latency_tolerance"):
            c.latency_tolerance = LatencyCategory(
                data["latency_tolerance"].replace("_", "-")
            )

        c.max_cost_per_token = data.get("max_cost_per_token")
        c.min_context_window = data.get("min_context_window", 8000)
        c.required_capabilities = data.get("required_capabilities", ["text_generation"])
        c.deployment_preferences = data.get("deployment_preferences", ["api"])

        weights = data.get("priority_weights")
        if weights and abs(sum(weights.values()) - 1.0) < 0.05:
            c.priority_weights = weights
        else:
            c.priority_weights = self._default_weights(c.task_type)

        return c

    # -----------------------------------------------------
    # RULE-BASED FALLBACK (UNCHANGED LOGIC)
    # -----------------------------------------------------
    def _extract_rule_based(
    self,
    user_input: str,
    use_case_type: Optional[str],
    budget_input: Optional[str],
    latency_input: Optional[str],
    context_input: Optional[str],
) -> UserConstraints:

        c = UserConstraints()

        # Task type (only if explicitly provided)
        if use_case_type:
            c.task_type = TaskType(use_case_type.lower())
        else:
            c.task_type = None  # IMPORTANT: do not guess

        # Latency (only if explicitly provided)
        if latency_input:
            try:
                c.latency_tolerance = LatencyCategory(latency_input)
            except ValueError:
                c.latency_tolerance = None
        else:
            c.latency_tolerance = None

        # Budget (do NOT invent)
        if budget_input:
            text = budget_input.lower()

            if "low" in text or "budget" in text or "cheap" in text:
                c.max_cost_per_token = 0.00001
            elif "medium" in text:
                c.max_cost_per_token = 0.00005
            elif "high" in text or "no limit" in text:
                c.max_cost_per_token = None
            else:
                c.max_cost_per_token = None
        else:
            c.max_cost_per_token = None

        # Context window (do NOT invent)
        if context_input:
            text = context_input.lower()

            if "32k" in text:
                c.min_context_window = 32000
            elif "16k" in text:
                c.min_context_window = 16000
            elif "8k" in text:
                c.min_context_window = 8000
            elif "long" in text:
                c.min_context_window = 16000
            else:
                c.min_context_window = None
        else:
            c.min_context_window = None

        # Capabilities (minimum viable)
        c.required_capabilities = ["text_generation"]

        # Deployment (neutral)
        c.deployment_preferences = ["api"]

        # Weights (default-neutral)
        c.priority_weights = self._default_weights(c.task_type)

        return c


    # -----------------------------------------------------
    # HELPERS
    # -----------------------------------------------------
    def _default_weights(self, task_type: TaskType) -> Dict[str, float]:
        if task_type == TaskType.ANALYTICAL:
            return {"reasoning": 0.4, "latency": 0.2, "cost": 0.2, "reliability": 0.2}
        if task_type == TaskType.AGENTIC:
            return {"reasoning": 0.25, "latency": 0.25, "cost": 0.2, "reliability": 0.3}
        return {"reasoning": 0.3, "latency": 0.3, "cost": 0.2, "reliability": 0.2}

    # -----------------------------------------------------
    # UI SUPPORT METHODS
    # -----------------------------------------------------
    def ask_clarifying_questions(self, constraints: UserConstraints) -> List[str]:
        questions = []
        if not constraints.task_type:
            questions.append("What is the primary task? (analysis, generation, automation)")
        if not constraints.max_cost_per_token:
            questions.append("Do you have a budget constraint?")
        return questions

    def has_sufficient_information(self, constraints: UserConstraints) -> bool:
        return constraints.task_type is not None


# =========================================================
# FACTORY (USED BY UI)
# =========================================================

def create_discovery_agent(
    enable_llm: bool = True,
    llm_config: Optional[LLMConfig] = None,
) -> DiscoveryAgent:

    if not enable_llm:
        return DiscoveryAgent()

    try:
        if llm_config is None:
            llm_config = LLMConfig(
                provider=LLMProvider.MOCK,
                model="mock-model",
            )

        manager = LLMManager(
            primary_config=llm_config,
            fallback_configs=[
                LLMConfig(provider=LLMProvider.MOCK, model="mock-model")
            ]
            if llm_config.provider != LLMProvider.MOCK
            else None,
        )

        return DiscoveryAgent(llm_manager=manager)

    except Exception as e:
        logger.warning(f"LLM Discovery init failed: {e}")
        return DiscoveryAgent()
