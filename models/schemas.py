"""
Pydantic schemas for the LLM Decision Agent system
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class LatencyCategory(str, Enum):
    REAL_TIME = "real-time"
    INTERACTIVE = "interactive"
    BATCH = "batch"


class CostTier(str, Enum):
    BUDGET = "budget"
    STANDARD = "standard"
    PREMIUM = "premium"


class TaskType(str, Enum):
    ANALYTICAL = "analytical"
    GENERATIVE = "generative"
    AGENTIC = "agentic"


class ModelInfo(BaseModel):
    """Schema for LLM model information"""
    name: str
    provider: str
    context_window: int
    latency_category: LatencyCategory
    cost_tier: CostTier
    reasoning_strength: float = Field(ge=1.0, le=10.0)
    tool_reliability: float = Field(ge=1.0, le=10.0)
    capabilities: List[str]
    deployment_options: List[str]
    cost_per_token: Dict[str, float]
    benchmark_scores: Dict[str, float]


class UserConstraints(BaseModel):
    """Schema for user constraints and requirements"""
    task_type: Optional[TaskType] = None
    latency_tolerance: Optional[LatencyCategory] = None
    max_cost_per_token: Optional[float] = None
    min_context_window: Optional[int] = None
    required_capabilities: List[str] = []
    deployment_preferences: List[str] = []
    priority_weights: Dict[str, float] = {
        "reasoning": 0.25,
        "latency": 0.25,
        "cost": 0.25,
        "reliability": 0.25
    }


class EliminationReason(BaseModel):
    """Schema for model elimination rationale"""
    model_name: str
    reason: str
    constraint_violated: str
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None


class ModelScore(BaseModel):
    """Schema for model scoring results"""
    model_name: str
    overall_score: float
    dimension_scores: Dict[str, float]
    explanations: Dict[str, str]


class SessionState(BaseModel):
    """Schema for session state management"""
    session_id: str
    current_step: int = 1
    constraints: Optional[UserConstraints] = None
    eliminated_models: List[EliminationReason] = []
    viable_models: List[str] = []
    model_scores: List[ModelScore] = []
    recommendation: Optional[Dict] = None