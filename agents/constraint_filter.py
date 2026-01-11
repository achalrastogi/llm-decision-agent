"""
Constraint Filter Agent
-----------------------
Eliminates models ONLY when they violate explicit, non-negotiable constraints.

Design principles:
- Never eliminate on inferred or default values
- Treat None as "no constraint"
- Capabilities and deployment are enforced ONLY if explicitly provided
"""

from typing import Dict, List, Tuple
from models.schemas import (
    UserConstraints,
    ModelInfo,
    EliminationReason,
    LatencyCategory,
    CostTier,
)
from models.registry import ModelRegistry


class ConstraintFilterAgent:
    """
    Eliminates models that violate explicit constraints.
    Records detailed rationale for each elimination.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

        # Latency hierarchy (lower = faster)
        self.latency_hierarchy = {
            LatencyCategory.REAL_TIME: 0,
            LatencyCategory.INTERACTIVE: 1,
            LatencyCategory.BATCH: 2,
        }

        # Cost tier hierarchy (lower = cheaper)
        self.cost_hierarchy = {
            CostTier.BUDGET: 0,
            CostTier.STANDARD: 1,
            CostTier.PREMIUM: 2,
        }

    # =====================================================
    # PUBLIC API
    # =====================================================
    def filter_models(
        self, constraints: UserConstraints
    ) -> Tuple[List[str], List[EliminationReason]]:
        """
        Apply constraints and return viable models + elimination reasons.
        """
        all_models = self.registry.get_all_models()
        viable_models: List[str] = []
        elimination_reasons: List[EliminationReason] = []

        for model_id, model_info in all_models.items():
            reason = self._check_model_constraints(
                model_id, model_info, constraints
            )

            if reason:
                elimination_reasons.append(reason)
            else:
                viable_models.append(model_id)

        return viable_models, elimination_reasons

    # =====================================================
    # CORE CONSTRAINT CHECK
    # =====================================================
    def _check_model_constraints(
        self,
        model_id: str,
        model_info: ModelInfo,
        constraints: UserConstraints,
    ) -> EliminationReason | None:
        """
        Returns EliminationReason if model should be eliminated,
        otherwise returns None.
        """

        # -----------------------------
        # Budget (ONLY if explicit)
        # -----------------------------
        if constraints.max_cost_per_token is not None:
            model_cost = max(
                model_info.cost_per_token.get("input", 0),
                model_info.cost_per_token.get("output", 0),
            )

            if model_cost > constraints.max_cost_per_token:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=(
                        f"Cost per token (${model_cost:.6f}) exceeds "
                        f"budget limit (${constraints.max_cost_per_token:.6f})"
                    ),
                    constraint_violated="budget",
                    threshold_value=constraints.max_cost_per_token,
                    actual_value=model_cost,
                )

        # -----------------------------
        # Latency (ONLY if explicit)
        # -----------------------------
        if constraints.latency_tolerance is not None:
            required_level = self.latency_hierarchy[
                constraints.latency_tolerance
            ]
            model_level = self.latency_hierarchy[
                model_info.latency_category
            ]

            if model_level > required_level:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=(
                        f"Latency '{model_info.latency_category.value}' "
                        f"is slower than required "
                        f"'{constraints.latency_tolerance.value}'"
                    ),
                    constraint_violated="latency",
                    threshold_value=required_level,
                    actual_value=model_level,
                )

        # -----------------------------
        # Context window (ONLY if explicit)
        # -----------------------------
        if constraints.min_context_window is not None:
            if model_info.context_window < constraints.min_context_window:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=(
                        f"Context window ({model_info.context_window:,}) "
                        f"is smaller than required "
                        f"({constraints.min_context_window:,})"
                    ),
                    constraint_violated="context_window",
                    threshold_value=constraints.min_context_window,
                    actual_value=model_info.context_window,
                )

        # -----------------------------
        # Capabilities (ONLY if explicit)
        # -----------------------------
        if (
            constraints.required_capabilities
            and getattr(constraints, "explicit_capabilities", False)
        ):
            missing = [
                cap
                for cap in constraints.required_capabilities
                if cap not in model_info.capabilities
            ]

            if missing:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=f"Missing required capabilities: {', '.join(missing)}",
                    constraint_violated="capabilities",
                    threshold_value=None,
                    actual_value=None,
                )

        # -----------------------------
        # Deployment (ONLY if explicit)
        # -----------------------------
        if (
            constraints.deployment_preferences
            and getattr(constraints, "explicit_deployment", False)
        ):
            compatible = [
                pref
                for pref in constraints.deployment_preferences
                if pref in model_info.deployment_options
            ]

            if not compatible:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=(
                        f"No compatible deployment options. "
                        f"Required: {', '.join(constraints.deployment_preferences)}; "
                        f"Available: {', '.join(model_info.deployment_options)}"
                    ),
                    constraint_violated="deployment",
                    threshold_value=None,
                    actual_value=None,
                )

        # -----------------------------
        # PASSES ALL CONSTRAINTS
        # -----------------------------
        return None

    # =====================================================
    # SUMMARY & ANALYTICS (UNCHANGED BEHAVIOR)
    # =====================================================
    def get_filtering_summary(
        self,
        viable_models: List[str],
        elimination_reasons: List[EliminationReason],
    ) -> Dict:
        total_models = len(self.registry.get_all_models())
        eliminated_count = len(elimination_reasons)
        viable_count = len(viable_models)

        elimination_by_constraint: Dict[str, List[EliminationReason]] = {}
        for reason in elimination_reasons:
            elimination_by_constraint.setdefault(
                reason.constraint_violated, []
            ).append(reason)

        viable_model_details = []
        for model_id in viable_models:
            model_info = self.registry.get_model(model_id)
            viable_model_details.append(
                {
                    "id": model_id,
                    "name": model_info.name,
                    "provider": model_info.provider,
                    "cost_tier": model_info.cost_tier.value,
                    "latency": model_info.latency_category.value,
                    "context_window": model_info.context_window,
                    "reasoning_strength": model_info.reasoning_strength,
                }
            )

        return {
            "total_models": total_models,
            "eliminated_count": eliminated_count,
            "viable_count": viable_count,
            "elimination_rate": (
                eliminated_count / total_models
                if total_models > 0
                else 0
            ),
            "elimination_by_constraint": elimination_by_constraint,
            "viable_models": viable_model_details,
            "elimination_reasons": elimination_reasons,
        }

    def get_constraint_impact_analysis(
        self, constraints: UserConstraints
    ) -> Dict:
        """
        Counts how many models WOULD be eliminated by each explicit constraint.
        """
        all_models = self.registry.get_all_models()
        impacts = {
            "budget": 0,
            "latency": 0,
            "context_window": 0,
            "capabilities": 0,
            "deployment": 0,
        }

        for _, model_info in all_models.items():

            if constraints.max_cost_per_token is not None:
                model_cost = max(
                    model_info.cost_per_token.get("input", 0),
                    model_info.cost_per_token.get("output", 0),
                )
                if model_cost > constraints.max_cost_per_token:
                    impacts["budget"] += 1

            if constraints.latency_tolerance is not None:
                if (
                    self.latency_hierarchy[model_info.latency_category]
                    > self.latency_hierarchy[constraints.latency_tolerance]
                ):
                    impacts["latency"] += 1

            if constraints.min_context_window is not None:
                if model_info.context_window < constraints.min_context_window:
                    impacts["context_window"] += 1

            if (
                constraints.required_capabilities
                and getattr(constraints, "explicit_capabilities", False)
            ):
                missing = [
                    cap
                    for cap in constraints.required_capabilities
                    if cap not in model_info.capabilities
                ]
                if missing:
                    impacts["capabilities"] += 1

            if (
                constraints.deployment_preferences
                and getattr(constraints, "explicit_deployment", False)
            ):
                compatible = [
                    pref
                    for pref in constraints.deployment_preferences
                    if pref in model_info.deployment_options
                ]
                if not compatible:
                    impacts["deployment"] += 1

        return impacts
