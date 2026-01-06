"""
Constraint Filter Agent - Eliminates incompatible models based on hard constraints
"""

from typing import Dict, List, Tuple
from models.schemas import UserConstraints, ModelInfo, EliminationReason, LatencyCategory, CostTier
from models.registry import ModelRegistry


class ConstraintFilterAgent:
    """
    Eliminates models that violate non-negotiable constraints.
    Records detailed rationale for each elimination.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
        # Define latency hierarchy (lower index = faster)
        self.latency_hierarchy = {
            LatencyCategory.REAL_TIME: 0,
            LatencyCategory.INTERACTIVE: 1,
            LatencyCategory.BATCH: 2
        }
        
        # Define cost tier hierarchy (lower index = cheaper)
        self.cost_hierarchy = {
            CostTier.BUDGET: 0,
            CostTier.STANDARD: 1,
            CostTier.PREMIUM: 2
        }
    
    def filter_models(self, constraints: UserConstraints) -> Tuple[List[str], List[EliminationReason]]:
        """
        Filter models based on constraints and return viable models + elimination reasons.
        
        Args:
            constraints: User constraints to apply
            
        Returns:
            Tuple of (viable_model_ids, elimination_reasons)
        """
        all_models = self.registry.get_all_models()
        viable_models = []
        elimination_reasons = []
        
        for model_id, model_info in all_models.items():
            elimination_reason = self._check_model_constraints(model_id, model_info, constraints)
            
            if elimination_reason:
                elimination_reasons.append(elimination_reason)
            else:
                viable_models.append(model_id)
        
        return viable_models, elimination_reasons
    
    def _check_model_constraints(self, model_id: str, model_info: ModelInfo, 
                                constraints: UserConstraints) -> EliminationReason:
        """
        Check if a model violates any constraints.
        
        Returns:
            EliminationReason if model should be eliminated, None otherwise
        """
        
        # Check budget constraints
        if constraints.max_cost_per_token is not None:
            model_cost = max(model_info.cost_per_token.get('input', 0), 
                           model_info.cost_per_token.get('output', 0))
            
            if model_cost > constraints.max_cost_per_token:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=f"Cost per token (${model_cost:.6f}) exceeds budget limit (${constraints.max_cost_per_token:.6f})",
                    constraint_violated="budget",
                    threshold_value=constraints.max_cost_per_token,
                    actual_value=model_cost
                )
        
        # Check latency requirements
        if constraints.latency_tolerance is not None:
            required_latency_level = self.latency_hierarchy[constraints.latency_tolerance]
            model_latency_level = self.latency_hierarchy[model_info.latency_category]
            
            # Model must be at least as fast as required (lower or equal hierarchy level)
            if model_latency_level > required_latency_level:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=f"Latency category '{model_info.latency_category.value}' is too slow for required '{constraints.latency_tolerance.value}'",
                    constraint_violated="latency",
                    threshold_value=required_latency_level,
                    actual_value=model_latency_level
                )
        
        # Check context window requirements
        if constraints.min_context_window is not None:
            if model_info.context_window < constraints.min_context_window:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=f"Context window ({model_info.context_window:,} tokens) is smaller than required ({constraints.min_context_window:,} tokens)",
                    constraint_violated="context_window",
                    threshold_value=constraints.min_context_window,
                    actual_value=model_info.context_window
                )
        
        # Check required capabilities
        if constraints.required_capabilities:
            missing_capabilities = []
            for required_cap in constraints.required_capabilities:
                if required_cap not in model_info.capabilities:
                    missing_capabilities.append(required_cap)
            
            if missing_capabilities:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=f"Missing required capabilities: {', '.join(missing_capabilities)}",
                    constraint_violated="capabilities",
                    threshold_value=None,
                    actual_value=None
                )
        
        # Check deployment preferences
        if constraints.deployment_preferences:
            compatible_deployments = []
            for pref in constraints.deployment_preferences:
                if pref in model_info.deployment_options:
                    compatible_deployments.append(pref)
            
            if not compatible_deployments:
                return EliminationReason(
                    model_name=model_info.name,
                    reason=f"No compatible deployment options. Required: {', '.join(constraints.deployment_preferences)}, Available: {', '.join(model_info.deployment_options)}",
                    constraint_violated="deployment",
                    threshold_value=None,
                    actual_value=None
                )
        
        # Model passes all constraints
        return None
    
    def get_filtering_summary(self, viable_models: List[str], 
                            elimination_reasons: List[EliminationReason]) -> Dict:
        """
        Generate a summary of the filtering process.
        
        Returns:
            Dictionary with filtering statistics and summaries
        """
        total_models = len(self.registry.get_all_models())
        eliminated_count = len(elimination_reasons)
        viable_count = len(viable_models)
        
        # Group eliminations by constraint type
        elimination_by_constraint = {}
        for reason in elimination_reasons:
            constraint = reason.constraint_violated
            if constraint not in elimination_by_constraint:
                elimination_by_constraint[constraint] = []
            elimination_by_constraint[constraint].append(reason)
        
        # Get viable model details
        viable_model_details = []
        for model_id in viable_models:
            model_info = self.registry.get_model(model_id)
            viable_model_details.append({
                'id': model_id,
                'name': model_info.name,
                'provider': model_info.provider,
                'cost_tier': model_info.cost_tier.value,
                'latency': model_info.latency_category.value,
                'context_window': model_info.context_window,
                'reasoning_strength': model_info.reasoning_strength
            })
        
        return {
            'total_models': total_models,
            'eliminated_count': eliminated_count,
            'viable_count': viable_count,
            'elimination_rate': eliminated_count / total_models if total_models > 0 else 0,
            'elimination_by_constraint': elimination_by_constraint,
            'viable_models': viable_model_details,
            'elimination_reasons': elimination_reasons
        }
    
    def explain_elimination(self, model_id: str, constraints: UserConstraints) -> str:
        """
        Provide detailed explanation for why a specific model was eliminated.
        
        Args:
            model_id: ID of the model to explain
            constraints: User constraints that were applied
            
        Returns:
            Detailed explanation string
        """
        try:
            model_info = self.registry.get_model(model_id)
            elimination_reason = self._check_model_constraints(model_id, model_info, constraints)
            
            if elimination_reason:
                explanation = f"**{model_info.name}** was eliminated because:\n\n"
                explanation += f"• **Constraint Violated:** {elimination_reason.constraint_violated}\n"
                explanation += f"• **Reason:** {elimination_reason.reason}\n"
                
                if elimination_reason.threshold_value is not None:
                    explanation += f"• **Required:** {elimination_reason.threshold_value}\n"
                    explanation += f"• **Actual:** {elimination_reason.actual_value}\n"
                
                return explanation
            else:
                return f"**{model_info.name}** was not eliminated and should be in the viable models list."
                
        except KeyError:
            return f"Model '{model_id}' not found in registry."
    
    def get_constraint_impact_analysis(self, constraints: UserConstraints) -> Dict:
        """
        Analyze the impact of each constraint on model elimination.
        
        Returns:
            Dictionary showing how many models each constraint eliminates
        """
        all_models = self.registry.get_all_models()
        constraint_impacts = {
            'budget': 0,
            'latency': 0,
            'context_window': 0,
            'capabilities': 0,
            'deployment': 0
        }
        
        for model_id, model_info in all_models.items():
            # Test each constraint individually
            
            # Budget impact
            if constraints.max_cost_per_token is not None:
                model_cost = max(model_info.cost_per_token.get('input', 0), 
                               model_info.cost_per_token.get('output', 0))
                if model_cost > constraints.max_cost_per_token:
                    constraint_impacts['budget'] += 1
            
            # Latency impact
            if constraints.latency_tolerance is not None:
                required_latency_level = self.latency_hierarchy[constraints.latency_tolerance]
                model_latency_level = self.latency_hierarchy[model_info.latency_category]
                if model_latency_level > required_latency_level:
                    constraint_impacts['latency'] += 1
            
            # Context window impact
            if constraints.min_context_window is not None:
                if model_info.context_window < constraints.min_context_window:
                    constraint_impacts['context_window'] += 1
            
            # Capabilities impact
            if constraints.required_capabilities:
                missing_capabilities = [cap for cap in constraints.required_capabilities 
                                      if cap not in model_info.capabilities]
                if missing_capabilities:
                    constraint_impacts['capabilities'] += 1
            
            # Deployment impact
            if constraints.deployment_preferences:
                compatible_deployments = [pref for pref in constraints.deployment_preferences 
                                        if pref in model_info.deployment_options]
                if not compatible_deployments:
                    constraint_impacts['deployment'] += 1
        
        return constraint_impacts