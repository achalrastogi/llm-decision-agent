"""
Decision Engine Agent - Applies weighted scoring and computes trade-offs
"""

from typing import Dict, List, Tuple
import math
from models.schemas import UserConstraints, ModelInfo, ModelScore
from models.registry import ModelRegistry


class DecisionEngineAgent:
    """
    Applies weighted scoring to viable models and computes trade-offs.
    Generates relative scores with transparent explanations.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
        # Scoring weights and normalization factors
        self.scoring_config = {
            'reasoning': {
                'weight_key': 'reasoning',
                'normalize_range': (1.0, 10.0),  # reasoning_strength range
                'higher_is_better': True
            },
            'latency': {
                'weight_key': 'latency', 
                'categories': {'real-time': 1.0, 'interactive': 0.7, 'batch': 0.3},
                'higher_is_better': True
            },
            'cost': {
                'weight_key': 'cost',
                'higher_is_better': False,  # Lower cost is better
                'log_scale': True  # Use log scale for cost comparison
            },
            'reliability': {
                'weight_key': 'reliability',
                'normalize_range': (1.0, 10.0),  # tool_reliability range
                'higher_is_better': True
            }
        }
    
    def score_models(self, viable_model_ids: List[str], 
                    constraints: UserConstraints) -> List[ModelScore]:
        """
        Score viable models using weighted criteria.
        
        Args:
            viable_model_ids: List of model IDs that passed filtering
            constraints: User constraints including priority weights
            
        Returns:
            List of ModelScore objects with relative scores
        """
        if not viable_model_ids:
            return []
        
        # Get model information
        models = {model_id: self.registry.get_model(model_id) 
                 for model_id in viable_model_ids}
        
        # Calculate raw scores for each dimension
        raw_scores = self._calculate_raw_scores(models)
        
        # Normalize scores to 0-1 range for comparison
        normalized_scores = self._normalize_scores(raw_scores)
        
        # Apply user priority weights
        weighted_scores = self._apply_weights(normalized_scores, constraints.priority_weights)
        
        # Generate explanations
        model_scores = []
        for model_id in viable_model_ids:
            explanations = self._generate_explanations(
                model_id, models[model_id], raw_scores[model_id], 
                normalized_scores[model_id], constraints
            )
            
            model_score = ModelScore(
                model_name=models[model_id].name,
                overall_score=weighted_scores[model_id]['overall'],
                dimension_scores=weighted_scores[model_id]['dimensions'],
                explanations=explanations
            )
            model_scores.append(model_score)
        
        # Sort by overall score (descending)
        model_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return model_scores
    
    def _calculate_raw_scores(self, models: Dict[str, ModelInfo]) -> Dict[str, Dict[str, float]]:
        """Calculate raw scores for each model and dimension"""
        raw_scores = {}
        
        for model_id, model_info in models.items():
            scores = {}
            
            # Reasoning score (direct from model data)
            scores['reasoning'] = model_info.reasoning_strength
            
            # Latency score (categorical mapping)
            latency_mapping = self.scoring_config['latency']['categories']
            scores['latency'] = latency_mapping.get(model_info.latency_category.value, 0.5)
            
            # Cost score (inverse of cost - lower cost = higher score)
            max_cost = max(model_info.cost_per_token.get('input', 0),
                          model_info.cost_per_token.get('output', 0))
            scores['cost'] = max_cost
            
            # Reliability score (direct from model data)
            scores['reliability'] = model_info.tool_reliability
            
            raw_scores[model_id] = scores
        
        return raw_scores
    
    def _normalize_scores(self, raw_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize scores to 0-1 range for fair comparison"""
        if not raw_scores:
            return {}
        
        # Find min/max for each dimension across all models
        dimensions = ['reasoning', 'latency', 'cost', 'reliability']
        min_max = {}
        
        for dim in dimensions:
            values = [scores[dim] for scores in raw_scores.values()]
            min_max[dim] = {'min': min(values), 'max': max(values)}
        
        # Normalize each score
        normalized = {}
        for model_id, scores in raw_scores.items():
            normalized[model_id] = {}
            
            for dim in dimensions:
                min_val = min_max[dim]['min']
                max_val = min_max[dim]['max']
                
                if max_val == min_val:
                    # All models have same value for this dimension
                    normalized_score = 1.0
                else:
                    # Normalize to 0-1 range
                    normalized_score = (scores[dim] - min_val) / (max_val - min_val)
                    
                    # Invert for cost (lower cost = higher score)
                    if dim == 'cost':
                        normalized_score = 1.0 - normalized_score
                
                normalized[model_id][dim] = normalized_score
        
        return normalized
    
    def _apply_weights(self, normalized_scores: Dict[str, Dict[str, float]], 
                      priority_weights: Dict[str, float]) -> Dict[str, Dict]:
        """Apply user priority weights to normalized scores"""
        weighted_scores = {}
        
        for model_id, scores in normalized_scores.items():
            # Calculate weighted score for each dimension
            weighted_dimensions = {}
            overall_score = 0.0
            
            for dimension, normalized_score in scores.items():
                weight = priority_weights.get(dimension, 0.25)  # Default equal weight
                weighted_score = normalized_score * weight
                weighted_dimensions[dimension] = weighted_score
                overall_score += weighted_score
            
            weighted_scores[model_id] = {
                'overall': overall_score,
                'dimensions': weighted_dimensions
            }
        
        return weighted_scores
    
    def _generate_explanations(self, model_id: str, model_info: ModelInfo,
                             raw_scores: Dict[str, float], 
                             normalized_scores: Dict[str, float],
                             constraints: UserConstraints) -> Dict[str, str]:
        """Generate explanations for each score dimension"""
        explanations = {}
        
        # Reasoning explanation
        reasoning_raw = raw_scores['reasoning']
        reasoning_norm = normalized_scores['reasoning']
        explanations['reasoning'] = (
            f"Reasoning strength of {reasoning_raw}/10 "
            f"(normalized to {reasoning_norm:.2f} relative to other viable models). "
            f"Based on model's cognitive capabilities and benchmark performance."
        )
        
        # Latency explanation
        latency_category = model_info.latency_category.value
        latency_norm = normalized_scores['latency']
        explanations['latency'] = (
            f"Latency category '{latency_category}' "
            f"(normalized to {latency_norm:.2f} relative to other viable models). "
            f"Reflects expected response time characteristics."
        )
        
        # Cost explanation
        max_cost = max(model_info.cost_per_token.get('input', 0),
                      model_info.cost_per_token.get('output', 0))
        cost_norm = normalized_scores['cost']
        explanations['cost'] = (
            f"Maximum cost per token of ${max_cost:.6f} "
            f"(normalized to {cost_norm:.2f} where higher = more cost-effective). "
            f"Based on input/output token pricing."
        )
        
        # Reliability explanation
        reliability_raw = raw_scores['reliability']
        reliability_norm = normalized_scores['reliability']
        explanations['reliability'] = (
            f"Tool reliability of {reliability_raw}/10 "
            f"(normalized to {reliability_norm:.2f} relative to other viable models). "
            f"Reflects API stability and tool integration quality."
        )
        
        return explanations
    
    def generate_trade_off_analysis(self, model_scores: List[ModelScore]) -> Dict:
        """
        Generate trade-off analysis between top models.
        
        Args:
            model_scores: List of scored models (should be sorted by overall score)
            
        Returns:
            Dictionary with trade-off analysis
        """
        if len(model_scores) < 2:
            return {
                'top_models': model_scores[:1] if model_scores else [],
                'trade_offs': [],
                'recommendations': []
            }
        
        # Analyze top 3 models
        top_models = model_scores[:3]
        trade_offs = []
        
        # Compare top model with alternatives
        best_model = top_models[0]
        
        for i, alternative in enumerate(top_models[1:], 1):
            trade_off = self._compare_models(best_model, alternative, i)
            trade_offs.append(trade_off)
        
        # Generate recommendations based on trade-offs
        recommendations = self._generate_trade_off_recommendations(top_models)
        
        return {
            'top_models': top_models,
            'trade_offs': trade_offs,
            'recommendations': recommendations,
            'score_gap_analysis': self._analyze_score_gaps(top_models)
        }
    
    def _compare_models(self, model_a: ModelScore, model_b: ModelScore, rank: int) -> Dict:
        """Compare two models and identify key trade-offs"""
        comparison = {
            'primary_model': model_a.model_name,
            'alternative_model': model_b.model_name,
            'alternative_rank': rank + 1,
            'score_difference': model_a.overall_score - model_b.overall_score,
            'advantages': [],
            'disadvantages': []
        }
        
        # Compare each dimension
        for dimension in ['reasoning', 'latency', 'cost', 'reliability']:
            score_a = model_a.dimension_scores.get(dimension, 0)
            score_b = model_b.dimension_scores.get(dimension, 0)
            
            difference = score_a - score_b
            
            if abs(difference) > 0.05:  # Significant difference threshold
                if difference > 0:
                    comparison['advantages'].append({
                        'dimension': dimension,
                        'advantage': f"Better {dimension}",
                        'score_difference': difference
                    })
                else:
                    comparison['disadvantages'].append({
                        'dimension': dimension,
                        'disadvantage': f"Weaker {dimension}",
                        'score_difference': abs(difference)
                    })
        
        return comparison
    
    def _generate_trade_off_recommendations(self, top_models: List[ModelScore]) -> List[str]:
        """Generate recommendations based on trade-off analysis"""
        recommendations = []
        
        if len(top_models) < 2:
            return recommendations
        
        best_model = top_models[0]
        second_best = top_models[1]
        
        # Check if scores are very close
        score_gap = best_model.overall_score - second_best.overall_score
        
        if score_gap < 0.1:  # Very close scores
            recommendations.append(
                f"Scores are very close between {best_model.model_name} and "
                f"{second_best.model_name}. Consider running evaluation tests."
            )
        
        # Check for dimension-specific recommendations
        for dimension in ['reasoning', 'latency', 'cost', 'reliability']:
            best_score = best_model.dimension_scores.get(dimension, 0)
            second_score = second_best.dimension_scores.get(dimension, 0)
            
            if second_score > best_score + 0.15:  # Alternative significantly better
                recommendations.append(
                    f"Consider {second_best.model_name} if {dimension} is your top priority "
                    f"(significantly stronger in this area)."
                )
        
        return recommendations
    
    def _analyze_score_gaps(self, top_models: List[ModelScore]) -> Dict:
        """Analyze score gaps between top models"""
        if len(top_models) < 2:
            return {}
        
        gaps = []
        for i in range(len(top_models) - 1):
            gap = top_models[i].overall_score - top_models[i + 1].overall_score
            gaps.append({
                'rank_from': i + 1,
                'rank_to': i + 2,
                'score_gap': gap,
                'gap_significance': 'large' if gap > 0.2 else 'medium' if gap > 0.1 else 'small'
            })
        
        return {
            'gaps': gaps,
            'largest_gap': max(gaps, key=lambda x: x['score_gap']) if gaps else None,
            'close_competition': any(gap['score_gap'] < 0.05 for gap in gaps)
        }
    
    def explain_scoring_methodology(self) -> str:
        """Provide explanation of the scoring methodology"""
        explanation = """
        ## Scoring Methodology
        
        **1. Raw Score Collection:**
        - Reasoning: Model's cognitive strength (1-10 scale)
        - Latency: Performance category (real-time=1.0, interactive=0.7, batch=0.3)
        - Cost: Token pricing (lower cost = higher score)
        - Reliability: Tool integration quality (1-10 scale)
        
        **2. Normalization:**
        - All scores normalized to 0-1 range relative to viable models
        - Enables fair comparison across different measurement scales
        - Cost scores inverted (lower cost = higher normalized score)
        
        **3. Weighting:**
        - User priority weights applied to each dimension
        - Overall score = sum of (normalized_score × user_weight)
        - Weights sum to 1.0 for consistent scaling
        
        **4. Relative Scoring:**
        - Scores are comparative, not absolute
        - Rankings valid only within the current viable model set
        - Adding/removing models may change relative scores
        """
        
        return explanation