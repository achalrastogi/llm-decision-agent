"""
Decision Engine Agent
---------------------
Applies deterministic weighted scoring and computes transparent trade-offs.
NO LLM usage by design.
"""

from typing import Dict, List
import math
from models.schemas import UserConstraints, ModelInfo, ModelScore
from models.registry import ModelRegistry


class DecisionEngineAgent:
    """
    Applies weighted scoring to viable models and computes trade-offs.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

        # Latency categorical scores (FINAL, not normalized again)
        self.latency_scores = {
            "real-time": 1.0,
            "interactive": 0.7,
            "batch": 0.3,
        }

    # =====================================================
    # PUBLIC API
    # =====================================================
    def score_models(
        self,
        viable_model_ids: List[str],
        constraints: UserConstraints,
    ) -> List[ModelScore]:

        if not viable_model_ids:
            return []

        models = {
            model_id: self.registry.get_model(model_id)
            for model_id in viable_model_ids
        }

        raw_scores = self._calculate_raw_scores(models)
        normalized_scores = self._normalize_scores(raw_scores)

        weighted = self._apply_weights(
            normalized_scores, constraints.priority_weights
        )

        model_scores = []
        for model_id in viable_model_ids:
            explanations = self._generate_explanations(
                models[model_id],
                raw_scores[model_id],
                normalized_scores[model_id],
            )

            model_scores.append(
                ModelScore(
                    model_name=models[model_id].name,
                    overall_score=weighted[model_id]["overall"],
                    dimension_scores=normalized_scores[model_id],
                    explanations=explanations,
                )
            )

        model_scores.sort(key=lambda x: x.overall_score, reverse=True)
        return model_scores

    # =====================================================
    # RAW SCORES
    # =====================================================
    def _calculate_raw_scores(
        self, models: Dict[str, ModelInfo]
    ) -> Dict[str, Dict[str, float]]:

        raw = {}
        for model_id, m in models.items():
            raw[model_id] = {
                "reasoning": m.reasoning_strength,
                "latency": self.latency_scores.get(
                    m.latency_category.value, 0.5
                ),
                "cost": max(
                    m.cost_per_token.get("input", 0),
                    m.cost_per_token.get("output", 0),
                ),
                "reliability": m.tool_reliability,
            }
        return raw

    # =====================================================
    # NORMALIZATION (FAIR)
    # =====================================================
    def _normalize_scores(
        self, raw_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:

        dimensions = ["reasoning", "latency", "cost", "reliability"]
        min_max = {}

        for d in dimensions:
            values = [s[d] for s in raw_scores.values()]
            min_max[d] = (min(values), max(values))

        normalized = {}
        for model_id, scores in raw_scores.items():
            normalized[model_id] = {}

            for d in dimensions:
                min_v, max_v = min_max[d]

                if max_v == min_v:
                    norm = 1.0
                else:
                    if d == "cost":
                        # log-scale to avoid domination
                        norm = 1.0 - (
                            math.log(scores[d] + 1e-9)
                            - math.log(min_v + 1e-9)
                        ) / (
                            math.log(max_v + 1e-9)
                            - math.log(min_v + 1e-9)
                        )
                    else:
                        norm = (scores[d] - min_v) / (max_v - min_v)

                normalized[model_id][d] = round(norm, 4)

        return normalized

    # =====================================================
    # WEIGHT APPLICATION
    # =====================================================
    def _apply_weights(
        self,
        normalized_scores: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
    ) -> Dict[str, Dict]:

        weighted = {}
        for model_id, dims in normalized_scores.items():
            overall = 0.0
            for d, score in dims.items():
                overall += score * weights.get(d, 0.25)

            weighted[model_id] = {
                "overall": round(overall, 4)
            }

        return weighted

    # =====================================================
    # EXPLANATIONS (DETERMINISTIC)
    # =====================================================
    def _generate_explanations(
        self,
        model: ModelInfo,
        raw: Dict[str, float],
        norm: Dict[str, float],
    ) -> Dict[str, str]:

        return {
            "reasoning": (
                f"Reasoning strength {raw['reasoning']}/10 "
                f"(relative score {norm['reasoning']:.2f})."
            ),
            "latency": (
                f"Latency category '{model.latency_category.value}' "
                f"(relative score {norm['latency']:.2f})."
            ),
            "cost": (
                f"Max token cost ${raw['cost']:.6f} "
                f"(relative cost-efficiency {norm['cost']:.2f})."
            ),
            "reliability": (
                f"Reliability score {raw['reliability']}/10 "
                f"(relative score {norm['reliability']:.2f})."
            ),
        }

    # =====================================================
    # TRADE-OFF ANALYSIS (UNWEIGHTED)
    # =====================================================
    def generate_trade_off_analysis(
        self, model_scores: List[ModelScore]
    ) -> Dict:

        if len(model_scores) < 2:
            return {"trade_offs": []}

        best = model_scores[0]
        trade_offs = []

        for alt in model_scores[1:3]:
            trade = {
                "primary_model": best.model_name,
                "alternative_model": alt.model_name,
                "score_difference": round(
                    best.overall_score - alt.overall_score, 4
                ),
                "advantages": [],
                "disadvantages": [],
            }

            for d in ["reasoning", "latency", "cost", "reliability"]:
                diff = (
                    best.dimension_scores[d]
                    - alt.dimension_scores[d]
                )
                if abs(diff) > 0.05:
                    if diff > 0:
                        trade["advantages"].append(
                            {"dimension": d, "delta": round(diff, 4)}
                        )
                    else:
                        trade["disadvantages"].append(
                            {"dimension": d, "delta": round(abs(diff), 4)}
                        )

            trade_offs.append(trade)

        return {"trade_offs": trade_offs}
