import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.schema_geographic import (
    GeographicMetrics,
    GeographicOutput,
    Prediction,
    RankedOpportunity,
    SupplyGap,
    WalkForwardIteration,
    WalkForwardResult,
)


def _build_prediction() -> Prediction:
    return {
        "state": "RJ",
        "category": "health_beauty",
        "predicted_growth_pct": 0.12,
        "confidence": "HIGH",
        "confidence_score": 1.0,
        "reasoning": "Consistent recent growth and stable seller base.",
    }


def _build_supply_gap() -> SupplyGap:
    return {
        "state": "RJ",
        "category": "health_beauty",
        "current_sellers": 3,
        "current_month_order_count": 120,
        "predicted_order_volume": 134.4,
        "supply_gap_ratio": 44.8,
        "supply_gap_severity": 44.8,
    }


def _build_ranked_opportunity() -> RankedOpportunity:
    return {
        "rank": 1,
        "state": "RJ",
        "category": "health_beauty",
        "predicted_growth_pct": 0.12,
        "current_sellers": 3,
        "current_month_order_count": 120,
        "predicted_order_volume": 134.4,
        "supply_gap_ratio": 44.8,
        "supply_gap_severity": 44.8,
        "composite_score": 53.76,
        "recommended_action": "Onboard additional health_beauty sellers in RJ",
        "urgency": "HIGH",
    }


def _build_metrics() -> GeographicMetrics:
    return {
        "top5_states": ["SP", "RJ", "MG", "RS", "PR"],
        "top5_categories": [
            "health_beauty",
            "bed_bath_table",
            "sports_leisure",
            "watches_gifts",
            "computers_accessories",
        ],
        "growth_matrix": {"RJ": {"health_beauty": 0.12}},
        "momentum_scores": {"RJ": {"health_beauty": 0.10}},
        "seller_counts": {"RJ": {"health_beauty": 3}},
        "order_counts": {"RJ": {"health_beauty": 120}},
        "supply_gap_matrix": {"RJ": {"health_beauty": 44.8}},
    }


def test_typed_dicts_can_be_instantiated_and_json_serialized() -> None:
    prediction = _build_prediction()
    supply_gap = _build_supply_gap()
    ranked_opportunity = _build_ranked_opportunity()
    metrics = _build_metrics()

    geographic_output: GeographicOutput = {
        "agent": "geographic",
        "timestamp": "2026-04-17T00:00:00+00:00",
        "iteration": 1,
        "training_window": ("2016-09", "2017-09"),
        "prediction_month": "2017-10",
        "predictions": [prediction],
        "supply_gaps": [supply_gap],
        "ranked_opportunities": [ranked_opportunity],
        "metrics": metrics,
        "risk_flags": ["thin_volume_cell"],
    }

    walk_forward_iteration: WalkForwardIteration = {
        "iteration": 1,
        "training_window": ("2016-09", "2017-09"),
        "prediction_month": "2017-10",
        "previous_prediction": None,
        "predictions": [prediction],
        "supply_gaps": [supply_gap],
        "ranked_opportunities": [ranked_opportunity],
        "validation": {
            "predicted_growth": 0.12,
            "actual_growth": 0.10,
            "directional_accuracy": True,
            "pct_error": 0.02,
        },
    }

    walk_forward_result: WalkForwardResult = {
        "completed_iterations": 12,
        "total_iterations": 12,
        "iterations": [walk_forward_iteration],
        "aggregate_accuracy": {"directional_accuracy": 0.72, "avg_pct_error": 0.032},
    }

    json.dumps(prediction)
    json.dumps(supply_gap)
    json.dumps(ranked_opportunity)
    json.dumps(metrics)
    json.dumps(geographic_output)
    json.dumps(walk_forward_iteration)
    json.dumps(walk_forward_result)


def test_confidence_score_and_training_window_types() -> None:
    prediction = _build_prediction()
    geographic_output: GeographicOutput = {
        "agent": "geographic",
        "timestamp": "2026-04-17T00:00:00+00:00",
        "iteration": 1,
        "training_window": ("2016-09", "2017-09"),
        "prediction_month": "2017-10",
        "predictions": [prediction],
        "supply_gaps": [_build_supply_gap()],
        "ranked_opportunities": [_build_ranked_opportunity()],
        "metrics": _build_metrics(),
        "risk_flags": [],
    }

    walk_forward_iteration: WalkForwardIteration = {
        "iteration": 1,
        "training_window": ("2016-09", "2017-09"),
        "prediction_month": "2017-10",
        "previous_prediction": [prediction],
        "predictions": [prediction],
        "supply_gaps": [_build_supply_gap()],
        "ranked_opportunities": [_build_ranked_opportunity()],
        "validation": {},
    }

    assert isinstance(prediction["confidence_score"], float)
    assert not isinstance(prediction["confidence_score"], str)
    assert isinstance(geographic_output["training_window"], tuple)
    assert isinstance(walk_forward_iteration["training_window"], tuple)
