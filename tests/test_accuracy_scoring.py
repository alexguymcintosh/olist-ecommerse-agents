from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from walk_forward import (
    _compute_aggregate_accuracy,
    _compute_directional_accuracy,
    _compute_pct_error,
    _json_safe,
    _score_iteration_predictions,
)


def _joined_df() -> pd.DataFrame:
    # Two months for RJ/health_beauty with >=10 orders each, plus one sparse month for SP.
    rj_prev = pd.DataFrame(
        {
            "customer_state": ["RJ"] * 10,
            "product_category_name_english": ["health_beauty"] * 10,
            "month": [pd.Period("2017-09", "M")] * 10,
            "order_id": [f"rj_prev_{i}" for i in range(10)],
            "price": [10.0] * 10,
        }
    )
    rj_curr = pd.DataFrame(
        {
            "customer_state": ["RJ"] * 10,
            "product_category_name_english": ["health_beauty"] * 10,
            "month": [pd.Period("2017-10", "M")] * 10,
            "order_id": [f"rj_curr_{i}" for i in range(10)],
            "price": [12.0] * 10,
        }
    )
    sp_sparse = pd.DataFrame(
        {
            "customer_state": ["SP"] * 9,
            "product_category_name_english": ["health_beauty"] * 9,
            "month": [pd.Period("2017-10", "M")] * 9,
            "order_id": [f"sp_curr_{i}" for i in range(9)],
            "price": [8.0] * 9,
        }
    )
    return pd.concat([rj_prev, rj_curr, sp_sparse], ignore_index=True)


def test_directional_accuracy_sign_match() -> None:
    assert _compute_directional_accuracy(0.10, 0.02) is True
    assert _compute_directional_accuracy(-0.10, -0.02) is True
    assert _compute_directional_accuracy(0.10, -0.02) is False


def test_pct_error_safe_division_guard() -> None:
    pct_error = _compute_pct_error(predicted_growth=0.10, actual_growth=0.0)
    assert pct_error == 10.0


def test_validation_handles_nan_and_sparse_rows() -> None:
    joined = _joined_df()
    predictions = [
        {
            "state": "RJ",
            "category": "health_beauty",
            "predicted_growth_pct": 0.10,
        },
        {
            "state": "SP",
            "category": "health_beauty",
            "predicted_growth_pct": 0.10,
        },
    ]
    validation = _score_iteration_predictions(
        joined_df=joined, prediction_month="2017-10", predictions=predictions
    )
    assert len(validation["items"]) == 2
    dense_item = next(i for i in validation["items"] if i["state"] == "RJ")
    sparse_item = next(i for i in validation["items"] if i["state"] == "SP")

    assert dense_item["actual_growth"] is not None
    assert isinstance(dense_item["directional_accuracy"], bool)
    assert isinstance(dense_item["pct_error"], float)

    assert sparse_item["actual_growth"] is None
    assert sparse_item["directional_accuracy"] is None
    assert sparse_item["pct_error"] is None


def test_aggregate_accuracy_computation() -> None:
    iterations = [
        {
            "iteration": 1,
            "training_window": ("2016-09", "2017-08"),
            "prediction_month": "2017-09",
            "previous_prediction": None,
            "predictions": [],
            "supply_gaps": [],
            "ranked_opportunities": [],
            "validation": {
                "items": [
                    {"directional_accuracy": True, "pct_error": 0.1},
                    {"directional_accuracy": False, "pct_error": 0.3},
                    {"directional_accuracy": None, "pct_error": None},
                ]
            },
        }
    ]
    agg = _compute_aggregate_accuracy(iterations)  # type: ignore[arg-type]
    assert agg["scored_predictions"] == 2
    assert agg["avg_directional_accuracy"] == 0.5
    assert agg["avg_pct_error"] == 0.2


def test_validation_payload_json_serializable() -> None:
    joined = _joined_df()
    predictions = [
        {
            "state": "SP",
            "category": "health_beauty",
            "predicted_growth_pct": 0.10,
        }
    ]
    payload = _score_iteration_predictions(
        joined_df=joined, prediction_month="2017-10", predictions=predictions
    )
    safe_payload = _json_safe(payload)
    json.dumps(safe_payload)
