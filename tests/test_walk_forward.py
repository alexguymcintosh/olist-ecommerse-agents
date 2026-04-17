from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from walk_forward import run_walk_forward


class MockGeographicAgent:
    def __init__(self, data=None, model=None) -> None:
        self.data = data
        self.model = model

    def _load_geographic_data(self) -> pd.DataFrame:
        months = pd.period_range("2016-09", "2018-09", freq="M")  # 25 months
        return pd.DataFrame(
            {
                "month": months,
                "order_id": [f"o{i}" for i in range(len(months))],
                "customer_state": ["RJ"] * len(months),
                "seller_state": ["RJ"] * len(months),
                "product_category_name_english": ["health_beauty"] * len(months),
                "price": [10.0] * len(months),
            }
        )

    def run(
        self,
        training_df: pd.DataFrame | None = None,
        iteration: int = 1,
        training_window: tuple[str, str] = ("", ""),
        prediction_month: str = "",
    ) -> dict:
        prediction = {
            "state": "RJ",
            "category": "health_beauty",
            "predicted_growth_pct": 0.1 + iteration / 1000.0,
            "confidence": "HIGH",
            "confidence_score": 1.0,
            "reasoning": "mock",
        }
        supply_gap = {
            "state": "RJ",
            "category": "health_beauty",
            "current_sellers": 1,
            "current_month_order_count": int(training_df["order_id"].nunique()) if training_df is not None else 0,
            "predicted_order_volume": 12.0,
            "supply_gap_ratio": 12.0,
            "supply_gap_severity": 12.0,
        }
        ranked = {
            "rank": 1,
            "state": "RJ",
            "category": "health_beauty",
            "predicted_growth_pct": prediction["predicted_growth_pct"],
            "current_sellers": 1,
            "current_month_order_count": supply_gap["current_month_order_count"],
            "predicted_order_volume": 12.0,
            "supply_gap_ratio": 12.0,
            "supply_gap_severity": 12.0,
            "composite_score": 1.2,
            "recommended_action": "mock action",
            "urgency": "HIGH",
        }
        return {
            "agent": "geographic",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "iteration": iteration,
            "training_window": training_window,
            "prediction_month": prediction_month,
            "predictions": [prediction],
            "supply_gaps": [supply_gap],
            "ranked_opportunities": [ranked],
            "metrics": {
                "top5_states": ["RJ"],
                "top5_categories": ["health_beauty"],
                "growth_matrix": {"RJ": {"health_beauty": 0.1}},
                "momentum_scores": {"RJ": {"health_beauty": 0.1}},
                "seller_counts": {"RJ": {"health_beauty": 1}},
                "order_counts": {"RJ": {"health_beauty": 12}},
                "supply_gap_matrix": {"RJ": {"health_beauty": 12.0}},
            },
            "risk_flags": [],
        }


def test_walk_forward_runs_three_iterations(tmp_path: Path) -> None:
    result = run_walk_forward(
        start=1,
        end=3,
        validate=True,
        agent_class=MockGeographicAgent,
        output_dir=tmp_path,
    )
    assert result["completed_iterations"] == 3
    assert len(result["iterations"]) == 3


def test_walk_forward_carries_previous_prediction(tmp_path: Path) -> None:
    result = run_walk_forward(
        start=1,
        end=3,
        validate=True,
        agent_class=MockGeographicAgent,
        output_dir=tmp_path,
    )
    assert result["iterations"][0]["previous_prediction"] is None
    assert (
        result["iterations"][1]["previous_prediction"]
        == result["iterations"][0]["predictions"]
    )
    assert (
        result["iterations"][2]["previous_prediction"]
        == result["iterations"][1]["predictions"]
    )


def test_walk_forward_emits_training_window_tuple(tmp_path: Path) -> None:
    result = run_walk_forward(
        start=1,
        end=3,
        validate=True,
        agent_class=MockGeographicAgent,
        output_dir=tmp_path,
    )
    for iteration in result["iterations"]:
        assert isinstance(iteration["training_window"], tuple)
        assert len(iteration["training_window"]) == 2
        assert all(isinstance(x, str) for x in iteration["training_window"])


def test_walk_forward_json_serializable(tmp_path: Path) -> None:
    result = run_walk_forward(
        start=1,
        end=3,
        validate=True,
        agent_class=MockGeographicAgent,
        output_dir=tmp_path,
    )
    json.dumps(result)
    files = list(tmp_path.glob("walk_forward_*.json"))
    assert len(files) == 1
    loaded = json.loads(files[0].read_text(encoding="utf-8"))
    assert loaded["completed_iterations"] == 3


def test_walk_forward_respects_start_end_bounds(tmp_path: Path) -> None:
    result = run_walk_forward(
        start=2,
        end=4,
        validate=True,
        agent_class=MockGeographicAgent,
        output_dir=tmp_path,
    )
    assert result["completed_iterations"] == 3
    assert [it["iteration"] for it in result["iterations"]] == [2, 3, 4]
