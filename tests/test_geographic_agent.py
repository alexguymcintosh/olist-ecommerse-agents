from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.geographic.geographic_agent import GeographicAgent
from utils.base_agent import BaseAgent


def _build_core_data() -> dict[str, pd.DataFrame]:
    orders = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(1, 13)],
            "customer_id": [f"c{i}" for i in range(1, 13)],
            "order_purchase_timestamp": [
                "2016-10-05",
                "2016-10-06",
                "2016-10-07",
                "2016-10-08",
                "2016-10-09",
                "2016-10-10",
                "2016-11-11",
                "2016-11-12",
                "2016-11-13",
                "2016-11-14",
                "2016-11-15",
                "2016-11-16",
            ],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(1, 13)],
            "customer_state": ["RJ"] * 12,
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(1, 13)],
            "order_item_id": [1] * 12,
            "product_id": ["p1"] * 12,
            "seller_id": ["s_rj"] * 6 + ["s_sp"] * 6,
            "price": [10.0] * 12,
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1"],
            "product_category_name": ["beleza_saude"],
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["s_rj", "s_sp"],
            "seller_state": ["RJ", "SP"],
        }
    )
    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "products": products,
        "sellers": sellers,
    }


def _categories_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_category_name": ["beleza_saude"],
            "product_category_name_english": ["health_beauty"],
        }
    )


def test_load_geographic_data_uses_utf8_sig_and_low_null_category_ratio(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_read_csv(path: Path, encoding: str | None = None) -> pd.DataFrame:
        seen["encoding"] = str(encoding)
        return _categories_df()

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    agent = GeographicAgent(data=_build_core_data())
    out = agent._load_geographic_data()

    assert seen["encoding"] == "utf-8-sig"
    assert out["product_category_name_english"].isna().mean() < 0.01


def test_load_geographic_data_uses_seller_state_for_supply_counts(monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = GeographicAgent(data=_build_core_data())
    merged = agent._load_geographic_data()

    predictions = [
        {
            "state": "RJ",
            "category": "health_beauty",
            "predicted_growth_pct": 0.2,
            "confidence": "HIGH",
            "confidence_score": 1.0,
            "reasoning": "test",
        }
    ]
    sparse_flags = {"RJ": {"health_beauty": False}}
    gaps = agent._compute_supply_gaps(merged, predictions, sparse_flags)
    assert gaps[0]["current_sellers"] == 1  # seller_state='RJ' only


def test_compute_growth_matrix_handles_missing_month_with_reindex() -> None:
    agent = GeographicAgent(data={})
    training = pd.DataFrame(
        {
            "customer_state": ["RJ"] * 20,
            "product_category_name_english": ["health_beauty"] * 20,
            "month": [pd.Period("2016-10", "M")] * 10 + [pd.Period("2016-12", "M")] * 10,
            "price": [10.0] * 20,
            "order_id": [f"o{i}" for i in range(20)],
        }
    )
    growth, _momentum, _orders, _flags = agent._compute_growth_matrix(
        training, ["RJ"], ["health_beauty"]
    )
    assert pd.isna(growth["RJ"]["health_beauty"])


def test_compute_growth_matrix_replaces_inf_with_nan() -> None:
    agent = GeographicAgent(data={})
    training = pd.DataFrame(
        {
            "customer_state": ["RJ"] * 20,
            "product_category_name_english": ["health_beauty"] * 20,
            "month": [pd.Period("2016-10", "M")] * 10 + [pd.Period("2016-11", "M")] * 10,
            "price": [0.0] * 10 + [10.0] * 10,  # 0 -> positive => inf before guard
            "order_id": [f"o{i}" for i in range(20)],
        }
    )
    growth, _momentum, _orders, _flags = agent._compute_growth_matrix(
        training, ["RJ"], ["health_beauty"]
    )
    assert pd.isna(growth["RJ"]["health_beauty"])


def test_sparse_month_sets_low_confidence() -> None:
    agent = GeographicAgent(data={})
    confidence, score = agent._score_confidence(0.15, latest_order_count=9)
    assert confidence == "LOW"
    assert score == 0.0


def test_supply_gap_ratio_formula(monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    data = _build_core_data()
    # Add more orders for denominator check.
    data["orders"] = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(1, 21)],
            "customer_id": [f"c{i}" for i in range(1, 21)],
            "order_purchase_timestamp": ["2016-11-01"] * 20,
        }
    )
    data["customers"] = pd.DataFrame(
        {"customer_id": [f"c{i}" for i in range(1, 21)], "customer_state": ["RJ"] * 20}
    )
    data["order_items"] = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(1, 21)],
            "order_item_id": [1] * 20,
            "product_id": ["p1"] * 20,
            "seller_id": ["s_rj"] * 20,
            "price": [10.0] * 20,
        }
    )
    data["sellers"] = pd.DataFrame({"seller_id": ["s_rj"], "seller_state": ["RJ"]})

    agent = GeographicAgent(data=data)
    merged = agent._load_geographic_data()
    pred = [
        {
            "state": "RJ",
            "category": "health_beauty",
            "predicted_growth_pct": 0.5,
            "confidence": "HIGH",
            "confidence_score": 1.0,
            "reasoning": "test",
        }
    ]
    gaps = agent._compute_supply_gaps(
        merged, pred, sparse_flags={"RJ": {"health_beauty": False}}
    )
    assert gaps[0]["predicted_order_volume"] == 30.0
    assert gaps[0]["supply_gap_ratio"] == 30.0


def test_rank_opportunities_uses_confidence_score_numeric() -> None:
    agent = GeographicAgent(data={})
    predictions = [
        {
            "state": "RJ",
            "category": "health_beauty",
            "predicted_growth_pct": 0.2,
            "confidence": "LOW",
            "confidence_score": 0.0,
            "reasoning": "test",
        },
        {
            "state": "SP",
            "category": "health_beauty",
            "predicted_growth_pct": 0.2,
            "confidence": "HIGH",
            "confidence_score": 1.0,
            "reasoning": "test",
        },
    ]
    gaps = [
        {
            "state": "RJ",
            "category": "health_beauty",
            "current_sellers": 1,
            "current_month_order_count": 10,
            "predicted_order_volume": 12.0,
            "supply_gap_ratio": 12.0,
            "supply_gap_severity": 1.0,
        },
        {
            "state": "SP",
            "category": "health_beauty",
            "current_sellers": 1,
            "current_month_order_count": 10,
            "predicted_order_volume": 12.0,
            "supply_gap_ratio": 12.0,
            "supply_gap_severity": 1.0,
        },
    ]
    ranked = agent._rank_opportunities(predictions, gaps)
    assert ranked[0]["state"] == "SP"
    assert isinstance(ranked[0]["composite_score"], float)


def test_run_returns_geographic_output_contract(monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    data = _build_core_data()
    agent = GeographicAgent(data=data)

    # Mock LLM method to avoid real API calls.
    monkeypatch.setattr(
        agent,
        "_predict_next_month_growth",
        lambda state, category, momentum: (0.1 if pd.notna(momentum) else 0.0, "mock"),
    )

    out = agent.run(
        iteration=2, training_window=("2016-10", "2016-11"), prediction_month="2016-12"
    )
    assert not issubclass(GeographicAgent, BaseAgent)
    assert out["agent"] == "geographic"
    assert isinstance(out["predictions"], list)
    assert isinstance(out["supply_gaps"], list)
    assert isinstance(out["ranked_opportunities"], list)
    assert isinstance(out["metrics"], dict)
    assert out["training_window"] == ("2016-10", "2016-11")
    assert out["prediction_month"] == "2016-12"
