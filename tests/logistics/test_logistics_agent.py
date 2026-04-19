from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.logistics.logistics_agent import LogisticsAgent


def _build_data(
    *,
    order_rows: list[dict[str, object]] | None = None,
    item_rows: list[dict[str, object]] | None = None,
    customer_rows: list[dict[str, object]] | None = None,
    seller_rows: list[dict[str, object]] | None = None,
    product_rows: list[dict[str, object]] | None = None,
    category_rows: list[dict[str, object]] | None = None,
) -> dict[str, pd.DataFrame]:
    orders = order_rows or [
        {
            "order_id": "o1",
            "customer_id": "c1",
            "order_status": "delivered",
            "order_purchase_timestamp": "2018-07-01",
            "order_delivered_customer_date": "2018-07-04",
            "order_estimated_delivery_date": "2018-07-06",
        }
    ]
    order_items = item_rows or [
        {
            "order_id": "o1",
            "order_item_id": 1,
            "product_id": "p1",
            "seller_id": "s1",
            "price": 100.0,
            "freight_value": 10.0,
        }
    ]
    customers = customer_rows or [{"customer_id": "c1", "customer_state": "RJ"}]
    sellers = seller_rows or [{"seller_id": "s1", "seller_state": "SP"}]
    products = product_rows or [{"product_id": "p1", "product_category_name": "cat_hb"}]
    categories = category_rows or [
        {
            "product_category_name": "cat_hb",
            "product_category_name_english": "health_beauty",
        }
    ]
    return {
        "orders": pd.DataFrame(orders),
        "order_items": pd.DataFrame(order_items),
        "customers": pd.DataFrame(customers),
        "sellers": pd.DataFrame(sellers),
        "products": pd.DataFrame(products),
        "categories": pd.DataFrame(categories),
    }


class _MemoryStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def write_row(self, state: str, category: str, month: str, **kwargs: object) -> None:
        self.calls.append(
            {
                "state": state,
                "category": category,
                "month": month,
                "kwargs": kwargs,
            }
        )


def test_run_returns_25_outputs_for_focus_grid(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.logistics.logistics_agent.query_llm",
        lambda *_args, **_kwargs: '{"feasibility":"ADEQUATE","reasoning":"ok","risk_flags":[]}',
    )
    memory = _MemoryStub()
    agent = LogisticsAgent(data=_build_data(), memory=memory)

    outputs = agent.run(month="2018-08")

    assert len(outputs) == 25
    assert len(memory.calls) == 25
    assert set(memory.calls[0]["kwargs"].keys()) == {
        "log_avg_delivery_days",
        "log_pct_on_time",
        "log_freight_ratio",
        "log_fastest_seller_state",
        "log_reasoning",
    }


def test_delivered_filter_applied_before_time_metrics(monkeypatch) -> None:
    data = _build_data(
        order_rows=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-07-01",
                "order_delivered_customer_date": "2018-07-03",
                "order_estimated_delivery_date": "2018-07-05",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_status": "canceled",
                "order_purchase_timestamp": "2018-07-01",
                "order_delivered_customer_date": None,
                "order_estimated_delivery_date": "2018-07-07",
            },
        ],
        item_rows=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 50.0,
                "freight_value": 5.0,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 50.0,
                "freight_value": 5.0,
            },
        ],
        customer_rows=[
            {"customer_id": "c1", "customer_state": "RJ"},
            {"customer_id": "c2", "customer_state": "RJ"},
        ],
    )
    agent = LogisticsAgent(data=data)

    def _checked_metrics(delivered_df: pd.DataFrame, state: str, category: str) -> dict[str, float | str]:
        assert delivered_df["order_status"].eq("delivered").all()
        return {
            "avg_delivery_days": 0.0,
            "pct_on_time": 0.0,
            "freight_ratio": 0.0,
            "fastest_seller_state": "",
            "delivery_variance": 0.0,
            "cross_state_dependency": 0.0,
        }

    monkeypatch.setattr(agent, "_compute_metrics", _checked_metrics)
    monkeypatch.setattr(
        "agents.logistics.logistics_agent.query_llm",
        lambda *_a, **_k: "[]",
    )
    agent.run(month="2018-08")


def test_avg_delivery_days_scoped_to_customer_state_and_category() -> None:
    data = _build_data(
        order_rows=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-07-01",
                "order_delivered_customer_date": "2018-07-05",
                "order_estimated_delivery_date": "2018-07-06",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-07-01",
                "order_delivered_customer_date": "2018-07-07",
                "order_estimated_delivery_date": "2018-07-08",
            },
            {
                "order_id": "o3",
                "customer_id": "c3",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-07-01",
                "order_delivered_customer_date": "2018-07-30",
                "order_estimated_delivery_date": "2018-08-01",
            },
        ],
        item_rows=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 10.0,
                "freight_value": 1.0,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 10.0,
                "freight_value": 1.0,
            },
            {
                "order_id": "o3",
                "order_item_id": 1,
                "product_id": "p2",
                "seller_id": "s1",
                "price": 10.0,
                "freight_value": 1.0,
            },
        ],
        customer_rows=[
            {"customer_id": "c1", "customer_state": "RJ"},
            {"customer_id": "c2", "customer_state": "RJ"},
            {"customer_id": "c3", "customer_state": "SP"},
        ],
        product_rows=[
            {"product_id": "p1", "product_category_name": "cat_hb"},
            {"product_id": "p2", "product_category_name": "cat_other"},
        ],
        category_rows=[
            {
                "product_category_name": "cat_hb",
                "product_category_name_english": "health_beauty",
            },
            {
                "product_category_name": "cat_other",
                "product_category_name_english": "sports_leisure",
            },
        ],
    )
    agent = LogisticsAgent(data=data)
    joined = agent._load_joined_data()
    delivered = joined[joined["order_status"] == "delivered"].copy()

    metrics = agent._compute_metrics(delivered, "RJ", "health_beauty")

    assert metrics["avg_delivery_days"] == 5.0


def test_pct_on_time_defaults_zero_when_no_delivered_orders() -> None:
    agent = LogisticsAgent(data=_build_data())
    joined = agent._load_joined_data()
    delivered = joined[joined["order_status"] == "delivered"].copy()

    metrics = agent._compute_metrics(delivered, "MG", "watches_gifts")

    assert metrics["pct_on_time"] == 0.0


def test_freight_ratio_uses_price_floor_guard() -> None:
    data = _build_data(
        item_rows=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 0.0,
                "freight_value": 5.0,
            }
        ]
    )
    agent = LogisticsAgent(data=data)
    joined = agent._load_joined_data()
    delivered = joined[joined["order_status"] == "delivered"].copy()

    metrics = agent._compute_metrics(delivered, "RJ", "health_beauty")

    assert metrics["freight_ratio"] == 500.0


def test_fastest_seller_state_defaults_empty_when_no_delivered_rows() -> None:
    agent = LogisticsAgent(data=_build_data())
    joined = agent._load_joined_data()
    delivered = joined[joined["order_status"] == "delivered"].copy()

    metrics = agent._compute_metrics(delivered, "PR", "computers_accessories")

    assert metrics["fastest_seller_state"] == ""


def test_delivery_variance_is_zero_with_single_delivered_order() -> None:
    agent = LogisticsAgent(data=_build_data())
    joined = agent._load_joined_data()
    delivered = joined[joined["order_status"] == "delivered"].copy()

    metrics = agent._compute_metrics(delivered, "RJ", "health_beauty")

    assert metrics["delivery_variance"] == 0.0


def test_cross_state_dependency_uses_seller_vs_customer_state() -> None:
    data = _build_data(
        order_rows=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-07-01",
                "order_delivered_customer_date": "2018-07-04",
                "order_estimated_delivery_date": "2018-07-06",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-07-01",
                "order_delivered_customer_date": "2018-07-04",
                "order_estimated_delivery_date": "2018-07-06",
            },
        ],
        item_rows=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 50.0,
                "freight_value": 5.0,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s2",
                "price": 50.0,
                "freight_value": 5.0,
            },
        ],
        customer_rows=[
            {"customer_id": "c1", "customer_state": "RJ"},
            {"customer_id": "c2", "customer_state": "RJ"},
        ],
        seller_rows=[
            {"seller_id": "s1", "seller_state": "RJ"},
            {"seller_id": "s2", "seller_state": "SP"},
        ],
    )
    agent = LogisticsAgent(data=data)
    joined = agent._load_joined_data()
    delivered = joined[joined["order_status"] == "delivered"].copy()

    metrics = agent._compute_metrics(delivered, "RJ", "health_beauty")

    assert metrics["cross_state_dependency"] == 0.5


def test_llm_parse_failure_returns_agent_failed_risk_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.logistics.logistics_agent.query_llm",
        lambda *_args, **_kwargs: "non-json response",
    )
    agent = LogisticsAgent(data=_build_data())

    outputs = agent.run(month="2018-08")

    assert len(outputs) == 25
    assert all(output["risk_flags"] == ["agent_failed"] for output in outputs)
    assert all(output["feasibility"] == "WEAK" for output in outputs)
