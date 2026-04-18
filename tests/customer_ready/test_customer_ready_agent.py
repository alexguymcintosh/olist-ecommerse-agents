from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.customer_ready.customer_ready_agent import CustomerReadinessAgent


def _categories_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_category_name": ["beleza_saude"],
            "product_category_name_english": ["health_beauty"],
        }
    )


def _build_data(
    orders: list[dict[str, object]],
    customers: list[dict[str, object]],
    order_items: list[dict[str, object]],
    payments: list[dict[str, object]],
) -> dict[str, pd.DataFrame]:
    return {
        "orders": pd.DataFrame(orders),
        "customers": pd.DataFrame(customers),
        "order_items": pd.DataFrame(order_items),
        "products": pd.DataFrame(
            [{"product_id": "p1", "product_category_name": "beleza_saude"}]
        ),
        "payments": pd.DataFrame(payments),
    }


def _llm_ok(*_args, **_kwargs) -> str:
    return '{"readiness":"HIGH","reasoning":"solid demand","risk_flags":[]}'


def _base_records() -> dict[str, list[dict[str, object]]]:
    return {
        "orders": [
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_purchase_timestamp": "2018-01-10",
            }
        ],
        "customers": [
            {"customer_id": "c1", "customer_unique_id": "u1", "customer_state": "RJ"}
        ],
        "order_items": [
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 100.0,
            }
        ],
        "payments": [
            {
                "order_id": "o1",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 6,
                "payment_value": 100.0,
            }
        ],
    }


def test_run_returns_25_outputs_for_focus_pairs(monkeypatch) -> None:
    records = _base_records()
    data = _build_data(
        orders=records["orders"],
        customers=records["customers"],
        order_items=records["order_items"],
        payments=records["payments"],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok)
    out = agent.run()
    assert len(out) == 25


def test_readiness_values_constrained_to_high_medium_low(monkeypatch) -> None:
    records = _base_records()
    data = _build_data(
        orders=records["orders"],
        customers=records["customers"],
        order_items=records["order_items"],
        payments=records["payments"],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok)
    out = agent.run(states=["RJ"], categories=["health_beauty"])
    assert out[0]["readiness"] in {"HIGH", "MEDIUM", "LOW"}


def test_payments_are_aggregated_per_order_before_join(monkeypatch) -> None:
    data = _build_data(
        orders=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_purchase_timestamp": "2018-01-10",
            }
        ],
        customers=[
            {"customer_id": "c1", "customer_unique_id": "u1", "customer_state": "RJ"}
        ],
        order_items=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 100.0,
            }
        ],
        payments=[
            {
                "order_id": "o1",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 60.0,
            },
            {
                "order_id": "o1",
                "payment_sequential": 2,
                "payment_type": "voucher",
                "payment_installments": 1,
                "payment_value": 40.0,
            },
        ],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok)
    out = agent.run(states=["RJ"], categories=["health_beauty"])
    assert out[0]["avg_spend"] == 100.0


def test_money_metrics_use_per_order_not_order_item_rows(monkeypatch) -> None:
    data = _build_data(
        orders=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_purchase_timestamp": "2018-01-10",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_purchase_timestamp": "2018-01-11",
            },
        ],
        customers=[
            {"customer_id": "c1", "customer_unique_id": "u1", "customer_state": "RJ"},
            {"customer_id": "c2", "customer_unique_id": "u2", "customer_state": "RJ"},
        ],
        order_items=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 30.0,
            },
            {
                "order_id": "o1",
                "order_item_id": 2,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 30.0,
            },
            {
                "order_id": "o1",
                "order_item_id": 3,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 30.0,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 30.0,
            },
        ],
        payments=[
            {
                "order_id": "o1",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 90.0,
            },
            {
                "order_id": "o2",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 30.0,
            },
        ],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok)
    out = agent.run(states=["RJ"], categories=["health_beauty"])
    assert out[0]["avg_spend"] == 60.0


def test_installment_pct_uses_credit_card_only(monkeypatch) -> None:
    data = _build_data(
        orders=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_purchase_timestamp": "2018-01-10",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_purchase_timestamp": "2018-01-11",
            },
        ],
        customers=[
            {"customer_id": "c1", "customer_unique_id": "u1", "customer_state": "RJ"},
            {"customer_id": "c2", "customer_unique_id": "u2", "customer_state": "RJ"},
        ],
        order_items=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 40.0,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 60.0,
            },
        ],
        payments=[
            {
                "order_id": "o1",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 6,
                "payment_value": 40.0,
            },
            {
                "order_id": "o2",
                "payment_sequential": 1,
                "payment_type": "boleto",
                "payment_installments": 1,
                "payment_value": 60.0,
            },
        ],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok)
    out = agent.run(states=["RJ"], categories=["health_beauty"])
    assert out[0]["installment_pct"] == 1.0


def test_order_volume_trend_uses_distinct_order_count(monkeypatch) -> None:
    data = _build_data(
        orders=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_purchase_timestamp": "2018-01-10",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_purchase_timestamp": "2018-02-10",
            },
            {
                "order_id": "o3",
                "customer_id": "c3",
                "order_purchase_timestamp": "2018-02-11",
            },
        ],
        customers=[
            {"customer_id": "c1", "customer_unique_id": "u1", "customer_state": "RJ"},
            {"customer_id": "c2", "customer_unique_id": "u2", "customer_state": "RJ"},
            {"customer_id": "c3", "customer_unique_id": "u3", "customer_state": "RJ"},
        ],
        order_items=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 10.0,
            },
            {
                "order_id": "o1",
                "order_item_id": 2,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 10.0,
            },
            {
                "order_id": "o1",
                "order_item_id": 3,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 10.0,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 10.0,
            },
            {
                "order_id": "o3",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 10.0,
            },
        ],
        payments=[
            {
                "order_id": "o1",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 30.0,
            },
            {
                "order_id": "o2",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 10.0,
            },
            {
                "order_id": "o3",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 10.0,
            },
        ],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok)
    out = agent.run(states=["RJ"], categories=["health_beauty"])
    assert out[0]["order_volume_trend"] == 1.0


def test_repeat_rate_zero_is_valid_output(monkeypatch) -> None:
    data = _build_data(
        orders=[
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_purchase_timestamp": "2018-01-10",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_purchase_timestamp": "2018-01-11",
            },
        ],
        customers=[
            {"customer_id": "c1", "customer_unique_id": "u1", "customer_state": "RJ"},
            {"customer_id": "c2", "customer_unique_id": "u2", "customer_state": "RJ"},
        ],
        order_items=[
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 50.0,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 60.0,
            },
        ],
        payments=[
            {
                "order_id": "o1",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 50.0,
            },
            {
                "order_id": "o2",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 60.0,
            },
        ],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok)
    out = agent.run(states=["RJ"], categories=["health_beauty"])
    assert out[0]["repeat_rate"] == 0.0


def test_llm_parse_failure_returns_agent_failed_flag(monkeypatch) -> None:
    records = _base_records()
    data = _build_data(
        orders=records["orders"],
        customers=records["customers"],
        order_items=records["order_items"],
        payments=records["payments"],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())
    agent = CustomerReadinessAgent(data=data, llm_client=lambda *_a, **_k: "not-json")
    out = agent.run(states=["RJ"], categories=["health_beauty"])
    assert out[0]["risk_flags"] == ["agent_failed"]
    assert out[0]["readiness"] == "MEDIUM"


def test_memory_write_maps_to_customer_readiness_columns(monkeypatch) -> None:
    records = _base_records()
    data = _build_data(
        orders=records["orders"],
        customers=records["customers"],
        order_items=records["order_items"],
        payments=records["payments"],
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: _categories_df())

    class FakeMemory:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, dict[str, object]]] = []

        def write_row(
            self, state: str, category: str, month: str, **cols: object
        ) -> None:
            self.calls.append((state, category, month, cols))

    memory = FakeMemory()
    agent = CustomerReadinessAgent(data=data, llm_client=_llm_ok, memory=memory)
    out = agent.run(states=["RJ"], categories=["health_beauty"], month="2018-01")
    assert len(memory.calls) == 1
    state, category, month, cols = memory.calls[0]
    assert (state, category, month) == ("RJ", "health_beauty", "2018-01")
    assert set(cols.keys()) == {
        "cr_avg_spend",
        "cr_order_volume_trend",
        "cr_top_payment_type",
        "cr_high_value_customer_count",
        "cr_repeat_rate",
        "cr_reasoning",
    }
    assert cols["cr_avg_spend"] == out[0]["avg_spend"]
