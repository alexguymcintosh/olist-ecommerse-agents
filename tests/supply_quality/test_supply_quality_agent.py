from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.supply_quality.supply_quality_agent import SupplyQualityAgent
from utils.config import FOCUS_CATEGORIES, FOCUS_STATES


def _build_row(
    *,
    order_id: str,
    seller_id: str,
    seller_state: str,
    category: str,
    purchase_ts: str,
    delivered_ts: str | None,
    order_status: str = "delivered",
    review_score: float | None = 4.0,
    price: float = 100.0,
) -> dict:
    return {
        "order_id": order_id,
        "seller_id": seller_id,
        "seller_state": seller_state,
        "product_category_name_english": category,
        "order_purchase_timestamp": purchase_ts,
        "order_delivered_customer_date": delivered_ts,
        "order_status": order_status,
        "review_score": review_score,
        "price": price,
    }


def _build_focus_grid_df() -> pd.DataFrame:
    rows: list[dict] = []
    months = ["2018-04-03", "2018-05-03", "2018-06-03", "2018-07-03", "2018-08-03"]
    order_seq = 0
    for state in FOCUS_STATES:
        for category in FOCUS_CATEGORIES:
            for month_idx, date_str in enumerate(months):
                order_count = 10 if month_idx == len(months) - 1 else 2
                for i in range(order_count):
                    order_seq += 1
                    rows.append(
                        _build_row(
                            order_id=f"o{order_seq}",
                            seller_id=f"{state}_{category}_s{(i % 2) + 1}",
                            seller_state=state,
                            category=category,
                            purchase_ts=date_str,
                            delivered_ts="2018-08-10",
                            review_score=4.0,
                            price=100.0 + i,
                        )
                    )
    return pd.DataFrame(rows)


def test_run_returns_25_outputs_for_focus_grid(monkeypatch) -> None:
    df = _build_focus_grid_df()
    agent = SupplyQualityAgent(data={})
    monkeypatch.setattr(
        "agents.supply_quality.supply_quality_agent.query_llm",
        lambda *_a, **_k: '{"supply_confidence":"ADEQUATE","reasoning":"ok","risk_flags":[]}',
    )
    outputs = agent.run(df, FOCUS_STATES, FOCUS_CATEGORIES, month="2018-09")
    assert len(outputs) == 25


def test_output_matches_supply_quality_output_contract(monkeypatch) -> None:
    df = _build_focus_grid_df()
    agent = SupplyQualityAgent(data={})
    monkeypatch.setattr(
        "agents.supply_quality.supply_quality_agent.query_llm",
        lambda *_a, **_k: '{"supply_confidence":"STRONG","reasoning":"healthy","risk_flags":[]}',
    )
    output = agent.run(df, FOCUS_STATES, FOCUS_CATEGORIES, month="2018-09")[0]
    assert set(output.keys()) == {
        "agent",
        "timestamp",
        "state",
        "category",
        "month",
        "seller_count",
        "avg_review_score",
        "avg_delivery_days",
        "churn_risk",
        "churn_rate",
        "top_seller_id",
        "seller_concentration",
        "supply_confidence",
        "reasoning",
        "risk_flags",
    }


def test_avg_review_score_uses_sentinel_when_no_reviews(monkeypatch) -> None:
    df = _build_focus_grid_df()
    mask = (df["seller_state"] == "SP") & (
        df["product_category_name_english"] == "health_beauty"
    )
    df.loc[mask, "review_score"] = None
    agent = SupplyQualityAgent(data={})
    monkeypatch.setattr(
        "agents.supply_quality.supply_quality_agent.query_llm",
        lambda *_a, **_k: '{"supply_confidence":"ADEQUATE","reasoning":"ok","risk_flags":[]}',
    )
    outputs = agent.run(df, FOCUS_STATES, FOCUS_CATEGORIES, month="2018-09")
    out = next(
        x
        for x in outputs
        if x["state"] == "SP" and x["category"] == "health_beauty"
    )
    assert out["avg_review_score"] == 3.0


def test_churn_risk_threshold_mapping() -> None:
    agent = SupplyQualityAgent(data={})
    assert agent._map_churn_risk(0.31) == "HIGH"
    assert agent._map_churn_risk(0.10) == "MEDIUM"
    assert agent._map_churn_risk(0.05) == "LOW"


def test_seller_count_zero_adds_critical_seller_gap() -> None:
    df = _build_focus_grid_df()
    df = df[
        ~(
            (df["seller_state"] == "RJ")
            & (df["product_category_name_english"] == "sports_leisure")
        )
    ].copy()
    agent = SupplyQualityAgent(data={})
    outputs = agent.run(df, FOCUS_STATES, FOCUS_CATEGORIES, month="2018-09")
    out = next(
        x for x in outputs if x["state"] == "RJ" and x["category"] == "sports_leisure"
    )
    assert out["seller_count"] == 0
    assert "critical_seller_gap" in out["risk_flags"]


def test_churn_rate_above_half_adds_high_seller_churn(monkeypatch) -> None:
    df = _build_focus_grid_df()
    pair_mask = (df["seller_state"] == "MG") & (
        df["product_category_name_english"] == "watches_gifts"
    )
    # Add extra historical months so churn logic has >= 7 months of history.
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    _build_row(
                        order_id="extra_churn_1",
                        seller_id="mg_s2",
                        seller_state="MG",
                        category="watches_gifts",
                        purchase_ts="2018-02-03",
                        delivered_ts="2018-02-10",
                    ),
                    _build_row(
                        order_id="extra_churn_2",
                        seller_id="mg_s3",
                        seller_state="MG",
                        category="watches_gifts",
                        purchase_ts="2018-03-03",
                        delivered_ts="2018-03-10",
                    ),
                    _build_row(
                        order_id="extra_churn_3",
                        seller_id="mg_s4",
                        seller_state="MG",
                        category="watches_gifts",
                        purchase_ts="2018-03-05",
                        delivered_ts="2018-03-12",
                    ),
                    _build_row(
                        order_id="extra_churn_4",
                        seller_id="mg_s5",
                        seller_state="MG",
                        category="watches_gifts",
                        purchase_ts="2018-02-07",
                        delivered_ts="2018-02-14",
                    ),
                ]
            ),
        ],
        ignore_index=True,
    )
    pair_mask = (df["seller_state"] == "MG") & (
        df["product_category_name_english"] == "watches_gifts"
    )
    # Ensure old window has multiple sellers, while previous/latest are single seller.
    old_window_mask = pair_mask & (
        pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M").isin(
            [pd.Period("2018-02", freq="M"), pd.Period("2018-03", freq="M")]
        )
    )
    old_indices = df[old_window_mask].index.tolist()
    old_sellers = ["mg_s2", "mg_s3", "mg_s4", "mg_s5"]
    for idx, seller_id in enumerate(old_sellers):
        if idx < len(old_indices):
            df.loc[old_indices[idx], "seller_id"] = seller_id
    prev_month_mask = pair_mask & (
        pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M")
        == pd.Period("2018-07", freq="M")
    )
    df.loc[prev_month_mask, "seller_id"] = "mg_s1"
    latest_mask = pair_mask & (
        pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M")
        == pd.Period("2018-08", freq="M")
    )
    df.loc[latest_mask, "seller_id"] = "mg_s1"

    agent = SupplyQualityAgent(data={})
    monkeypatch.setattr(
        "agents.supply_quality.supply_quality_agent.query_llm",
        lambda *_a, **_k: '{"supply_confidence":"WEAK","reasoning":"attrition","risk_flags":[]}',
    )
    outputs = agent.run(df, FOCUS_STATES, FOCUS_CATEGORIES, month="2018-09")
    out = next(x for x in outputs if x["state"] == "MG" and x["category"] == "watches_gifts")
    assert out["churn_rate"] > 0.5
    assert "high_seller_churn" in out["risk_flags"]


def test_sparse_pair_returns_sparse_data_and_skips_llm(monkeypatch) -> None:
    df = _build_focus_grid_df()
    pair_mask = (df["seller_state"] == "PR") & (
        df["product_category_name_english"] == "computers_accessories"
    )
    latest_mask = pair_mask & (
        pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M")
        == pd.Period("2018-08", freq="M")
    )
    # Make latest-month order count below MIN_MONTHLY_ORDERS.
    latest_rows = df[latest_mask].iloc[2:].index
    df = df.drop(index=latest_rows).copy()

    called = {"count": 0}

    def _fake_query(*_a, **_k):
        called["count"] += 1
        return '{"supply_confidence":"ADEQUATE","reasoning":"ok","risk_flags":[]}'

    monkeypatch.setattr("agents.supply_quality.supply_quality_agent.query_llm", _fake_query)
    agent = SupplyQualityAgent(data={})
    outputs = agent.run(df, FOCUS_STATES, FOCUS_CATEGORIES, month="2018-09")
    out = next(
        x
        for x in outputs
        if x["state"] == "PR" and x["category"] == "computers_accessories"
    )
    assert "sparse_data" in out["risk_flags"]
    assert out["supply_confidence"] == "WEAK"
    # One batch call covers all 24 non-sparse pairs (the 25th is sparse and skipped).
    assert called["count"] == 1


def test_llm_parse_failure_returns_agent_failed_fallback(monkeypatch) -> None:
    df = _build_focus_grid_df()
    monkeypatch.setattr(
        "agents.supply_quality.supply_quality_agent.query_llm",
        lambda *_a, **_k: "not-json",
    )
    agent = SupplyQualityAgent(data={})
    outputs = agent.run(df, FOCUS_STATES, FOCUS_CATEGORIES, month="2018-09")
    sample = outputs[0]
    assert sample["supply_confidence"] == "WEAK"
    assert "agent_failed" in sample["risk_flags"]


def test_top_seller_and_concentration_safe_guard() -> None:
    agent = SupplyQualityAgent(data={})
    empty = pd.DataFrame(columns=["seller_id", "price", "order_id"])
    top_seller, concentration = agent._compute_top_seller_and_concentration(empty)
    assert top_seller == ""
    assert concentration == 0.0
