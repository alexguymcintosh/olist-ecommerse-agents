from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.supply_quality.supply_quality_agent import SupplyQualityAgent
from utils.config import FOCUS_CATEGORIES, FOCUS_STATES
from utils.data_loader import load_all


def _build_real_training_df() -> pd.DataFrame:
    data = load_all()
    orders = data["orders"].copy()
    order_items = data["order_items"].copy()
    sellers = data["sellers"].copy()
    reviews = data["reviews"].copy()
    products = data["products"].copy()
    categories = data["categories"].copy()

    merged = (
        order_items.merge(orders, on="order_id", how="inner")
        .merge(sellers, on="seller_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(categories, on="product_category_name", how="left")
    )
    merged["order_purchase_timestamp"] = pd.to_datetime(
        merged["order_purchase_timestamp"], errors="coerce"
    )
    merged["order_delivered_customer_date"] = pd.to_datetime(
        merged["order_delivered_customer_date"], errors="coerce"
    )
    merged["month"] = merged["order_purchase_timestamp"].dt.to_period("M")
    merged = merged.dropna(subset=["month"])
    # Use real history up to a stable month boundary for deterministic smoke behavior.
    return merged[merged["month"] <= pd.Period("2018-08", freq="M")].copy()


def test_supply_quality_smoke_real_data_mocked_llm_25_outputs_no_nan(
    monkeypatch,
) -> None:
    training_df = _build_real_training_df()
    agent = SupplyQualityAgent(data={})

    monkeypatch.setattr(
        "agents.supply_quality.supply_quality_agent.query_llm",
        lambda *_a, **_k: (
            '{"supply_confidence":"ADEQUATE","reasoning":"mocked assessment","risk_flags":[]}'
        ),
    )

    outputs = agent.run(
        training_df=training_df,
        states=FOCUS_STATES,
        categories=FOCUS_CATEGORIES,
        month="2018-09",
    )

    assert len(outputs) == 25
    for output in outputs:
        assert not pd.isna(output["seller_count"])
        assert not pd.isna(output["avg_review_score"])
        assert not pd.isna(output["avg_delivery_days"])
        assert not pd.isna(output["churn_rate"])
        assert not pd.isna(output["seller_concentration"])
