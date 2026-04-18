"""Integration smoke tests for shared memory table."""

from utils.memory import Memory, SCHEMA_COLUMNS


def test_memory_smoke_all_agent_columns_roundtrip(tmp_path) -> None:
    memory = Memory(tmp_path / "memory_story3.db")
    state, category, month = "SP", "health_beauty", "2018-06"

    memory.write_row(
        state,
        category,
        month,
        geo_predicted_growth=0.21,
        geo_actual_growth=0.13,
        geo_directional_accuracy=1,
        geo_confidence="HIGH",
        geo_confidence_score=0.92,
        geo_reasoning="geo-ok",
    )
    memory.write_row(
        state,
        category,
        month,
        sq_seller_count=12,
        sq_avg_review=4.6,
        sq_avg_delivery_days=8.1,
        sq_churn_risk="LOW",
        sq_top_seller_id="seller-1",
        sq_reasoning="sq-ok",
    )
    memory.write_row(
        state,
        category,
        month,
        cr_avg_spend=187.3,
        cr_order_volume_trend=0.09,
        cr_top_payment_type="credit_card",
        cr_high_value_customer_count=54,
        cr_repeat_rate=0.02,
        cr_reasoning="cr-ok",
    )
    memory.write_row(
        state,
        category,
        month,
        log_avg_delivery_days=7.8,
        log_pct_on_time=0.95,
        log_freight_ratio=0.18,
        log_fastest_seller_state="SP",
        log_delivery_variance=1.2,
        log_reasoning="log-ok",
    )
    memory.write_row(
        state,
        category,
        month,
        conn_decision="recruit_local_sellers",
        conn_confidence="HIGH",
        conn_reasoning="conn-ok",
        conn_actual_outcome="pending",
        conn_most_predictive_agent="geographic",
    )

    row = memory.read_row(state, category, month)
    assert row is not None
    assert row["geo_confidence"] == "HIGH"
    assert row["sq_top_seller_id"] == "seller-1"
    assert row["cr_top_payment_type"] == "credit_card"
    assert row["log_fastest_seller_state"] == "SP"
    assert row["conn_decision"] == "recruit_local_sellers"

    all_rows = memory.read_all()
    assert all_rows.shape == (1, 32)
    assert list(all_rows.columns) == list(SCHEMA_COLUMNS)
    assert all_rows.iloc[0]["state"] == state
    assert all_rows.iloc[0]["category"] == category
    assert all_rows.iloc[0]["month"] == month
