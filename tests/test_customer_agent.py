import json
from unittest.mock import patch

import pytest

from agents.customer.customer_agent import CustomerAgent


@pytest.fixture
def mock_query_llm():
    """Patch ``query_llm`` where ``BaseAgent`` resolves it (import binding)."""
    payload = json.dumps(
        {
            "insights": ["Repeat purchases are rare.", "Delivery timing varies widely."],
            "top_opportunity": "Reduce late deliveries to protect satisfaction.",
            "risk_flags": ["late_delivery_high"],
        }
    )
    with patch("utils.base_agent.query_llm", return_value=payload) as mock:
        yield mock


def test_prepare_data_has_expected_join_columns(sample_data):
    agent = CustomerAgent(sample_data)
    df = agent._prepare_data()
    for col in ("customer_id", "order_id", "review_score", "payment_type"):
        assert col in df.columns


def test_compute_metrics_returns_required_keys_and_floats(sample_data):
    agent = CustomerAgent(sample_data)
    df = agent._prepare_data()
    metrics = agent._compute_metrics(df)
    assert set(metrics.keys()) == {
        "repeat_rate",
        "avg_review_score",
        "avg_delivery_days",
        "pct_late_delivery",
        "top_payment_type",
    }
    assert isinstance(metrics["repeat_rate"], float)
    assert isinstance(metrics["avg_review_score"], float)
    assert isinstance(metrics["avg_delivery_days"], float)
    assert isinstance(metrics["pct_late_delivery"], float)
    assert isinstance(metrics["top_payment_type"], str)
    assert metrics["pct_late_delivery"] == metrics["pct_late_delivery"]  # not NaN


def test_build_question_embeds_metrics_and_sample(sample_data):
    agent = CustomerAgent(sample_data)
    df = agent._prepare_data()
    metrics = agent._compute_metrics(df)
    sample_str = "order_id,customer_id\nabc,def\n"
    question = agent._build_question(metrics, sample_str)
    assert str(metrics["repeat_rate"]) in question
    assert sample_str in question
    assert "insights" in question


def test_run_passes_validation_and_calls_llm(sample_data, mock_query_llm):
    agent = CustomerAgent(sample_data)
    out = agent.run()
    assert out["agent"] == "customer"
    assert out["insights"]
    assert out["top_opportunity"]
    assert "repeat_rate" in out["metrics"]
    mock_query_llm.assert_called_once()
    call_kw = mock_query_llm.call_args
    messages = call_kw[0][0]
    user = messages[1]["content"]
    assert "Pre-computed metrics" in user
    assert "insights" in user
