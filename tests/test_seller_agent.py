import json
from unittest.mock import patch

import pytest

from agents.seller.seller_agent import SellerAgent


@pytest.fixture
def mock_query_llm():
    """Patch ``query_llm`` where ``BaseAgent`` resolves it (import binding)."""
    payload = json.dumps(
        {
            "insights": ["Seller base is geographically concentrated.", "Revenue is concentrated among top sellers."],
            "top_opportunity": "Diversify seller base beyond São Paulo.",
            "risk_flags": ["seller_concentration_high"],
        }
    )
    with patch("utils.base_agent.query_llm", return_value=payload) as mock:
        yield mock


def test_prepare_data_has_expected_join_columns(sample_data):
    agent = SellerAgent(sample_data)
    df = agent._prepare_data()
    for col in ("seller_id", "order_id", "seller_state", "review_score", "price"):
        assert col in df.columns


def test_compute_metrics_returns_required_keys_and_types(sample_data):
    agent = SellerAgent(sample_data)
    df = agent._prepare_data()
    metrics = agent._compute_metrics(df)
    assert set(metrics.keys()) == {
        "total_sellers",
        "pct_sao_paulo",
        "top_seller_revenue_concentration",
    }
    assert isinstance(metrics["total_sellers"], int)
    assert isinstance(metrics["pct_sao_paulo"], float)
    assert isinstance(metrics["top_seller_revenue_concentration"], float)
    assert metrics["pct_sao_paulo"] == metrics["pct_sao_paulo"]  # not NaN
    assert 0.0 <= metrics["top_seller_revenue_concentration"] <= 1.0


def test_build_question_embeds_metrics_sample_and_delivery_scope_note(sample_data):
    agent = SellerAgent(sample_data)
    df = agent._prepare_data()
    metrics = agent._compute_metrics(df)
    sample_str = "order_id,seller_id\nabc,def\n"
    question = agent._build_question(metrics, sample_str)
    assert str(metrics["total_sellers"]) in question
    assert sample_str in question
    assert "insights" in question
    assert "delivery time analysis belongs to CustomerAgent, do not analyse it here" in question


def test_run_passes_validation_and_calls_llm(sample_data, mock_query_llm):
    agent = SellerAgent(sample_data)
    out = agent.run()
    assert out["agent"] == "seller"
    assert out["insights"]
    assert out["top_opportunity"]
    assert "total_sellers" in out["metrics"]
    mock_query_llm.assert_called_once()
    call_kw = mock_query_llm.call_args
    messages = call_kw[0][0]
    user = messages[1]["content"]
    assert "Pre-computed metrics" in user
    assert "insights" in user
    assert "delivery time analysis belongs to CustomerAgent, do not analyse it here" in user
