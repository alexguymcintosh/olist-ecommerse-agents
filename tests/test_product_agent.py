import json
from unittest.mock import patch

import pytest

from agents.product.product_agent import ProductAgent


@pytest.fixture
def mock_query_llm():
    """Patch ``query_llm`` where ``BaseAgent`` resolves it (import binding)."""
    payload = json.dumps(
        {
            "insights": [
                "Health & beauty leads category revenue.",
                "Category mix spans dozens of segments.",
            ],
            "top_opportunity": "Double down on high-revenue category partnerships.",
            "risk_flags": ["category_concentration"],
        }
    )
    with patch("utils.base_agent.query_llm", return_value=payload) as mock:
        yield mock


def test_prepare_data_has_expected_join_columns(sample_data):
    agent = ProductAgent(sample_data)
    df = agent._prepare_data()
    for col in (
        "order_id",
        "product_id",
        "price",
        "product_category_name",
        "product_category_name_english",
    ):
        assert col in df.columns


def test_compute_metrics_returns_required_keys_and_types(sample_data):
    agent = ProductAgent(sample_data)
    df = agent._prepare_data()
    metrics = agent._compute_metrics(df)
    assert set(metrics.keys()) == {
        "top_category",
        "top_category_revenue",
        "total_categories",
        "avg_order_value",
    }
    assert isinstance(metrics["top_category"], str)
    assert isinstance(metrics["top_category_revenue"], float)
    assert isinstance(metrics["total_categories"], int)
    assert isinstance(metrics["avg_order_value"], float)
    assert metrics["avg_order_value"] == metrics["avg_order_value"]  # not NaN


def test_build_question_embeds_metrics_and_sample(sample_data):
    agent = ProductAgent(sample_data)
    df = agent._prepare_data()
    metrics = agent._compute_metrics(df)
    sample_str = "order_id,product_id,price\nabc,p1,10.0\n"
    question = agent._build_question(metrics, sample_str)
    assert str(metrics["total_categories"]) in question
    assert sample_str in question
    assert "insights" in question


def test_run_passes_validation_and_calls_llm(sample_data, mock_query_llm):
    agent = ProductAgent(sample_data)
    out = agent.run()
    assert out["agent"] == "product"
    assert out["insights"]
    assert out["top_opportunity"]
    assert "top_category" in out["metrics"]
    mock_query_llm.assert_called_once()
    call_kw = mock_query_llm.call_args
    messages = call_kw[0][0]
    user = messages[1]["content"]
    assert "Pre-computed metrics" in user
    assert "insights" in user
