"""Tests for ConnectorAgent."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agents.connector.connector_agent import ConnectorAgent
from utils.schema import ConnectorOutput, DomainAgentOutput


def _minimal_domain_outputs() -> list[DomainAgentOutput]:
    return [
        {
            "agent": "customer",
            "timestamp": "2020-01-01T00:00:00+00:00",
            "insights": ["i1"],
            "metrics": {"repeat_rate": 0.0},
            "top_opportunity": "Grow retention",
            "risk_flags": [],
        }
    ]


def _assert_valid_connector_output(out: ConnectorOutput) -> None:
    assert set(out.keys()) == {
        "timestamp",
        "cross_domain_insights",
        "strategic_recommendation",
        "priority_actions",
        "briefing",
    }
    assert isinstance(out["timestamp"], str)
    assert isinstance(out["cross_domain_insights"], list)
    assert isinstance(out["strategic_recommendation"], str)
    assert isinstance(out["priority_actions"], list)
    for pa in out["priority_actions"]:
        assert set(pa.keys()) == {"action", "agent", "urgency"}
        assert pa["urgency"] in ("HIGH", "MEDIUM", "LOW")
    assert isinstance(out["briefing"], str)


@pytest.fixture
def mock_connector_llm():
    payload = json.dumps(
        {
            "cross_domain_insights": [
                "Delivery ties reviews to repeat visits.",
                "Top category revenue is concentrated.",
            ],
            "strategic_recommendation": "Prioritise logistics to lift satisfaction and repeat rate.",
            "priority_actions": [
                {
                    "action": "Reduce delivery SLA to under 7 days",
                    "agent": "cross-domain",
                    "urgency": "HIGH",
                }
            ],
            "briefing": (
                "Customer metrics show repeat risk; product mix rewards focus; "
                "seller concentration suggests leverage points."
            ),
        }
    )
    with patch("agents.connector.connector_agent.query_llm", return_value=payload) as mock:
        yield mock


def test_run_with_mocked_query_llm_returns_valid_connector_output(
    mock_connector_llm,
) -> None:
    agent = ConnectorAgent(_minimal_domain_outputs(), teach=False)
    out = agent.run()

    _assert_valid_connector_output(out)
    assert len(out["cross_domain_insights"]) == 2
    assert "Prioritise logistics" in out["strategic_recommendation"]
    assert len(out["priority_actions"]) == 1
    assert out["priority_actions"][0]["urgency"] == "HIGH"
    assert "Customer metrics" in out["briefing"]
    mock_connector_llm.assert_called_once()
    assert mock_connector_llm.call_args.kwargs["max_tokens"] == 2000


def test_init_empty_outputs_raises_value_error() -> None:
    with pytest.raises(ValueError, match="No domain agent outputs to synthesise"):
        ConnectorAgent([])


def test_fallback_output_all_agents_failed_returns_valid_connector_output() -> None:
    failed: list[DomainAgentOutput] = [
        {
            "agent": "customer",
            "timestamp": "t1",
            "insights": [],
            "metrics": {},
            "top_opportunity": "",
            "risk_flags": ["agent_failed"],
        },
        {
            "agent": "product",
            "timestamp": "t2",
            "insights": [],
            "metrics": {},
            "top_opportunity": "unavailable",
            "risk_flags": ["agent_failed"],
        },
        {
            "agent": "seller",
            "timestamp": "t3",
            "insights": [],
            "metrics": {},
            "top_opportunity": "",
            "risk_flags": ["agent_failed"],
        },
    ]
    agent = ConnectorAgent(failed)
    out = agent._fallback_output()

    _assert_valid_connector_output(out)
    assert len(out["cross_domain_insights"]) >= 1
    assert all("unavailable" in s or "skipped" in s for s in out["cross_domain_insights"])
    assert out["priority_actions"] == []


def test_parse_llm_response_prose_prefix() -> None:
    agent = ConnectorAgent(_minimal_domain_outputs())
    raw = (
        "Sure — here is the analysis.\n\n"
        '{"cross_domain_insights": ["x"], "strategic_recommendation": "go", '
        '"priority_actions": [], "briefing": "b"}\n'
        "Hope this helps."
    )
    parsed = agent._parse_llm_response(raw)
    assert parsed["cross_domain_insights"] == ["x"]
    assert parsed["strategic_recommendation"] == "go"
    assert parsed["briefing"] == "b"


def test_parse_llm_response_code_fences() -> None:
    agent = ConnectorAgent(_minimal_domain_outputs())
    inner = {
        "cross_domain_insights": ["a"],
        "strategic_recommendation": "s",
        "priority_actions": [],
        "briefing": "full briefing text",
    }
    raw = "```json\n" + json.dumps(inner) + "\n```"
    parsed = agent._parse_llm_response(raw)
    assert parsed == inner
