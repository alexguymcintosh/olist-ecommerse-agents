from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.connector.connector_agent import ConnectorAgent
from utils.config import FOCUS_CATEGORIES, FOCUS_STATES


class _MemoryStub:
    def __init__(self) -> None:
        self.write_calls: list[dict[str, object]] = []
        self.rows: dict[tuple[str, str, str], dict[str, object]] = {}

    def read_row(self, state: str, category: str, month: str) -> dict[str, object] | None:
        return self.rows.get((state, category, month))

    def write_row(self, state: str, category: str, month: str, **kwargs: object) -> None:
        self.write_calls.append(
            {"state": state, "category": category, "month": month, "kwargs": kwargs}
        )


def _build_inputs(
    *,
    growth: float = 0.2,
    confidence_score: float = 1.0,
    supply_confidence: str = "STRONG",
    readiness: str = "HIGH",
    feasibility: str = "STRONG",
    agent_failed: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    geographic_outputs: list[dict] = []
    supply_outputs: list[dict] = []
    customer_outputs: list[dict] = []
    logistics_outputs: list[dict] = []

    for state in FOCUS_STATES:
        for category in FOCUS_CATEGORIES:
            flags = ["agent_failed"] if agent_failed else []
            geographic_outputs.append(
                {
                    "state": state,
                    "category": category,
                    "predicted_growth_pct": growth,
                    "confidence": "HIGH",
                    "confidence_score": confidence_score,
                    "reasoning": "geo",
                }
            )
            supply_outputs.append(
                {
                    "agent": "supply_quality",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "state": state,
                    "category": category,
                    "month": "2018-08",
                    "seller_count": 5,
                    "avg_review_score": 4.0,
                    "avg_delivery_days": 8.0,
                    "churn_risk": "LOW",
                    "churn_rate": 0.0,
                    "top_seller_id": "s1",
                    "seller_concentration": 0.2,
                    "supply_confidence": supply_confidence,
                    "reasoning": "supply",
                    "risk_flags": list(flags),
                }
            )
            customer_outputs.append(
                {
                    "agent": "customer_readiness",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "state": state,
                    "category": category,
                    "month": "2018-08",
                    "avg_spend": 100.0,
                    "order_volume_trend": 0.1,
                    "top_payment_type": "credit_card",
                    "high_value_customer_count": 10,
                    "repeat_rate": 0.0,
                    "installment_pct": 0.2,
                    "readiness": readiness,
                    "reasoning": "customer",
                    "risk_flags": list(flags),
                }
            )
            logistics_outputs.append(
                {
                    "agent": "logistics",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "state": state,
                    "category": category,
                    "month": "2018-08",
                    "avg_delivery_days": 7.0,
                    "pct_on_time": 0.9,
                    "freight_ratio": 0.1,
                    "fastest_seller_state": "SP",
                    "delivery_variance": 1.0,
                    "cross_state_dependency": 0.5,
                    "feasibility": feasibility,
                    "reasoning": "logistics",
                    "risk_flags": list(flags),
                }
            )
    return geographic_outputs, supply_outputs, customer_outputs, logistics_outputs


def test_run_returns_25_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"act","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs()
    output = agent.run("2018-08", geo, sq, cr, log)
    assert len(output["decisions"]) == 25


def test_decisions_sorted_by_composite_score(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs()
    geo[0]["predicted_growth_pct"] = 0.7
    geo[1]["predicted_growth_pct"] = 0.3
    for item in geo[2:]:
        item["predicted_growth_pct"] = 0.1
    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"act","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    output = agent.run("2018-08", geo, sq, cr, log)
    scores = [x["composite_score"] for x in output["decisions"]]
    assert scores == sorted(scores, reverse=True)


def test_prev_month_none_first_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"act","confidence":"MEDIUM","urgency":"MEDIUM","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs()
    output = agent.run("2018-08", geo, sq, cr, log, prev_month=None)
    assert len(output["decisions"]) == 25


def test_all_agents_failed_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"act","confidence":"LOW","urgency":"LOW","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs(agent_failed=True)
    output = agent.run("2018-08", geo, sq, cr, log)
    assert len(output["decisions"]) == 25
    assert all("agent_failed" in x["risk_flags"] for x in output["decisions"])


def test_composite_score_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"grow","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)

    geo, sq, cr, log = _build_inputs(growth=0.2, confidence_score=0.8)
    output = agent.run("2018-08", geo, sq, cr, log)
    first_score = output["decisions"][0]["composite_score"]
    expected = ((0.8 * 0.35) + (1.0 * 0.25) + (1.0 * 0.20) + (1.0 * 0.20)) * 0.2
    assert first_score == pytest.approx(expected)

    zero_geo, zero_sq, zero_cr, zero_log = _build_inputs(growth=0.0)
    zero_output = agent.run("2018-08", zero_geo, zero_sq, zero_cr, zero_log)
    assert all(x["decision"] == "no_action" for x in zero_output["decisions"])
    assert all(x["urgency"] == "LOW" for x in zero_output["decisions"])

    neg_geo, neg_sq, neg_cr, neg_log = _build_inputs(growth=-0.1)
    neg_output = agent.run("2018-08", neg_geo, neg_sq, neg_cr, neg_log)
    assert all(x["composite_score"] <= 0 for x in neg_output["decisions"])
    assert all(x["decision"] == "no_action" for x in neg_output["decisions"])
    assert all(x["urgency"] == "LOW" for x in neg_output["decisions"])


def test_follow_up_triggered_when_top_dominates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_llm(*_a, **_k) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return '{"decision":"act","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}'
        return "follow-up answer"

    monkeypatch.setattr("agents.connector.connector_agent.query_llm", _fake_llm)
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs()
    geo[0]["predicted_growth_pct"] = 1.0
    geo[1]["predicted_growth_pct"] = 0.1
    for item in geo[2:]:
        item["predicted_growth_pct"] = 0.05
    output = agent.run("2018-08", geo, sq, cr, log)
    assert output["follow_up_used"] is True
    assert output["follow_up_question"] is not None
    assert output["follow_up_response"] == "follow-up answer"


def test_follow_up_not_triggered_when_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"act","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs()
    geo[0]["predicted_growth_pct"] = 0.25
    geo[1]["predicted_growth_pct"] = 0.2
    output = agent.run("2018-08", geo, sq, cr, log)
    assert output["follow_up_used"] is False
    assert output["follow_up_question"] is None


def test_memory_write_called_per_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"act","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs()
    agent.run("2018-08", geo, sq, cr, log)
    assert len(memory.write_calls) == 25
    assert set(memory.write_calls[0]["kwargs"].keys()) == {
        "conn_decision",
        "conn_confidence",
        "conn_reasoning",
        "conn_most_predictive_agent",
    }


def test_llm_parse_failure_preserves_composite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.connector.connector_agent.query_llm", lambda *_a, **_k: "bad-response")
    memory = _MemoryStub()
    agent = ConnectorAgent(memory=memory)
    geo, sq, cr, log = _build_inputs(growth=0.2, confidence_score=0.8)
    output = agent.run("2018-08", geo, sq, cr, log)
    expected = ((0.8 * 0.35) + (1.0 * 0.25) + (1.0 * 0.20) + (1.0 * 0.20)) * 0.2
    assert output["decisions"][0]["composite_score"] == pytest.approx(expected)
    assert "connector_failed" in output["decisions"][0]["risk_flags"]
