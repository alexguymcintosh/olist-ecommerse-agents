from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.connector.connector_agent import ConnectorAgent
from utils.config import FOCUS_CATEGORIES, FOCUS_STATES


class _MemoryStub:
    def __init__(self) -> None:
        self.write_calls: list[dict[str, object]] = []

    def read_row(self, state: str, category: str, month: str) -> None:
        del state, category, month
        return None

    def write_row(self, state: str, category: str, month: str, **kwargs: object) -> None:
        self.write_calls.append(
            {"state": state, "category": category, "month": month, "kwargs": kwargs}
        )


def _build_inputs(*, failed: bool = False) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    geographic_outputs: list[dict] = []
    supply_outputs: list[dict] = []
    customer_outputs: list[dict] = []
    logistics_outputs: list[dict] = []
    for state in FOCUS_STATES:
        for category in FOCUS_CATEGORIES:
            flags = ["agent_failed"] if failed else []
            geographic_outputs.append(
                {
                    "state": state,
                    "category": category,
                    "predicted_growth_pct": 0.2,
                    "confidence": "HIGH",
                    "confidence_score": 1.0,
                    "reasoning": "geo",
                }
            )
            supply_outputs.append(
                {
                    "agent": "supply_quality",
                    "timestamp": "2026-04-18T00:00:00+00:00",
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
                    "supply_confidence": "STRONG",
                    "reasoning": "supply",
                    "risk_flags": list(flags),
                }
            )
            customer_outputs.append(
                {
                    "agent": "customer_readiness",
                    "timestamp": "2026-04-18T00:00:00+00:00",
                    "state": state,
                    "category": category,
                    "month": "2018-08",
                    "avg_spend": 100.0,
                    "order_volume_trend": 0.1,
                    "top_payment_type": "credit_card",
                    "high_value_customer_count": 10,
                    "repeat_rate": 0.0,
                    "installment_pct": 0.2,
                    "readiness": "HIGH",
                    "reasoning": "customer",
                    "risk_flags": list(flags),
                }
            )
            logistics_outputs.append(
                {
                    "agent": "logistics",
                    "timestamp": "2026-04-18T00:00:00+00:00",
                    "state": state,
                    "category": category,
                    "month": "2018-08",
                    "avg_delivery_days": 7.0,
                    "pct_on_time": 0.9,
                    "freight_ratio": 0.1,
                    "fastest_seller_state": "SP",
                    "delivery_variance": 1.0,
                    "cross_state_dependency": 0.5,
                    "feasibility": "STRONG",
                    "reasoning": "logistics",
                    "risk_flags": list(flags),
                }
            )
    return geographic_outputs, supply_outputs, customer_outputs, logistics_outputs


def test_connector_smoke_25_decisions_sorted_with_follow_up_paths(monkeypatch) -> None:
    def _llm_trigger(*_args, **_kwargs) -> str:
        _llm_trigger.calls += 1
        if _llm_trigger.calls <= 25:
            return '{"decision":"act","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}'
        return "follow-up answer"

    _llm_trigger.calls = 0

    memory_trigger = _MemoryStub()
    agent_trigger = ConnectorAgent(memory=memory_trigger)
    geo, sq, cr, log = _build_inputs()
    geo[0]["predicted_growth_pct"] = 1.0
    geo[1]["predicted_growth_pct"] = 0.1
    for item in geo[2:]:
        item["predicted_growth_pct"] = 0.05

    monkeypatch.setattr("agents.connector.connector_agent.query_llm", _llm_trigger)
    out_trigger = agent_trigger.run("2018-08", geo, sq, cr, log, prev_month=None)
    assert len(out_trigger["decisions"]) == 25
    scores_trigger = [x["composite_score"] for x in out_trigger["decisions"]]
    assert scores_trigger == sorted(scores_trigger, reverse=True)
    assert out_trigger["follow_up_used"] is True
    assert out_trigger["follow_up_response"] == "follow-up answer"
    assert len(memory_trigger.write_calls) == 25

    monkeypatch.setattr(
        "agents.connector.connector_agent.query_llm",
        lambda *_a, **_k: '{"decision":"act","confidence":"HIGH","urgency":"HIGH","reasoning":"ok","challenge":"none","most_predictive_agent":"geographic"}',
    )
    memory_close = _MemoryStub()
    agent_close = ConnectorAgent(memory=memory_close)
    close_geo, close_sq, close_cr, close_log = _build_inputs()
    close_geo[0]["predicted_growth_pct"] = 0.25
    close_geo[1]["predicted_growth_pct"] = 0.2
    out_close = agent_close.run("2018-08", close_geo, close_sq, close_cr, close_log)
    assert len(out_close["decisions"]) == 25
    scores_close = [x["composite_score"] for x in out_close["decisions"]]
    assert scores_close == sorted(scores_close, reverse=True)
    assert out_close["follow_up_used"] is False
    assert len(memory_close.write_calls) == 25

    memory_failed = _MemoryStub()
    agent_failed = ConnectorAgent(memory=memory_failed)
    fail_geo, fail_sq, fail_cr, fail_log = _build_inputs(failed=True)
    out_failed = agent_failed.run("2018-08", fail_geo, fail_sq, fail_cr, fail_log)
    assert len(out_failed["decisions"]) == 25
    assert all("agent_failed" in x["risk_flags"] for x in out_failed["decisions"])
    assert len(memory_failed.write_calls) == 25
