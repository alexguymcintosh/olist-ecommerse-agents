from __future__ import annotations

from pathlib import Path
import sys
from typing import get_type_hints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.schema_agents import ConnectorDecision, ConnectorOutput


def _build_connector_decision() -> ConnectorDecision:
    return {
        "state": "SP",
        "category": "health_beauty",
        "month": "2018-08",
        "composite_score": 0.42,
        "decision": "Recruit local sellers",
        "confidence": "MEDIUM",
        "urgency": "LOW",
        "reasoning": "Composite signal is moderate.",
        "challenge": "Local supply might lag.",
        "most_predictive_agent": "geographic",
        "risk_flags": [],
    }


def _build_connector_output() -> ConnectorOutput:
    return {
        "agent": "connector",
        "timestamp": "2026-04-18T00:00:00+00:00",
        "month": "2018-08",
        "decisions": [_build_connector_decision()],
        "briefing": "Top opportunities identified.",
        "follow_up_used": False,
        "follow_up_agent": None,
        "follow_up_question": None,
        "follow_up_response": None,
    }


def test_connector_decision_required_keys() -> None:
    payload = _build_connector_decision()
    assert set(payload.keys()) == {
        "state",
        "category",
        "month",
        "composite_score",
        "decision",
        "confidence",
        "urgency",
        "reasoning",
        "challenge",
        "most_predictive_agent",
        "risk_flags",
    }


def test_connector_output_required_keys() -> None:
    payload = _build_connector_output()
    assert set(payload.keys()) == {
        "agent",
        "timestamp",
        "month",
        "decisions",
        "briefing",
        "follow_up_used",
        "follow_up_agent",
        "follow_up_question",
        "follow_up_response",
    }


def test_connector_output_decisions_typed_as_list_of_connector_decision() -> None:
    hints = get_type_hints(ConnectorOutput, include_extras=True)
    payload = _build_connector_output()
    assert hints["decisions"] == list[ConnectorDecision]
    assert isinstance(payload["decisions"], list)
    assert set(payload["decisions"][0].keys()) == set(_build_connector_decision().keys())


def test_connector_output_follow_up_fields_allow_none() -> None:
    hints = get_type_hints(ConnectorOutput, include_extras=True)
    payload = _build_connector_output()
    assert payload["follow_up_agent"] is None
    assert payload["follow_up_question"] is None
    assert payload["follow_up_response"] is None
    assert hints["follow_up_agent"] == str | None
    assert hints["follow_up_question"] == str | None
    assert hints["follow_up_response"] == str | None
