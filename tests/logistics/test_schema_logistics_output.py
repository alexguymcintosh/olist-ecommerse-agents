from __future__ import annotations

from pathlib import Path
import sys
from typing import get_type_hints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.schema_agents import LogisticsOutput


def _build_logistics_output() -> LogisticsOutput:
    return {
        "agent": "logistics",
        "timestamp": "2026-04-18T00:00:00+00:00",
        "state": "RJ",
        "category": "health_beauty",
        "month": "2018-08",
        "avg_delivery_days": 7.2,
        "pct_on_time": 0.86,
        "freight_ratio": 0.14,
        "fastest_seller_state": "SP",
        "delivery_variance": 1.8,
        "cross_state_dependency": 0.91,
        "feasibility": "ADEQUATE",
        "reasoning": "On-time performance is stable, but dependency is high.",
        "risk_flags": ["high_cross_state_dependency"],
    }


def test_logistics_output_valid_payload_has_all_required_keys() -> None:
    payload = _build_logistics_output()
    assert set(payload.keys()) == {
        "agent",
        "timestamp",
        "state",
        "category",
        "month",
        "avg_delivery_days",
        "pct_on_time",
        "freight_ratio",
        "fastest_seller_state",
        "delivery_variance",
        "cross_state_dependency",
        "feasibility",
        "reasoning",
        "risk_flags",
    }


def test_logistics_output_missing_required_key_is_detected() -> None:
    payload = _build_logistics_output()
    payload.pop("cross_state_dependency")
    assert "cross_state_dependency" not in payload


def test_logistics_output_feasibility_fixture_values_are_valid() -> None:
    payload = _build_logistics_output()
    assert payload["feasibility"] in {"STRONG", "ADEQUATE", "WEAK"}


def test_logistics_output_risk_flags_is_list() -> None:
    hints = get_type_hints(LogisticsOutput, include_extras=True)
    payload = _build_logistics_output()
    assert hints["risk_flags"] == list[str]
    assert isinstance(payload["risk_flags"], list)
