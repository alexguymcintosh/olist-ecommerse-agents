from __future__ import annotations

from pathlib import Path
import sys
from typing import get_type_hints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.schema_agents import SupplyQualityOutput


def _build_supply_output() -> SupplyQualityOutput:
    return {
        "agent": "supply_quality",
        "timestamp": "2026-04-18T00:00:00+00:00",
        "state": "RJ",
        "category": "health_beauty",
        "month": "2018-08",
        "seller_count": 12,
        "avg_review_score": 4.2,
        "avg_delivery_days": 8.5,
        "churn_risk": "LOW",
        "churn_rate": 0.08,
        "top_seller_id": "seller_123",
        "seller_concentration": 0.27,
        "supply_confidence": "STRONG",
        "reasoning": "Healthy seller base and low churn.",
        "risk_flags": [],
    }


def test_supply_quality_output_required_keys() -> None:
    payload = _build_supply_output()
    assert set(payload.keys()) == {
        "agent",
        "timestamp",
        "state",
        "category",
        "month",
        "seller_count",
        "avg_review_score",
        "avg_delivery_days",
        "churn_risk",
        "churn_rate",
        "top_seller_id",
        "seller_concentration",
        "supply_confidence",
        "reasoning",
        "risk_flags",
    }


def test_supply_quality_output_enum_domains_in_fixture() -> None:
    payload = _build_supply_output()
    assert payload["churn_risk"] in {"HIGH", "MEDIUM", "LOW"}
    assert payload["supply_confidence"] in {"STRONG", "ADEQUATE", "WEAK"}


def test_supply_quality_output_type_hints_are_exposed() -> None:
    hints = get_type_hints(SupplyQualityOutput, include_extras=True)
    assert hints["risk_flags"] == list[str]
    assert hints["seller_count"] is int
    assert hints["avg_review_score"] is float
