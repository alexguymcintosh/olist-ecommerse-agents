from __future__ import annotations

from pathlib import Path
import sys
from typing import get_type_hints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.schema_agents import CustomerReadinessOutput


def _build_customer_readiness_output() -> CustomerReadinessOutput:
    return {
        "agent": "customer_readiness",
        "timestamp": "2026-04-18T00:00:00+00:00",
        "state": "RJ",
        "category": "health_beauty",
        "month": "2018-08",
        "avg_spend": 117.2,
        "order_volume_trend": 0.12,
        "top_payment_type": "credit_card",
        "high_value_customer_count": 17,
        "repeat_rate": 0.0,
        "installment_pct": 0.31,
        "readiness": "MEDIUM",
        "reasoning": "Spend is stable and installment use is healthy.",
        "risk_flags": [],
    }


def test_customer_readiness_output_required_keys() -> None:
    payload = _build_customer_readiness_output()
    assert set(payload.keys()) == {
        "agent",
        "timestamp",
        "state",
        "category",
        "month",
        "avg_spend",
        "order_volume_trend",
        "top_payment_type",
        "high_value_customer_count",
        "repeat_rate",
        "installment_pct",
        "readiness",
        "reasoning",
        "risk_flags",
    }


def test_customer_readiness_output_readiness_enum() -> None:
    payload = _build_customer_readiness_output()
    assert payload["readiness"] in {"HIGH", "MEDIUM", "LOW"}


def test_customer_readiness_output_numeric_fields_are_declared() -> None:
    hints = get_type_hints(CustomerReadinessOutput, include_extras=True)
    assert hints["avg_spend"] is float
    assert hints["order_volume_trend"] is float
    assert hints["high_value_customer_count"] is int
    assert hints["repeat_rate"] is float
    assert hints["installment_pct"] is float
