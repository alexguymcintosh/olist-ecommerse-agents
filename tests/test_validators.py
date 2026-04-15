from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.validators import validate_output


def _valid_customer_output() -> dict:
    return {
        "agent": "customer",
        "timestamp": "2026-04-15T00:00:00+00:00",
        "insights": ["Repeat rate is zero."],
        "metrics": {
            "repeat_rate": 0.0,
            "avg_review_score": 4.1,
            "avg_delivery_days": 12.5,
            "pct_late_delivery": 0.18,
            "top_payment_type": "credit_card",
        },
        "top_opportunity": "Improve delivery performance.",
        "risk_flags": ["late_delivery_high"],
    }


def test_valid_output_passes() -> None:
    validate_output(_valid_customer_output(), "CustomerAgent")


def test_missing_top_level_key_fails() -> None:
    output = _valid_customer_output()
    del output["insights"]
    with pytest.raises(ValueError, match=r"missing required keys"):
        validate_output(output, "CustomerAgent")


def test_missing_metric_key_fails() -> None:
    output = _valid_customer_output()
    del output["metrics"]["repeat_rate"]
    with pytest.raises(ValueError, match=r"metrics missing keys"):
        validate_output(output, "CustomerAgent")


def test_unknown_agent_type_passes_without_metric_check() -> None:
    output = {
        "agent": "experimental",
        "timestamp": "2026-04-15T00:00:00+00:00",
        "insights": ["Smoke test."],
        "metrics": {},
        "top_opportunity": "n/a",
        "risk_flags": [],
    }
    validate_output(output, "ExperimentalAgent")
