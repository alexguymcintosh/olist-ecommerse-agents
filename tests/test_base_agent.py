from pathlib import Path
import json
import sys
from unittest.mock import patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.base_agent import BaseAgent
from utils.validators import validate_output

_FULL_CUSTOMER_METRICS = {
    "repeat_rate": 0.0,
    "avg_review_score": 4.0,
    "avg_delivery_days": 10.0,
    "pct_late_delivery": 0.1,
    "top_payment_type": "credit_card",
}


class _StubAgent(BaseAgent):
    AGENT_NAME = "customer"

    def _prepare_data(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1]})

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        return dict(_FULL_CUSTOMER_METRICS)

    def _build_question(self, metrics: dict, sample_str: str) -> str:
        return "q"


@pytest.fixture
def parse_agent() -> _StubAgent:
    return _StubAgent({}, teach=False)


def test_parse_llm_response_clean_json(parse_agent: _StubAgent) -> None:
    raw = '{"insights": ["x"], "top_opportunity": "o", "risk_flags": []}'
    out = parse_agent._parse_llm_response(raw)
    assert out["insights"] == ["x"]
    assert out["top_opportunity"] == "o"
    assert out["risk_flags"] == []


def test_parse_llm_response_prose_prefix(parse_agent: _StubAgent) -> None:
    raw = 'Here is the JSON you asked for:\n\n{"a": 1, "b": 2}\nThanks!'
    out = parse_agent._parse_llm_response(raw)
    assert out == {"a": 1, "b": 2}


def test_parse_llm_response_code_fence(parse_agent: _StubAgent) -> None:
    raw = '```json\n{"insights": [], "top_opportunity": "t", "risk_flags": ["r"]}\n```'
    out = parse_agent._parse_llm_response(raw)
    assert out["insights"] == []
    assert out["top_opportunity"] == "t"
    assert out["risk_flags"] == ["r"]


def test_parse_llm_response_non_json_raises(parse_agent: _StubAgent) -> None:
    raw = "this is not json at all"
    with pytest.raises(ValueError, match=r"no JSON object"):
        parse_agent._parse_llm_response(raw)


def test_run_integration_passes_validate_output() -> None:
    llm_payload = json.dumps(
        {
            "insights": ["Stub insight."],
            "metrics": _FULL_CUSTOMER_METRICS,
            "top_opportunity": "Test opportunity.",
            "risk_flags": ["stub_flag"],
        }
    )
    with patch("utils.base_agent.query_llm", return_value=llm_payload):
        agent = _StubAgent({})
        out = agent.run()

    validate_output(out, "_StubAgent")
    assert out["agent"] == "customer"
    assert out["insights"] == ["Stub insight."]
    assert out["top_opportunity"] == "Test opportunity."
    assert out["risk_flags"] == ["stub_flag"]
    assert out["metrics"] == _FULL_CUSTOMER_METRICS
