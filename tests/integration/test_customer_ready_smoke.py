from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.customer_ready.customer_ready_agent import CustomerReadinessAgent
from utils.data_loader import load_all


def test_customer_ready_smoke_real_data_mocked_llm_25_outputs_no_nan() -> None:
    data = load_all()

    def mocked_llm(*_args, **_kwargs) -> str:
        return '{"readiness":"MEDIUM","reasoning":"stable demand","risk_flags":[]}'

    agent = CustomerReadinessAgent(data=data, llm_client=mocked_llm)
    outputs = agent.run()
    assert len(outputs) == 25

    numeric_fields = [
        "avg_spend",
        "order_volume_trend",
        "high_value_customer_count",
        "repeat_rate",
        "installment_pct",
    ]
    for output in outputs:
        for field in numeric_fields:
            assert pd.notna(output[field])
        assert output["state"] != ""
        assert output["category"] != ""
        assert output["month"] != ""
        assert output["readiness"] in {"HIGH", "MEDIUM", "LOW"}

    second_outputs = agent.run()
    stable_first = [
        (
            o["state"],
            o["category"],
            o["month"],
            o["avg_spend"],
            o["order_volume_trend"],
            o["top_payment_type"],
            o["high_value_customer_count"],
            o["repeat_rate"],
            o["installment_pct"],
            o["readiness"],
            o["reasoning"],
            tuple(o["risk_flags"]),
        )
        for o in outputs
    ]
    stable_second = [
        (
            o["state"],
            o["category"],
            o["month"],
            o["avg_spend"],
            o["order_volume_trend"],
            o["top_payment_type"],
            o["high_value_customer_count"],
            o["repeat_rate"],
            o["installment_pct"],
            o["readiness"],
            o["reasoning"],
            tuple(o["risk_flags"]),
        )
        for o in second_outputs
    ]
    assert stable_first == stable_second
