from __future__ import annotations

from pathlib import Path
import sys
from typing import get_type_hints

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.logistics.logistics_agent import LogisticsAgent
from utils.data_loader import load_all
from utils.schema_agents import LogisticsOutput


def test_logistics_smoke_real_data_mocked_llm_returns_25_outputs_without_nan(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agents.logistics.logistics_agent.query_llm",
        lambda *_args, **_kwargs: '{"feasibility":"ADEQUATE","reasoning":"Mocked logistics assessment","risk_flags":[]}',
    )
    data = load_all()
    agent = LogisticsAgent(data=data)

    outputs = agent.run(month="2018-08")

    assert len(outputs) == 25
    required_keys = set(get_type_hints(LogisticsOutput, include_extras=True).keys())

    for output in outputs:
        assert set(output.keys()) == required_keys
        assert output["feasibility"] in {"STRONG", "ADEQUATE", "WEAK"}
        for key, value in output.items():
            if isinstance(value, float):
                assert not pd.isna(value), f"{key} is NaN"
