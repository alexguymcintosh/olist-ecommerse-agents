from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.geographic.geographic_agent import GeographicAgent
from utils.data_loader import load_all
from utils.schema import ConnectorOutput
from walk_forward import run_walk_forward


def _build_real_data_for_integration() -> dict[str, pd.DataFrame]:
    """
    Load real CSV data and minimally normalize for integration constraints:
    - keep real dataset joins
    - ensure category null ratio stays below GeographicAgent gate
    - ensure 25 calendar months are present so full-range == 13 iterations
    """
    data = load_all()
    products = data["products"].copy()
    orders = data["orders"].copy()

    # Keep real data but avoid category-null gate failures from missing product categories.
    products["product_category_name"] = products["product_category_name"].fillna(
        "beleza_saude"
    )
    # Collapse to one category/state to reduce LLM call volume while keeping real-data wiring.
    products["product_category_name"] = "beleza_saude"
    customers = data["customers"].copy()
    customers["customer_state"] = "RJ"
    sellers = data["sellers"].copy()
    sellers["seller_state"] = "RJ"

    # Add a single real-like row to introduce month 2018-10 into joined order-item history.
    seed_row = orders.iloc[[0]].copy()
    seed_row["order_id"] = "wf_integration_synth_order_2018_10"
    seed_row["order_purchase_timestamp"] = "2018-10-15 00:00:00"
    seed_row["customer_id"] = orders.iloc[0]["customer_id"]
    orders = pd.concat([orders, seed_row], ignore_index=True)

    oi = data["order_items"].copy()
    oi_seed = oi.iloc[[0]].copy()
    oi_seed["order_id"] = "wf_integration_synth_order_2018_10"
    oi = pd.concat([oi, oi_seed], ignore_index=True)
    data["order_items"] = oi

    data["products"] = products
    data["orders"] = orders
    data["customers"] = customers
    data["sellers"] = sellers
    return data


def test_walk_forward_full_13_iterations_end_to_end(
    monkeypatch, tmp_path: Path
) -> None:
    data = _build_real_data_for_integration()
    monkeypatch.setattr(
        "agents.geographic.geographic_agent.query_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline test")),
    )

    result = run_walk_forward(
        start=1,
        end=13,
        validate=True,
        data=data,
        output_dir=tmp_path,
    )

    assert result["completed_iterations"] == 13
    assert len(result["iterations"]) == 13

    output_files = list(tmp_path.glob("walk_forward_*.json"))
    assert len(output_files) == 1
    loaded = json.loads(output_files[0].read_text(encoding="utf-8"))
    assert loaded["completed_iterations"] == 13
    assert len(loaded["iterations"]) == 13


def test_walk_forward_iteration_payload_shapes(monkeypatch, tmp_path: Path) -> None:
    data = _build_real_data_for_integration()
    monkeypatch.setattr(
        "agents.geographic.geographic_agent.query_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline test")),
    )

    result = run_walk_forward(
        start=1,
        end=3,
        validate=True,
        data=data,
        output_dir=tmp_path,
    )
    required = {
        "training_window",
        "prediction_month",
        "predictions",
        "supply_gaps",
        "ranked_opportunities",
        "previous_prediction",
    }
    for iteration in result["iterations"]:
        assert required.issubset(iteration.keys())
        tw = iteration["training_window"]
        assert isinstance(tw, (tuple, list))
        assert len(tw) == 2
        assert all(isinstance(x, str) for x in tw)
        assert isinstance(iteration["predictions"], list)


def test_v1_path_unchanged_after_v2_integration() -> None:
    run_all_mod = importlib.import_module("run_all")
    assert run_all_mod is not None

    # v1 ConnectorOutput remains unchanged by walk_forward integration path.
    assert "geographic_output" not in ConnectorOutput.__annotations__
