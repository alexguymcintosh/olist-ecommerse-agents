from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agents.geographic.geographic_agent import GeographicAgent
from utils.data_loader import temporal_window
from utils.schema_geographic import WalkForwardIteration, WalkForwardResult


def _period_to_str(value: Any) -> str:
    if isinstance(value, pd.Period):
        return str(value)
    return str(pd.Period(str(value), freq="M"))


def _sorted_months(joined_df: pd.DataFrame) -> list[pd.Period]:
    month_idx = pd.PeriodIndex(joined_df["month"].astype(str), freq="M")
    return sorted(month_idx.unique())


def _compute_directional_accuracy(predicted_growth: float, actual_growth: float) -> bool:
    return math.copysign(1.0, predicted_growth) == math.copysign(1.0, actual_growth)


def _compute_pct_error(predicted_growth: float, actual_growth: float) -> float:
    return abs(predicted_growth - actual_growth) / max(abs(actual_growth), 0.01)


def _compute_actual_growth(
    joined_df: pd.DataFrame,
    state: str,
    category: str,
    prediction_month: str,
) -> float | None:
    pred_period = pd.Period(prediction_month, freq="M")
    prev_period = pred_period - 1

    scoped = joined_df[
        (joined_df["customer_state"] == state)
        & (joined_df["product_category_name_english"] == category)
    ]
    if scoped.empty:
        return None

    current = scoped[scoped["month"] == pred_period]
    current_orders = int(current["order_id"].nunique())
    if current_orders < 10:
        return None

    prev = scoped[scoped["month"] == prev_period]
    current_revenue = float(pd.to_numeric(current["price"], errors="coerce").fillna(0.0).sum())
    prev_revenue = float(pd.to_numeric(prev["price"], errors="coerce").fillna(0.0).sum())
    if prev_revenue == 0.0:
        return None

    growth = (current_revenue - prev_revenue) / prev_revenue
    if not math.isfinite(growth):
        return None
    return float(growth)


def _score_iteration_predictions(
    joined_df: pd.DataFrame,
    prediction_month: str,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for pred in predictions:
        predicted_growth = float(pred.get("predicted_growth_pct", 0.0))
        actual_growth = _compute_actual_growth(
            joined_df,
            state=str(pred.get("state", "")),
            category=str(pred.get("category", "")),
            prediction_month=prediction_month,
        )

        if actual_growth is None:
            directional_accuracy: bool | None = None
            pct_error: float | None = None
        else:
            directional_accuracy = _compute_directional_accuracy(
                predicted_growth, actual_growth
            )
            pct_error = _compute_pct_error(predicted_growth, actual_growth)

        items.append(
            {
                "state": str(pred.get("state", "")),
                "category": str(pred.get("category", "")),
                "predicted_growth": predicted_growth,
                "actual_growth": actual_growth,
                "directional_accuracy": directional_accuracy,
                "pct_error": pct_error,
            }
        )
    return {"prediction_month": prediction_month, "items": items}


def _compute_aggregate_accuracy(iterations: list[WalkForwardIteration]) -> dict[str, Any]:
    directional_values: list[float] = []
    pct_errors: list[float] = []

    for iteration in iterations:
        validation = iteration.get("validation", {})
        if not isinstance(validation, dict):
            continue
        items = validation.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            directional = item.get("directional_accuracy")
            pct_error = item.get("pct_error")
            if isinstance(directional, bool) and pct_error is not None:
                directional_values.append(1.0 if directional else 0.0)
            if isinstance(pct_error, (int, float)) and math.isfinite(float(pct_error)):
                pct_errors.append(float(pct_error))

    scored_predictions = len(directional_values)
    if scored_predictions == 0:
        return {
            "avg_directional_accuracy": 0.0,
            "avg_pct_error": 0.0,
            "scored_predictions": 0,
        }

    return {
        "avg_directional_accuracy": float(sum(directional_values) / scored_predictions),
        "avg_pct_error": float(sum(pct_errors) / max(len(pct_errors), 1)),
        "scored_predictions": scored_predictions,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    return value


def run_walk_forward(
    start: int = 1,
    end: int | None = None,
    validate: bool = True,
    ranked: bool = False,
    *,
    data: dict[str, pd.DataFrame] | None = None,
    agent_class: type[GeographicAgent] = GeographicAgent,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> WalkForwardResult:
    """
    Execute walk-forward orchestration with a maximum of 13 iterations.

    The maximum is derived from 25 months of data and a 12-month training window.
    """
    agent = agent_class(data=data)
    joined = agent._load_geographic_data()
    months = _sorted_months(joined)
    max_iterations = max(0, len(months) - 12)
    max_iterations = min(max_iterations, 13)

    if max_iterations <= 0:
        result: WalkForwardResult = {
            "completed_iterations": 0,
            "total_iterations": 0,
            "iterations": [],
            "aggregate_accuracy": {},
        }
        return result

    effective_end = max_iterations if end is None else min(end, max_iterations)
    effective_start = max(1, start)
    if effective_start > effective_end:
        result = {
            "completed_iterations": 0,
            "total_iterations": max_iterations,
            "iterations": [],
            "aggregate_accuracy": {},
        }
        return result

    iterations: list[WalkForwardIteration] = []
    previous_prediction: list[dict[str, Any]] | None = None

    for iteration in range(effective_start, effective_end + 1):
        train_start_period = months[iteration - 1]
        train_end_period = train_start_period + 11  # 12 months inclusive
        prediction_period = train_end_period + 1

        train_start = _period_to_str(train_start_period)
        train_end = _period_to_str(train_end_period)
        prediction_month = _period_to_str(prediction_period)

        training_df, month_count = temporal_window(joined, train_start, train_end)
        if verbose:
            print(
                f"Iteration {iteration}: training_window=({train_start}, {train_end}) "
                f"prediction_month={prediction_month} month_count={month_count}"
            )

        output = agent.run(
            training_df=training_df,
            iteration=iteration,
            training_window=(train_start, train_end),
            prediction_month=prediction_month,
        )

        iteration_payload: WalkForwardIteration = {
            "iteration": iteration,
            "training_window": (train_start, train_end),
            "prediction_month": prediction_month,
            "previous_prediction": previous_prediction,
            "predictions": output["predictions"],
            "supply_gaps": output["supply_gaps"],
            "ranked_opportunities": output["ranked_opportunities"],
            "validation": {},
        }
        iterations.append(iteration_payload)

        # After iteration N>1, score iteration N-1 predictions against actuals.
        if validate and len(iterations) > 1:
            previous_iteration = iterations[-2]
            previous_iteration["validation"] = _score_iteration_predictions(
                joined_df=joined,
                prediction_month=previous_iteration["prediction_month"],
                predictions=previous_iteration["predictions"],
            )

        previous_prediction = output["predictions"]

    result: WalkForwardResult = {
        "completed_iterations": len(iterations),
        "total_iterations": max_iterations,
        "iterations": iterations,
        "aggregate_accuracy": _compute_aggregate_accuracy(iterations) if validate else {},
    }

    out_base = output_dir if output_dir is not None else Path("outputs")
    out_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
    output_path = out_base / f"walk_forward_{ts}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)

    if ranked and iterations:
        print(json.dumps(iterations[-1]["ranked_opportunities"], indent=2))

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run geographic walk-forward analysis.")
    parser.add_argument("--start", type=int, default=1, help="Start iteration (1-based).")
    parser.add_argument("--end", type=int, default=None, help="End iteration (1-based).")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable validation placeholders (Story 3: validation is {}).",
    )
    parser.add_argument(
        "--ranked",
        action="store_true",
        help="Print only final ranked opportunities list.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set")
        return 1

    parser = build_parser()
    args = parser.parse_args(argv)
    verbose = os.getenv("WALK_FORWARD_VERBOSE") == "1"
    run_walk_forward(
        start=args.start,
        end=args.end,
        validate=args.validate,
        ranked=args.ranked,
        verbose=verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
