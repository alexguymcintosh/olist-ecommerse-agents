from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agents.connector.connector_agent import ConnectorAgent
from agents.customer_ready.customer_ready_agent import CustomerReadinessAgent
from agents.geographic.geographic_agent import GeographicAgent
from agents.logistics.logistics_agent import LogisticsAgent
from agents.supply_quality.supply_quality_agent import SupplyQualityAgent
from utils.config import (
    FOCUS_CATEGORIES,
    FOCUS_STATES,
    MAX_ITERATIONS,
    MIN_MONTHLY_ORDERS,
    TRAINING_WINDOW_MONTHS,
)
from utils.data_loader import load_all, temporal_window
from utils.memory import Memory


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


def _build_base_df(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_df = (
        data["orders"]
        .merge(data["customers"], on="customer_id", how="inner")
        .merge(data["order_items"], on="order_id", how="inner")
        .merge(data["products"], on="product_id", how="left")
        .merge(data["categories"], on="product_category_name", how="left")
    )
    base_df["order_purchase_timestamp"] = pd.to_datetime(
        base_df["order_purchase_timestamp"], errors="coerce"
    )
    base_df["month"] = base_df["order_purchase_timestamp"].dt.to_period("M")
    base_df["price"] = pd.to_numeric(base_df["price"], errors="coerce").fillna(0.0)
    return base_df


def _sorted_months(base_df: pd.DataFrame) -> list[pd.Period]:
    month_index = pd.PeriodIndex(base_df["month"].astype(str), freq="M")
    return sorted(month_index.unique())


def _compute_actual_growth(
    base_df: pd.DataFrame,
    state: str,
    category: str,
    prediction_month: str,
) -> float | None:
    pred_period = pd.Period(prediction_month, freq="M")
    prev_period = pred_period - 1

    scoped = base_df[
        (base_df["customer_state"] == state)
        & (base_df["product_category_name_english"] == category)
    ]
    if scoped.empty:
        return None

    current = scoped[scoped["month"] == pred_period]
    prev = scoped[scoped["month"] == prev_period]
    current_orders = int(current["order_id"].nunique())
    if current_orders < MIN_MONTHLY_ORDERS:
        return None

    current_revenue = float(current["price"].sum())
    prev_revenue = float(prev["price"].sum())
    if prev_revenue == 0.0:
        return None

    growth = (current_revenue - prev_revenue) / prev_revenue
    if not math.isfinite(growth):
        return None
    return float(growth)


def _decision_expects_growth(decision: str) -> bool:
    normalized = decision.strip().lower()
    non_growth_tokens = ("no_action", "hold", "pause", "reduce", "decrease", "exit")
    return not any(token in normalized for token in non_growth_tokens)


def _validate_prev_iteration(
    base_df: pd.DataFrame,
    memory: Memory,
    prediction_month: str,
    states: list[str],
    categories: list[str],
) -> dict[str, int]:
    validated = 0
    correct = 0
    incorrect = 0
    unknown = 0

    for state in states:
        for category in categories:
            row = memory.read_row(state, category, prediction_month)
            if not row:
                continue

            decision = str(row.get("conn_decision") or "")
            if not decision:
                continue

            actual_growth = _compute_actual_growth(base_df, state, category, prediction_month)
            if actual_growth is None:
                outcome = "unknown"
                unknown += 1
                memory.write_row(
                    state,
                    category,
                    prediction_month,
                    conn_actual_outcome=outcome,
                )
                continue

            expected_positive = _decision_expects_growth(decision)
            actual_positive = actual_growth > 0
            outcome = "correct" if expected_positive == actual_positive else "incorrect"
            if outcome == "correct":
                correct += 1
            else:
                incorrect += 1
            validated += 1

            predicted_growth = row.get("geo_predicted_growth")
            geo_directional_accuracy: int | None = None
            if isinstance(predicted_growth, (int, float)) and math.isfinite(
                float(predicted_growth)
            ):
                geo_directional_accuracy = int(
                    math.copysign(1.0, float(predicted_growth))
                    == math.copysign(1.0, actual_growth)
                )

            memory.write_row(
                state,
                category,
                prediction_month,
                geo_actual_growth=actual_growth,
                geo_directional_accuracy=geo_directional_accuracy,
                conn_actual_outcome=outcome,
            )

    return {
        "validated": validated,
        "correct": correct,
        "incorrect": incorrect,
        "unknown": unknown,
    }


def run_walk_forward_full(
    *,
    db_path: str,
    max_iterations: int,
) -> dict[str, Any]:
    data = load_all()
    memory = Memory(db_path=db_path)
    base_df = _build_base_df(data)
    months = _sorted_months(base_df)

    max_available = max(0, len(months) - TRAINING_WINDOW_MONTHS)
    max_available = min(max_available, MAX_ITERATIONS)
    effective_iterations = min(max_iterations, max_available)

    geographic_agent = GeographicAgent(data=data)
    supply_agent = SupplyQualityAgent(data=data)
    customer_agent = CustomerReadinessAgent(data=data, memory=memory)
    logistics_agent = LogisticsAgent(data=data, memory=memory)
    connector_agent = ConnectorAgent(memory=memory)

    iteration_summaries: list[dict[str, Any]] = []
    prev_prediction_month: str | None = None

    for iteration in range(1, effective_iterations + 1):
        try:
            train_start_period = months[iteration - 1]
            train_end_period = train_start_period + (TRAINING_WINDOW_MONTHS - 1)
            prediction_period = train_end_period + 1

            train_start = str(train_start_period)
            train_end = str(train_end_period)
            prediction_month = str(prediction_period)

            training_df, month_count = temporal_window(base_df, train_start, train_end)
            if month_count == 0:
                raise ValueError(
                    f"Empty training window for iteration {iteration}: {train_start} to {train_end}"
                )

            geo_outputs = geographic_agent.run(
                training_df=training_df,
                iteration=iteration,
                training_window=(train_start, train_end),
                prediction_month=prediction_month,
            )

            prev_sq = {
                (state, category): (
                    memory.read_prev_row(state, category, prediction_month) or {}
                )
                for state in FOCUS_STATES
                for category in FOCUS_CATEGORIES
            }
            sq_outputs = supply_agent.run(
                training_df=training_df,
                states=FOCUS_STATES,
                categories=FOCUS_CATEGORIES,
                month=prediction_month,
                prev_memory=prev_sq,
            )

            prev_cr = {
                (state, category): (
                    memory.read_prev_row(state, category, prediction_month) or {}
                )
                for state in FOCUS_STATES
                for category in FOCUS_CATEGORIES
            }
            cr_outputs = customer_agent.run(
                training_df=training_df,
                states=FOCUS_STATES,
                categories=FOCUS_CATEGORIES,
                month=prediction_month,
                prev_memory=prev_cr,
            )

            prev_log = {
                (state, category): (
                    memory.read_prev_row(state, category, prediction_month) or {}
                )
                for state in FOCUS_STATES
                for category in FOCUS_CATEGORIES
            }
            _ = prev_log  # reserved for future logistics temporal coherence wiring
            log_outputs = logistics_agent.run(month=prediction_month)

            for output in geo_outputs["predictions"]:
                memory.write_row(
                    output["state"],
                    output["category"],
                    prediction_month,
                    geo_predicted_growth=output["predicted_growth_pct"],
                    geo_confidence=output["confidence"],
                    geo_confidence_score=output["confidence_score"],
                    geo_reasoning=output["reasoning"],
                )

            for sq in sq_outputs:
                memory.write_row(
                    sq["state"],
                    sq["category"],
                    prediction_month,
                    sq_seller_count=sq["seller_count"],
                    sq_avg_review=sq["avg_review_score"],
                    sq_avg_delivery_days=sq["avg_delivery_days"],
                    sq_churn_risk=sq["churn_risk"],
                    sq_top_seller_id=sq["top_seller_id"],
                    sq_reasoning=sq["reasoning"],
                )

            for cr in cr_outputs:
                memory.write_row(
                    cr["state"],
                    cr["category"],
                    prediction_month,
                    cr_avg_spend=cr["avg_spend"],
                    cr_order_volume_trend=cr["order_volume_trend"],
                    cr_top_payment_type=cr["top_payment_type"],
                    cr_high_value_customer_count=cr["high_value_customer_count"],
                    cr_repeat_rate=cr["repeat_rate"],
                    cr_reasoning=cr["reasoning"],
                )

            for log in log_outputs:
                memory.write_row(
                    log["state"],
                    log["category"],
                    prediction_month,
                    log_avg_delivery_days=log["avg_delivery_days"],
                    log_pct_on_time=log["pct_on_time"],
                    log_freight_ratio=log["freight_ratio"],
                    log_fastest_seller_state=log["fastest_seller_state"],
                    log_delivery_variance=log["delivery_variance"],
                    log_reasoning=log["reasoning"],
                )

            conn_output = connector_agent.run(
                month=prediction_month,
                geographic_outputs=geo_outputs["predictions"],
                supply_outputs=sq_outputs,
                customer_outputs=cr_outputs,
                logistics_outputs=log_outputs,
                prev_month=prev_prediction_month,
            )
            for decision in conn_output["decisions"]:
                memory.write_row(
                    decision["state"],
                    decision["category"],
                    prediction_month,
                    conn_decision=decision["decision"],
                    conn_confidence=decision["confidence"],
                    conn_reasoning=decision["reasoning"],
                    conn_most_predictive_agent=decision["most_predictive_agent"],
                )

            validation_summary: dict[str, int] | None = None
            if iteration > 1 and prev_prediction_month:
                validation_summary = _validate_prev_iteration(
                    base_df=base_df,
                    memory=memory,
                    prediction_month=prev_prediction_month,
                    states=FOCUS_STATES,
                    categories=FOCUS_CATEGORIES,
                )

            prev_prediction_month = prediction_month
            print(
                f"Iteration {iteration}/{effective_iterations} complete - {prediction_month}"
            )
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "training_window": [train_start, train_end],
                    "prediction_month": prediction_month,
                    "status": "ok",
                    "validation": validation_summary,
                }
            )
        except Exception as exc:
            error_text = str(exc)
            print(f"Iteration {iteration}/{effective_iterations} failed: {error_text}")
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "status": "error",
                    "error": error_text,
                }
            )
            continue

    if prev_prediction_month:
        final_validation = _validate_prev_iteration(
            base_df=base_df,
            memory=memory,
            prediction_month=prev_prediction_month,
            states=FOCUS_STATES,
            categories=FOCUS_CATEGORIES,
        )
    else:
        final_validation = None

    result: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "configured_max_iterations": max_iterations,
        "effective_iterations": effective_iterations,
        "max_available_iterations": max_available,
        "training_window_months": TRAINING_WINDOW_MONTHS,
        "states": FOCUS_STATES,
        "categories": FOCUS_CATEGORIES,
        "iterations": iteration_summaries,
        "final_validation": final_validation,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full 5-agent walk-forward.")
    parser.add_argument(
        "--db",
        type=str,
        default="memory.db",
        help="SQLite memory path (default: memory.db)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Iterations to run (default: 2 for testing, use 13 for full run).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_walk_forward_full(
        db_path=args.db,
        max_iterations=max(0, int(args.max_iterations)),
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
    output_path = output_dir / f"walk_forward_full_{ts}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2)

    print(f"Wrote walk-forward result: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
