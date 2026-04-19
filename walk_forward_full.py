from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ALL_AGENTS = [
    "geographic",
    "supply_quality",
    "customer_readiness",
    "logistics",
    "connector",
]

import pandas as pd

from agents.connector.connector_agent import ConnectorAgent
from agents.customer_ready.customer_ready_agent import CustomerReadinessAgent
from agents.geographic.geographic_agent import GeographicAgent
from agents.logistics.logistics_agent import LogisticsAgent
from agents.supply_quality.supply_quality_agent import SupplyQualityAgent
from utils.config import (
    MAX_ITERATIONS,
    MIN_MONTHLY_ORDERS,
    TRAINING_WINDOW_MONTHS,
    get_top_categories,
    get_top_states,
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


def _build_agent_signal_summary(
    geo_outputs: list[dict[str, Any]],
    sq_outputs: list[dict[str, Any]],
    cr_outputs: list[dict[str, Any]],
    log_outputs: list[dict[str, Any]],
    states: list[str],
    categories: list[str],
) -> list[dict[str, Any]]:
    geo_by_key = {
        (str(item["state"]), str(item["category"])): str(item.get("confidence", "UNKNOWN"))
        for item in geo_outputs
    }
    sq_by_key = {
        (str(item["state"]), str(item["category"])): str(
            item.get("supply_confidence", "UNKNOWN")
        )
        for item in sq_outputs
    }
    cr_by_key = {
        (str(item["state"]), str(item["category"])): str(item.get("readiness", "UNKNOWN"))
        for item in cr_outputs
    }
    log_by_key = {
        (str(item["state"]), str(item["category"])): str(item.get("feasibility", "UNKNOWN"))
        for item in log_outputs
    }

    rows: list[dict[str, Any]] = []
    for state in states:
        for category in categories:
            key = (state, category)
            rows.append(
                {
                    "state": state,
                    "category": category,
                    "geo_confidence": geo_by_key.get(key, "UNKNOWN"),
                    "supply_confidence": sq_by_key.get(key, "UNKNOWN"),
                    "customer_confidence": cr_by_key.get(key, "UNKNOWN"),
                    "logistics_confidence": log_by_key.get(key, "UNKNOWN"),
                }
            )
    return rows


def _write_iteration_output(
    *,
    output_dir: Path,
    iteration: int,
    training_window: tuple[str, str] | list[str] | None,
    prediction_month: str | None,
    status: str,
    validation: dict[str, int] | None,
    top_connector_decisions: list[dict[str, Any]] | None,
    agent_signal_summary: list[dict[str, Any]] | None,
    agent_timings: dict[str, dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    if not prediction_month:
        return

    payload: dict[str, Any] = {
        "iteration": iteration,
        "training_window": list(training_window) if training_window else None,
        "prediction_month": prediction_month,
        "status": status,
        "validation": validation,
        "top_connector_decisions": top_connector_decisions or [],
        "agent_signal_summary": agent_signal_summary or [],
        "agent_timings": agent_timings or {},
    }
    if error:
        payload["error"] = error

    output_path = output_dir / f"{prediction_month}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


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
    n_states: int,
    n_categories: int,
) -> dict[str, Any]:
    data = load_all()
    memory = Memory(db_path=db_path)
    states = get_top_states(min(max(1, n_states), 27), data)
    categories = get_top_categories(min(max(1, n_categories), 73), data)
    if not states:
        raise ValueError("Could not derive top states from loaded data.")
    if not categories:
        raise ValueError("Could not derive top categories from loaded data.")
    print(f"Selected states ({len(states)}): {states}")
    print(f"Selected categories ({len(categories)}): {categories}")

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
    iteration_output_dir = Path("outputs") / "iterations"
    iteration_output_dir.mkdir(parents=True, exist_ok=True)

    iteration_summaries: list[dict[str, Any]] = []
    prev_prediction_month: str | None = None

    for iteration in range(1, effective_iterations + 1):
        train_start: str | None = None
        train_end: str | None = None
        prediction_month: str | None = None
        agent_timings: dict[str, dict[str, Any]] = {}
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

            _t0 = time.perf_counter()
            geo_outputs = geographic_agent.run(
                training_df=training_df,
                iteration=iteration,
                training_window=(train_start, train_end),
                prediction_month=prediction_month,
            )
            agent_timings["geographic"] = {
                "wall_seconds": round(time.perf_counter() - _t0, 2),
                "llm_calls": 1,
            }

            prev_sq = {
                (state, category): (
                    memory.read_prev_row(state, category, prediction_month) or {}
                )
                for state in states
                for category in categories
            }
            _t0 = time.perf_counter()
            sq_outputs = supply_agent.run(
                training_df=training_df,
                states=states,
                categories=categories,
                month=prediction_month,
                prev_memory=prev_sq,
            )
            agent_timings["supply_quality"] = {
                "wall_seconds": round(time.perf_counter() - _t0, 2),
                "llm_calls": 1,
            }

            prev_cr = {
                (state, category): (
                    memory.read_prev_row(state, category, prediction_month) or {}
                )
                for state in states
                for category in categories
            }
            _t0 = time.perf_counter()
            cr_outputs = customer_agent.run(
                training_df=training_df,
                states=states,
                categories=categories,
                month=prediction_month,
                prev_memory=prev_cr,
            )
            agent_timings["customer_readiness"] = {
                "wall_seconds": round(time.perf_counter() - _t0, 2),
                "llm_calls": 1,
            }

            prev_log = {
                (state, category): (
                    memory.read_prev_row(state, category, prediction_month) or {}
                )
                for state in states
                for category in categories
            }
            _t0 = time.perf_counter()
            log_outputs = logistics_agent.run(
                month=prediction_month,
                training_df=training_df,
                states=states,
                categories=categories,
                prev_memory=prev_log,
            )
            agent_timings["logistics"] = {
                "wall_seconds": round(time.perf_counter() - _t0, 2),
                "llm_calls": 1,
            }

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

            _t0 = time.perf_counter()
            conn_output = connector_agent.run(
                month=prediction_month,
                geographic_outputs=geo_outputs["predictions"],
                supply_outputs=sq_outputs,
                customer_outputs=cr_outputs,
                logistics_outputs=log_outputs,
                prev_month=prev_prediction_month,
                states=states,
                categories=categories,
            )
            agent_timings["connector"] = {
                "wall_seconds": round(time.perf_counter() - _t0, 2),
                "llm_calls": 1,
            }
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
                    states=states,
                    categories=categories,
                )

            prev_prediction_month = prediction_month
            print(
                f"Iteration {iteration}/{effective_iterations} complete - {prediction_month}"
            )

            top_connector_decisions = sorted(
                conn_output["decisions"],
                key=lambda item: float(item.get("composite_score", 0.0)),
                reverse=True,
            )[:5]
            agent_signal_summary = _build_agent_signal_summary(
                geo_outputs=geo_outputs["predictions"],
                sq_outputs=sq_outputs,
                cr_outputs=cr_outputs,
                log_outputs=log_outputs,
                states=states,
                categories=categories,
            )
            _write_iteration_output(
                output_dir=iteration_output_dir,
                iteration=iteration,
                training_window=(train_start, train_end),
                prediction_month=prediction_month,
                status="ok",
                validation=validation_summary,
                top_connector_decisions=top_connector_decisions,
                agent_signal_summary=agent_signal_summary,
                agent_timings=agent_timings,
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
            for _agent_name in _ALL_AGENTS:
                if _agent_name not in agent_timings:
                    agent_timings[_agent_name] = {"wall_seconds": None, "llm_calls": 0}
            _write_iteration_output(
                output_dir=iteration_output_dir,
                iteration=iteration,
                training_window=None,
                prediction_month=prediction_month,
                status="error",
                validation=None,
                top_connector_decisions=[],
                agent_signal_summary=[],
                agent_timings=agent_timings,
                error=error_text,
            )
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
            states=states,
            categories=categories,
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
        "states": states,
        "categories": categories,
        "iterations": iteration_summaries,
        "final_validation": final_validation,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full 5-agent walk-forward.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run identifier used for the memory DB filename. "
            "Default: current UTC timestamp YYYY-MM-DD-HH-MM."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Iterations to run (default: 13).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Alias for --max-iterations.",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=5,
        help="Number of top states by order volume (default: 5, max: 27).",
    )
    parser.add_argument(
        "--n-categories",
        type=int,
        default=5,
        help="Number of top categories by revenue (default: 5, max: 73).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id_provided = args.run_id is not None
    run_id = (
        str(args.run_id)
        if run_id_provided
        else datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
    )
    db_path = f"memory_{run_id}.db"

    legacy_memory_path = Path("memory.db")
    if legacy_memory_path.exists() and not run_id_provided:
        print(
            "Warning: found legacy memory.db while --run-id was not provided. "
            f"Using isolated run database: {db_path}. "
            "If you no longer need it, remove it with: rm memory.db"
        )

    effective_iterations = (
        int(args.iterations) if args.iterations is not None else int(args.max_iterations)
    )
    result = run_walk_forward_full(
        db_path=db_path,
        max_iterations=max(0, effective_iterations),
        n_states=min(max(1, int(args.n_states)), 27),
        n_categories=min(max(1, int(args.n_categories)), 73),
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
