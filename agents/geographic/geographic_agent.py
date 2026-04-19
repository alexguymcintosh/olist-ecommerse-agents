"""Geographic demand agent: state/category growth, supply gaps, opportunity ranking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.data_loader import load_all
from utils.openrouter_client import RND_MODEL, build_analyst_prompt, parse_batch_llm_response, query_llm
from utils.config import FOCUS_CATEGORIES, FOCUS_STATES
from utils.schema_geographic import (
    GeographicMetrics,
    GeographicOutput,
    Prediction,
    RankedOpportunity,
    SupplyGap,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class GeographicAgent:
    """Does not subclass BaseAgent; computes geographic momentum and supply gaps."""

    AGENT_NAME = "geographic"

    def __init__(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        model: str = RND_MODEL,
    ) -> None:
        self.data = data if data is not None else load_all()
        self.model = model

    def _load_geographic_data(self) -> pd.DataFrame:
        """Join required tables and enforce category translation quality gate."""
        orders = self.data["orders"].copy()
        customers = self.data["customers"].copy()
        order_items = self.data["order_items"].copy()
        products = self.data["products"].copy()
        sellers = self.data["sellers"].copy()

        categories = pd.read_csv(
            DATA_DIR / "product_category_name_translation.csv",
            encoding="utf-8-sig",
        )

        merged = (
            orders.merge(customers, on="customer_id", how="inner")
            .merge(order_items, on="order_id", how="inner")
            .merge(products, on="product_id", how="left")
            .merge(categories, on="product_category_name", how="left")
            .merge(sellers, on="seller_id", how="left")
        )
        merged["order_purchase_timestamp"] = pd.to_datetime(
            merged["order_purchase_timestamp"], errors="coerce"
        )
        merged["month"] = merged["order_purchase_timestamp"].dt.to_period("M")
        merged["price"] = pd.to_numeric(merged["price"], errors="coerce").fillna(0.0)
        merged["product_category_name_english"] = merged[
            "product_category_name_english"
        ].astype("string")

        null_ratio = float(merged["product_category_name_english"].isna().mean())
        if null_ratio >= 0.03:
            raise ValueError(
                "Null ratio for product_category_name_english is >= 1% after join."
            )
        return merged

    def _identify_top5_states(self, df: pd.DataFrame) -> list[str]:
        rev = (
            df.groupby("customer_state", dropna=True)["price"]
            .sum()
            .sort_values(ascending=False)
        )
        return [str(x) for x in rev.head(5).index.tolist()]

    def _identify_top5_categories(self, df: pd.DataFrame) -> list[str]:
        rev = (
            df.groupby("product_category_name_english", dropna=True)["price"]
            .sum()
            .sort_values(ascending=False)
        )
        return [str(x) for x in rev.head(5).index.tolist()]

    def _score_confidence(
        self, latest_growth: float | None, latest_order_count: int
    ) -> tuple[str, float]:
        if latest_order_count < 10:
            return "LOW", 0.0
        if latest_growth is None or pd.isna(latest_growth):
            return "LOW", 0.0
        if abs(float(latest_growth)) >= 0.10:
            return "HIGH", 1.0
        if abs(float(latest_growth)) >= 0.03:
            return "MEDIUM", 0.6
        return "LOW", 0.3

    def _compute_growth_matrix(
        self, training_df: pd.DataFrame, states: list[str], categories: list[str]
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, dict[str, bool]]]:
        """
        Compute growth and momentum matrices per state/category.

        Returns:
            growth_matrix, momentum_scores, order_counts, sparse_flags
        """
        growth_matrix: dict[str, dict[str, float]] = {}
        momentum_scores: dict[str, dict[str, float]] = {}
        order_counts: dict[str, dict[str, int]] = {}
        sparse_flags: dict[str, dict[str, bool]] = {}

        for state in states:
            growth_matrix[state] = {}
            momentum_scores[state] = {}
            order_counts[state] = {}
            sparse_flags[state] = {}
            for category in categories:
                scoped = training_df[
                    (training_df["customer_state"] == state)
                    & (training_df["product_category_name_english"] == category)
                ].copy()
                if scoped.empty:
                    growth_matrix[state][category] = np.nan
                    momentum_scores[state][category] = np.nan
                    order_counts[state][category] = 0
                    sparse_flags[state][category] = True
                    continue

                monthly = (
                    scoped.groupby("month")
                    .agg(
                        revenue=("price", "sum"),
                        order_count=("order_id", "nunique"),
                    )
                    .sort_index()
                )

                full_range = pd.period_range(
                    monthly.index.min(), monthly.index.max(), freq="M"
                )
                missing = [p for p in full_range if p not in monthly.index]
                monthly = monthly.reindex(full_range, fill_value=0)

                growth = monthly["revenue"].pct_change()
                growth = growth.replace([np.inf, -np.inf], np.nan)

                sparse_series = monthly["order_count"] < 10
                growth = growth.mask(sparse_series, np.nan)
                if missing:
                    growth = growth.mask(growth.index.isin(missing), np.nan)

                latest_growth = (
                    float(growth.iloc[-1]) if pd.notna(growth.iloc[-1]) else np.nan
                )
                last_three = growth.tail(3).dropna()
                momentum = float(last_three.mean()) if not last_three.empty else np.nan
                latest_orders = int(monthly["order_count"].iloc[-1])

                growth_matrix[state][category] = latest_growth
                momentum_scores[state][category] = momentum
                order_counts[state][category] = latest_orders
                sparse_flags[state][category] = bool(latest_orders < 10)

        return growth_matrix, momentum_scores, order_counts, sparse_flags

    def _predict_next_month_growth(
        self, state: str, category: str, momentum: float | None
    ) -> tuple[float, str]:
        """LLM-assisted prediction with safe fallback to momentum (single-pair, kept for compat)."""
        base = 0.0 if momentum is None or pd.isna(momentum) else float(momentum)
        payload = json.dumps(
            {"state": state, "category": category, "momentum": base}, indent=2
        )
        question = (
            "Return ONLY JSON: {\"predicted_growth_pct\": <float>, \"reasoning\": <string>}."
        )
        try:
            messages = build_analyst_prompt(payload, question)
            raw = query_llm(messages, model=self.model, max_tokens=200)
            parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            pred = float(parsed.get("predicted_growth_pct", base))
            reasoning = str(parsed.get("reasoning", "LLM-assisted forecast"))
            return pred, reasoning
        except Exception:
            return base, "Momentum fallback forecast"

    def _predict_batch_growth(
        self, items: list[dict[str, Any]]
    ) -> dict[tuple[str, str], tuple[float, str]]:
        """Send all state×category momentum items in one LLM call.

        Each item: {state, category, momentum}.
        Returns {(state, category): (predicted_growth_pct, reasoning)}.
        Falls back to momentum for any item not returned by LLM.
        """
        def _fallback(item: dict[str, Any]) -> tuple[float, str]:
            base = float(item["momentum"]) if item["momentum"] is not None and not pd.isna(item["momentum"]) else 0.0
            return base, "Momentum fallback forecast"

        payload = json.dumps(items, indent=2)
        question = (
            "For each item predict next-month revenue growth. "
            "Return ONLY a JSON array where each object has exactly: "
            "{\"state\": <string>, \"category\": <string>, "
            "\"predicted_growth_pct\": <float>, \"reasoning\": <string>}. "
            "Array must have one entry per input item in any order."
        )
        try:
            messages = build_analyst_prompt(payload, question)
            raw = query_llm(messages, model=self.model, max_tokens=3000)
            parsed_items = parse_batch_llm_response(raw, items)
            result: dict[tuple[str, str], tuple[float, str]] = {}
            for item, parsed in zip(items, parsed_items):
                key = (item["state"], item["category"])
                base = float(item["momentum"]) if item["momentum"] is not None and not pd.isna(item["momentum"]) else 0.0
                if parsed is not None and isinstance(parsed, dict):
                    pred = float(parsed.get("predicted_growth_pct", base))
                    reasoning = str(parsed.get("reasoning", "LLM-assisted forecast"))
                else:
                    pred, reasoning = _fallback(item)
                result[key] = (pred, reasoning)
            return result
        except Exception:
            return {(item["state"], item["category"]): _fallback(item) for item in items}

    def _compute_supply_gaps(
        self,
        latest_df: pd.DataFrame,
        predictions: list[Prediction],
        sparse_flags: dict[str, dict[str, bool]],
    ) -> list[SupplyGap]:
        gaps: list[SupplyGap] = []
        for pred in predictions:
            state = pred["state"]
            category = pred["category"]
            scoped = latest_df[
                (latest_df["customer_state"] == state)
                & (latest_df["product_category_name_english"] == category)
            ]
            current_orders = int(scoped["order_id"].nunique())

            sellers_scoped = latest_df[
                (latest_df["seller_state"] == state)
                & (latest_df["product_category_name_english"] == category)
            ]
            current_sellers = int(sellers_scoped["seller_id"].nunique())

            if sparse_flags.get(state, {}).get(category, False):
                predicted_order_volume: float | None = None
                supply_gap_ratio: float | None = None
                severity = 0.0
            else:
                predicted_order_volume = float(
                    current_orders * (1.0 + float(pred["predicted_growth_pct"]))
                )
                supply_gap_ratio = float(
                    predicted_order_volume / max(current_sellers, 1)
                )
                severity = supply_gap_ratio

            gaps.append(
                {
                    "state": state,
                    "category": category,
                    "current_sellers": current_sellers,
                    "current_month_order_count": current_orders,
                    "predicted_order_volume": predicted_order_volume,
                    "supply_gap_ratio": supply_gap_ratio,
                    "supply_gap_severity": severity,
                }
            )
        return gaps

    def _rank_opportunities(
        self, predictions: list[Prediction], supply_gaps: list[SupplyGap]
    ) -> list[RankedOpportunity]:
        gap_lookup = {(g["state"], g["category"]): g for g in supply_gaps}
        ranked: list[RankedOpportunity] = []
        for pred in predictions:
            key = (pred["state"], pred["category"])
            gap = gap_lookup[key]
            composite = float(
                pred["predicted_growth_pct"]
                * pred["confidence_score"]
                * gap["supply_gap_severity"]
            )
            urgency = "HIGH" if composite >= 100 else "MEDIUM" if composite >= 10 else "LOW"
            current_orders = int(gap["current_month_order_count"])
            if gap["predicted_order_volume"] is None:
                predicted_order_volume: float | None = None
            else:
                predicted_order_volume = max(
                    0.0,
                    float(current_orders) * (1.0 + float(pred["predicted_growth_pct"])),
                )
            ranked.append(
                {
                    "rank": 0,
                    "state": pred["state"],
                    "category": pred["category"],
                    "predicted_growth_pct": float(pred["predicted_growth_pct"]),
                    "current_sellers": gap["current_sellers"],
                    "current_month_order_count": gap["current_month_order_count"],
                    "predicted_order_volume": predicted_order_volume,
                    "supply_gap_ratio": gap["supply_gap_ratio"],
                    "supply_gap_severity": gap["supply_gap_severity"],
                    "composite_score": composite,
                    "recommended_action": (
                        f"Increase seller coverage for {pred['category']} in {pred['state']}"
                    ),
                    "urgency": urgency,
                }
            )

        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx
        return ranked

    def run(
        self,
        training_df: pd.DataFrame | None = None,
        iteration: int = 1,
        training_window: tuple[str, str] = ("", ""),
        prediction_month: str = "",
    ) -> GeographicOutput:
        """Compute geographic output from training data."""
        joined = self._load_geographic_data()
        if training_df is None:
            training_df = joined

        states = FOCUS_STATES
        categories = FOCUS_CATEGORIES
        growth_matrix, momentum_scores, order_counts, sparse_flags = self._compute_growth_matrix(
            training_df, states, categories
        )

        predictions: list[Prediction] = []
        batch_items = []
        for state in states:
            for category in categories:
                momentum = momentum_scores.get(state, {}).get(category, np.nan)
                momentum_val: float | None = float(momentum) if pd.notna(momentum) else None
                batch_items.append({"state": state, "category": category, "momentum": momentum_val})

        batch_results = self._predict_batch_growth(batch_items)

        for item in batch_items:
            state = item["state"]
            category = item["category"]
            predicted_growth, reasoning = batch_results[(state, category)]
            latest_orders = order_counts.get(state, {}).get(category, 0)
            confidence, confidence_score = self._score_confidence(
                growth_matrix.get(state, {}).get(category, np.nan),
                latest_orders,
            )
            if sparse_flags.get(state, {}).get(category, False):
                confidence = "LOW"
                confidence_score = 0.0

            predictions.append(
                {
                    "state": state,
                    "category": category,
                    "predicted_growth_pct": float(predicted_growth),
                    "confidence": confidence,
                    "confidence_score": float(confidence_score),
                    "reasoning": reasoning,
                }
            )

        latest_month = training_df["month"].max()
        latest_df = joined[joined["month"] == latest_month].copy()
        supply_gaps = self._compute_supply_gaps(latest_df, predictions, sparse_flags)
        ranked = self._rank_opportunities(predictions, supply_gaps)

        metrics: GeographicMetrics = {
            "top5_states": states,
            "top5_categories": categories,
            "growth_matrix": growth_matrix,
            "momentum_scores": momentum_scores,
            "seller_counts": {
                state: {
                    category: int(
                        latest_df[
                            (latest_df["seller_state"] == state)
                            & (latest_df["product_category_name_english"] == category)
                        ]["seller_id"].nunique()
                    )
                    for category in categories
                }
                for state in states
            },
            "order_counts": order_counts,
            "supply_gap_matrix": {
                g["state"]: {
                    **{
                        k["category"]: k["supply_gap_ratio"]
                        for k in supply_gaps
                        if k["state"] == g["state"]
                    }
                }
                for g in supply_gaps
            },
        }

        return {
            "agent": self.AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "training_window": training_window,
            "prediction_month": prediction_month,
            "predictions": predictions,
            "supply_gaps": supply_gaps,
            "ranked_opportunities": ranked,
            "metrics": metrics,
            "risk_flags": [],
        }
