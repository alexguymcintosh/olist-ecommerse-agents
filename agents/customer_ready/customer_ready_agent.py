"""Customer readiness agent: buyer signal by state/category/month."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from utils.config import FOCUS_CATEGORIES, FOCUS_STATES
from utils.data_loader import load_all
from utils.openrouter_client import RND_MODEL, build_analyst_prompt, parse_batch_llm_response, query_llm
from utils.schema_agents import CustomerReadinessOutput

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
READINESS_VALUES = {"HIGH", "MEDIUM", "LOW"}


class CustomerReadinessAgent:
    """Assess buyer readiness for each focus state/category pair."""

    AGENT_NAME = "customer_readiness"

    def __init__(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        model: str = RND_MODEL,
        memory: Any | None = None,
        llm_client: Callable[..., str] = query_llm,
    ) -> None:
        self.data = data if data is not None else load_all()
        self.model = model
        self.memory = memory
        self.llm_client = llm_client

    def _load_customer_data(
        self, training_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        payments = self.data["payments"].copy()
        payments["payment_installments"] = pd.to_numeric(
            payments["payment_installments"], errors="coerce"
        ).fillna(0)

        payments_agg = (
            payments.groupby("order_id", dropna=True)["payment_value"]
            .sum()
            .reset_index()
            .rename(columns={"payment_value": "total_payment_value"})
        )
        if training_df is not None:
            main_df = training_df.copy()
            training_order_ids = (
                main_df["order_id"].dropna().astype(str).unique().tolist()
                if "order_id" in main_df.columns
                else []
            )
            if training_order_ids:
                payments_agg = payments_agg[
                    payments_agg["order_id"].astype(str).isin(training_order_ids)
                ].copy()
                payments = payments[
                    payments["order_id"].astype(str).isin(training_order_ids)
                ].copy()
            if "total_payment_value" in main_df.columns:
                main_df = main_df.drop(columns=["total_payment_value"])
            main_df = main_df.merge(payments_agg, on="order_id", how="left")
        else:
            orders = self.data["orders"].copy()
            customers = self.data["customers"].copy()
            order_items = self.data["order_items"].copy()
            products = self.data["products"].copy()
            categories = (
                self.data["categories"].copy()
                if "categories" in self.data
                else pd.read_csv(
                    DATA_DIR / "product_category_name_translation.csv",
                    encoding="utf-8-sig",
                )
            )
            main_df = (
                orders.merge(customers, on="customer_id", how="inner")
                .merge(order_items, on="order_id", how="inner")
                .merge(products, on="product_id", how="left")
                .merge(categories, on="product_category_name", how="left")
                .merge(payments_agg, on="order_id", how="left")
            )
        main_df["order_purchase_timestamp"] = pd.to_datetime(
            main_df["order_purchase_timestamp"], errors="coerce"
        )
        main_df["month"] = main_df["order_purchase_timestamp"].dt.to_period("M")
        main_df["total_payment_value"] = pd.to_numeric(
            main_df["total_payment_value"], errors="coerce"
        ).fillna(0.0)
        main_df["product_category_name_english"] = main_df[
            "product_category_name_english"
        ].astype("string")
        return main_df, payments

    @staticmethod
    def _last_three_months(max_month: pd.Period) -> list[pd.Period]:
        return [max_month - 2, max_month - 1, max_month]

    @staticmethod
    def _safe_float(value: Any) -> float:
        if pd.isna(value):
            return 0.0
        return float(value)

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")
        return json.loads(raw[start : end + 1])

    def _compute_metrics_for_pair(
        self,
        main_df: pd.DataFrame,
        payments: pd.DataFrame,
        state: str,
        category: str,
        months: list[pd.Period],
    ) -> dict[str, Any]:
        scoped = main_df[
            (main_df["customer_state"] == state)
            & (main_df["product_category_name_english"] == category)
            & (main_df["month"].isin(months))
        ].copy()

        if scoped.empty:
            return {
                "avg_spend": 0.0,
                "order_volume_trend": 0.0,
                "top_payment_type": "credit_card",
                "high_value_customer_count": 0,
                "repeat_rate": 0.0,
                "installment_pct": 0.0,
            }

        per_order = scoped.drop_duplicates(subset="order_id")[
            [
                "order_id",
                "customer_unique_id",
                "customer_state",
                "product_category_name_english",
                "total_payment_value",
                "month",
            ]
        ].copy()

        avg_spend = self._safe_float(per_order["total_payment_value"].mean())

        monthly_orders = (
            scoped.groupby("month", dropna=True)["order_id"]
            .nunique()
            .reindex(months, fill_value=0)
        )
        prev_orders = int(monthly_orders.iloc[-2]) if len(monthly_orders) >= 2 else 0
        curr_orders = int(monthly_orders.iloc[-1]) if len(monthly_orders) >= 1 else 0
        if prev_orders <= 0:
            order_volume_trend = 0.0
        else:
            order_volume_trend = self._safe_float(
                (curr_orders - prev_orders) / max(prev_orders, 1)
            )

        relevant_order_ids = per_order["order_id"].dropna().unique().tolist()
        relevant_payments = payments[payments["order_id"].isin(relevant_order_ids)].copy()
        payment_mode = (
            relevant_payments["payment_type"].mode().iloc[0]
            if not relevant_payments.empty
            else "credit_card"
        )
        top_payment_type = str(payment_mode)

        median_spend = self._safe_float(per_order["total_payment_value"].median())
        customer_avg = (
            per_order.groupby("customer_unique_id", dropna=True)["total_payment_value"]
            .mean()
            .fillna(0.0)
        )
        high_value_customer_count = int((customer_avg > median_spend).sum())

        repeat_counts = scoped.groupby("customer_unique_id", dropna=True)["order_id"].nunique()
        total_customers = int(repeat_counts.shape[0])
        repeat_rate = (
            self._safe_float((repeat_counts > 1).sum() / total_customers)
            if total_customers > 0
            else 0.0
        )

        card_rows = relevant_payments[relevant_payments["payment_type"] == "credit_card"]
        card_orders = card_rows.drop_duplicates(subset="order_id")
        installment_pct = (
            self._safe_float(card_orders["payment_installments"].ge(6).mean())
            if not card_orders.empty
            else 0.0
        )

        return {
            "avg_spend": avg_spend,
            "order_volume_trend": order_volume_trend,
            "top_payment_type": top_payment_type,
            "high_value_customer_count": high_value_customer_count,
            "repeat_rate": repeat_rate,
            "installment_pct": installment_pct,
        }

    def _assess_with_llm(self, state: str, category: str, month: str, metrics: dict[str, Any]) -> dict[str, Any]:
        payload = (
            f"Customer metrics for {category} in {state} state, training to {month}:\n"
            f"- avg_spend: R${metrics['avg_spend']:.2f}\n"
            f"- order_volume_trend: {metrics['order_volume_trend']:+.1%} MoM\n"
            f"- top_payment_type: {metrics['top_payment_type']}\n"
            f"- high_value_customer_count: {metrics['high_value_customer_count']}\n"
            f"- repeat_rate: {metrics['repeat_rate']:.1%}\n"
            f"- installment_pct: {metrics['installment_pct']:.1%}"
        )
        question = (
            'Assess customer readiness and return JSON: '
            '{"readiness":"HIGH|MEDIUM|LOW","reasoning":"...","risk_flags":[...]}'
        )
        messages = build_analyst_prompt(payload, question)
        raw = self.llm_client(messages, model=self.model, max_tokens=250)
        parsed = self._extract_json(raw)
        readiness = str(parsed.get("readiness", "MEDIUM")).upper()
        if readiness not in READINESS_VALUES:
            raise ValueError("Invalid readiness value.")
        risk_flags = parsed.get("risk_flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = ["agent_failed"]
        return {
            "readiness": readiness,
            "reasoning": str(parsed.get("reasoning", "LLM readiness assessment")),
            "risk_flags": [str(x) for x in risk_flags],
        }

    def _assess_batch(
        self,
        items: list[dict[str, Any]],
        month: str,
        prev_memory: dict[str, Any] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Send all state×category metrics in one LLM call.

        Returns {(state, category): {"readiness", "reasoning", "risk_flags"}}.
        Falls back to _fallback_assessment() for any item not returned by LLM.
        """
        batch_payload = []
        for item in items:
            payload_item = {
                "state": item["state"],
                "category": item["category"],
                "avg_spend": round(item["metrics"]["avg_spend"], 2),
                "order_volume_trend": round(item["metrics"]["order_volume_trend"], 4),
                "top_payment_type": item["metrics"]["top_payment_type"],
                "high_value_customer_count": item["metrics"]["high_value_customer_count"],
                "repeat_rate": round(item["metrics"]["repeat_rate"], 4),
                "installment_pct": round(item["metrics"]["installment_pct"], 4),
            }
            last_month_context = self._format_last_month_context(
                prev_memory=prev_memory,
                state=item["state"],
                category=item["category"],
            )
            if last_month_context is not None:
                payload_item["last_month_context"] = last_month_context
            batch_payload.append(payload_item)
        question = (
            f"Training month: {month}. Assess customer readiness for each item. "
            "Use last_month_context when present to keep temporal continuity. "
            "Return ONLY a JSON array where each object has exactly: "
            "{\"state\": <string>, \"category\": <string>, "
            "\"readiness\": \"HIGH|MEDIUM|LOW\", "
            "\"reasoning\": <string>, \"risk_flags\": [<string>, ...]}. "
            "One entry per input item in any order."
        )
        try:
            messages = build_analyst_prompt(
                json.dumps(batch_payload, indent=2), question
            )
            raw = self.llm_client(messages, model=self.model, max_tokens=4000)
            parsed_items = parse_batch_llm_response(raw, items)
            result: dict[tuple[str, str], dict[str, Any]] = {}
            for item, parsed in zip(items, parsed_items):
                key = (item["state"], item["category"])
                if parsed is not None and isinstance(parsed, dict):
                    readiness = str(parsed.get("readiness", "MEDIUM")).upper()
                    if readiness not in READINESS_VALUES:
                        readiness = "MEDIUM"
                    risk_flags = parsed.get("risk_flags", [])
                    if not isinstance(risk_flags, list):
                        risk_flags = ["agent_failed"]
                    result[key] = {
                        "readiness": readiness,
                        "reasoning": str(parsed.get("reasoning", "LLM readiness assessment")),
                        "risk_flags": [str(x) for x in risk_flags],
                    }
                else:
                    result[key] = self._fallback_assessment()
            return result
        except Exception:
            return {
                (item["state"], item["category"]): self._fallback_assessment()
                for item in items
            }

    @staticmethod
    def _fallback_assessment() -> dict[str, Any]:
        return {
            "readiness": "MEDIUM",
            "reasoning": "Fallback readiness due to LLM parsing failure.",
            "risk_flags": ["agent_failed"],
        }

    @staticmethod
    def _get_prev_pair_memory(
        prev_memory: dict[str, Any] | None, state: str, category: str
    ) -> dict[str, Any] | None:
        if not isinstance(prev_memory, dict):
            return None

        direct = prev_memory.get((state, category))
        if isinstance(direct, dict):
            return direct

        pipe_key = prev_memory.get(f"{state}|{category}")
        if isinstance(pipe_key, dict):
            return pipe_key

        colon_key = prev_memory.get(f"{state}:{category}")
        if isinstance(colon_key, dict):
            return colon_key

        nested = prev_memory.get(state)
        if isinstance(nested, dict):
            nested_pair = nested.get(category)
            if isinstance(nested_pair, dict):
                return nested_pair

        return None

    def _format_last_month_context(
        self, prev_memory: dict[str, Any] | None, state: str, category: str
    ) -> str | None:
        previous = self._get_prev_pair_memory(prev_memory, state, category)
        if not previous:
            return None

        readiness = str(previous.get("readiness", previous.get("cr_readiness", "UNKNOWN")))
        risk_flags = previous.get("risk_flags", [])
        if isinstance(risk_flags, list):
            risk_text = "|".join(str(flag) for flag in risk_flags) or "none"
        else:
            risk_text = str(risk_flags)
        reasoning = str(previous.get("reasoning", previous.get("cr_reasoning", "n/a")))
        return (
            f"Last month: readiness={readiness}, "
            f"risk_flags={risk_text}, reasoning={reasoning}"
        )

    def _write_memory(self, output: CustomerReadinessOutput) -> None:
        if self.memory is None or not hasattr(self.memory, "write_row"):
            return
        self.memory.write_row(
            output["state"],
            output["category"],
            output["month"],
            cr_avg_spend=output["avg_spend"],
            cr_order_volume_trend=output["order_volume_trend"],
            cr_top_payment_type=output["top_payment_type"],
            cr_high_value_customer_count=output["high_value_customer_count"],
            cr_repeat_rate=output["repeat_rate"],
            cr_reasoning=output["reasoning"],
        )

    def run(
        self,
        training_df: pd.DataFrame | None = None,
        states: list[str] | None = None,
        categories: list[str] | None = None,
        month: str | None = None,
        prev_memory: dict[str, Any] | None = None,
    ) -> list[CustomerReadinessOutput]:
        main_df, payments = self._load_customer_data(training_df=training_df)

        focus_states = states if states is not None else FOCUS_STATES
        focus_categories = categories if categories is not None else FOCUS_CATEGORIES

        if training_df is not None and "month" in training_df.columns and not training_df.empty:
            max_month = pd.PeriodIndex(training_df["month"].astype(str), freq="M").max()
        else:
            max_month = main_df["month"].max()
        months = self._last_three_months(max_month)
        output_month = month if month is not None else str(max_month)

        # --- Phase 1: compute all metrics (pandas only) ---
        batch_items: list[dict[str, Any]] = []
        for state in focus_states:
            for category in focus_categories:
                metrics = self._compute_metrics_for_pair(
                    main_df=main_df,
                    payments=payments,
                    state=state,
                    category=category,
                    months=months,
                )
                batch_items.append({"state": state, "category": category, "metrics": metrics})

        # --- Phase 2: one batch LLM call for all pairs ---
        batch_assessments = self._assess_batch(
            batch_items,
            output_month,
            prev_memory=prev_memory,
        )

        # --- Phase 3: assemble outputs ---
        outputs: list[CustomerReadinessOutput] = []
        for item in batch_items:
            state = item["state"]
            category = item["category"]
            metrics = item["metrics"]
            assessment = batch_assessments.get((state, category), self._fallback_assessment())

            output: CustomerReadinessOutput = {
                "agent": self.AGENT_NAME,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": state,
                "category": category,
                "month": output_month,
                "avg_spend": self._safe_float(metrics["avg_spend"]),
                "order_volume_trend": self._safe_float(metrics["order_volume_trend"]),
                "top_payment_type": str(metrics["top_payment_type"] or "credit_card"),
                "high_value_customer_count": int(metrics["high_value_customer_count"]),
                "repeat_rate": self._safe_float(metrics["repeat_rate"]),
                "installment_pct": self._safe_float(metrics["installment_pct"]),
                "readiness": str(assessment["readiness"]),
                "reasoning": str(assessment["reasoning"]),
                "risk_flags": [str(x) for x in assessment["risk_flags"]],
            }
            outputs.append(output)
            self._write_memory(output)
        return outputs
