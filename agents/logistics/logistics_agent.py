"""Logistics feasibility agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import FOCUS_CATEGORIES, FOCUS_STATES
from utils.data_loader import load_all
from utils.openrouter_client import RND_MODEL, build_analyst_prompt, parse_batch_llm_response, query_llm
from utils.schema_agents import LogisticsOutput

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FEASIBILITY_VALUES = {"STRONG", "ADEQUATE", "WEAK"}


class LogisticsAgent:
    """Assess delivery feasibility by state/category/month."""

    AGENT_NAME = "logistics"

    def __init__(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        memory: Any | None = None,
        model: str = RND_MODEL,
    ) -> None:
        self.data = data if data is not None else load_all()
        self.memory = memory
        self.model = model

    def _load_categories(self) -> pd.DataFrame:
        if "categories" in self.data:
            return self.data["categories"].copy()
        return pd.read_csv(
            DATA_DIR / "product_category_name_translation.csv", encoding="utf-8-sig"
        )

    def _load_joined_data(self, training_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if training_df is not None:
            joined = training_df.copy()
            if "seller_id" in joined.columns:
                seller_ids = joined["seller_id"].dropna().astype(str).unique().tolist()
                sellers = self.data["sellers"][["seller_id", "seller_state"]].copy()
                if seller_ids:
                    sellers = sellers[
                        sellers["seller_id"].astype(str).isin(seller_ids)
                    ].copy()
            else:
                sellers = self.data["sellers"][["seller_id", "seller_state"]].copy()
            if "seller_state" in joined.columns:
                joined = joined.drop(columns=["seller_state"])
            joined = joined.merge(sellers, on="seller_id", how="left")
        else:
            orders = self.data["orders"].copy()
            order_items = self.data["order_items"].copy()
            sellers = self.data["sellers"].copy()
            customers = self.data["customers"].copy()
            products = self.data["products"].copy()
            categories = self._load_categories()

            joined = (
                orders.merge(order_items, on="order_id", how="inner")
                .merge(sellers, on="seller_id", how="left")
                .merge(customers, on="customer_id", how="inner")
                .merge(products, on="product_id", how="left")
                .merge(categories, on="product_category_name", how="left")
            )

        joined["order_purchase_timestamp"] = pd.to_datetime(
            joined["order_purchase_timestamp"], errors="coerce"
        )
        joined["order_delivered_customer_date"] = pd.to_datetime(
            joined["order_delivered_customer_date"], errors="coerce"
        )
        joined["order_estimated_delivery_date"] = pd.to_datetime(
            joined["order_estimated_delivery_date"], errors="coerce"
        )
        joined["price"] = pd.to_numeric(joined["price"], errors="coerce")
        joined["freight_value"] = pd.to_numeric(joined["freight_value"], errors="coerce")
        return joined

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if pd.isna(parsed):
            return default
        return parsed

    def _compute_metrics(
        self, delivered_df: pd.DataFrame, state: str, category: str
    ) -> dict[str, float | str]:
        scoped = delivered_df[
            (delivered_df["customer_state"] == state)
            & (delivered_df["product_category_name_english"] == category)
        ].copy()

        if scoped.empty:
            return {
                "avg_delivery_days": 0.0,
                "pct_on_time": 0.0,
                "freight_ratio": 0.0,
                "fastest_seller_state": "",
                "delivery_variance": 0.0,
                "cross_state_dependency": 0.0,
            }

        delivery_days = (
            scoped["order_delivered_customer_date"] - scoped["order_purchase_timestamp"]
        ).dt.days
        delivery_days = pd.to_numeric(delivery_days, errors="coerce").dropna()

        avg_delivery_days = (
            self._safe_float(delivery_days.mean(), 0.0)
            if not delivery_days.empty
            else 0.0
        )

        on_time = (
            scoped["order_delivered_customer_date"] <= scoped["order_estimated_delivery_date"]
        )
        pct_on_time = self._safe_float(on_time.mean(), 0.0)

        mean_price = self._safe_float(scoped["price"].mean(), 0.0)
        mean_freight = self._safe_float(scoped["freight_value"].mean(), 0.0)
        freight_ratio = mean_freight / max(mean_price, 0.01)

        if delivery_days.shape[0] >= 2:
            delivery_variance = self._safe_float(delivery_days.std(), 0.0)
        else:
            delivery_variance = 0.0

        cross_state_dependency = self._safe_float(
            (scoped["seller_state"] != scoped["customer_state"]).mean(), 0.0
        )

        route_stats = (
            scoped.assign(delivery_days=(scoped["order_delivered_customer_date"] - scoped["order_purchase_timestamp"]).dt.days)
            .dropna(subset=["delivery_days"])
            .groupby("seller_state", dropna=True)["delivery_days"]
            .mean()
        )
        fastest_seller_state = (
            str(route_stats.idxmin()) if not route_stats.empty else ""
        )

        return {
            "avg_delivery_days": avg_delivery_days,
            "pct_on_time": pct_on_time,
            "freight_ratio": self._safe_float(freight_ratio, 0.0),
            "fastest_seller_state": fastest_seller_state,
            "delivery_variance": delivery_variance,
            "cross_state_dependency": cross_state_dependency,
        }

    def _build_prompt(self, month: str, state: str, category: str, metrics: dict[str, float | str]) -> list[dict[str, str]]:
        summary = (
            f"Logistics metrics for {category} -> {state} customers, training to {month}:\n"
            f"- avg_delivery_days: {self._safe_float(metrics['avg_delivery_days']):.1f}\n"
            f"- pct_on_time: {self._safe_float(metrics['pct_on_time']):.1%}\n"
            f"- freight_ratio: {self._safe_float(metrics['freight_ratio']):.2f}\n"
            f"- fastest_seller_state: {metrics['fastest_seller_state']}\n"
            f"- delivery_variance: {self._safe_float(metrics['delivery_variance']):.1f} days std dev\n"
            f"- cross_state_dependency: {self._safe_float(metrics['cross_state_dependency']):.1%}"
        )
        question = (
            "Assess logistics feasibility and return JSON:\n"
            "{\"feasibility\": \"STRONG|ADEQUATE|WEAK\", \"reasoning\": \"...\", \"risk_flags\": [...]}"
        )
        return build_analyst_prompt(summary, question)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM output")
        return json.loads(text[start : end + 1])

    def _parse_llm_response(self, raw: str) -> tuple[str, str, list[str]]:
        parsed = self._extract_json(raw)
        feasibility = str(parsed.get("feasibility", "WEAK")).upper()
        if feasibility not in FEASIBILITY_VALUES:
            feasibility = "WEAK"
        reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided."
        risk_flags_raw = parsed.get("risk_flags", [])
        risk_flags = [str(flag) for flag in risk_flags_raw] if isinstance(risk_flags_raw, list) else []
        return feasibility, reasoning, risk_flags

    @staticmethod
    def _get_prev_pair_memory(
        prev_memory: dict[tuple[str, str], dict] | None, state: str, category: str
    ) -> dict[str, Any] | None:
        if not prev_memory:
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
        self, prev_memory: dict[tuple[str, str], dict] | None, state: str, category: str
    ) -> str | None:
        previous = self._get_prev_pair_memory(prev_memory, state, category)
        if not previous:
            return None

        feasibility = str(previous.get("feasibility", previous.get("log_feasibility", "UNKNOWN")))
        risk_flags = previous.get("risk_flags", [])
        if isinstance(risk_flags, list):
            risk_text = "|".join(str(flag) for flag in risk_flags) or "none"
        else:
            risk_text = str(risk_flags)
        reasoning = str(previous.get("reasoning", previous.get("log_reasoning", "n/a")))
        return (
            f"Last month: feasibility={feasibility}, "
            f"risk_flags={risk_text}, reasoning={reasoning}"
        )

    def _llm_assessment(
        self, month: str, state: str, category: str, metrics: dict[str, float | str]
    ) -> tuple[str, str, list[str]]:
        prompt = self._build_prompt(month, state, category, metrics)
        raw = query_llm(prompt, model=self.model, max_tokens=300)
        return self._parse_llm_response(raw)

    def _llm_batch_assessment(
        self,
        month: str,
        items: list[dict[str, Any]],
        prev_memory: dict[tuple[str, str], dict] | None = None,
    ) -> dict[tuple[str, str], tuple[str, str, list[str]]]:
        """Send all state×category metrics in one LLM call.

        Returns {(state, category): (feasibility, reasoning, risk_flags)}.
        Falls back to WEAK/agent_failed for any item not returned.
        """
        batch_payload = []
        for item in items:
            payload_item = {
                "state": item["state"],
                "category": item["category"],
                "avg_delivery_days": round(self._safe_float(item["metrics"]["avg_delivery_days"]), 1),
                "pct_on_time": round(self._safe_float(item["metrics"]["pct_on_time"]), 4),
                "freight_ratio": round(self._safe_float(item["metrics"]["freight_ratio"]), 2),
                "fastest_seller_state": item["metrics"]["fastest_seller_state"],
                "delivery_variance": round(self._safe_float(item["metrics"]["delivery_variance"]), 1),
                "cross_state_dependency": round(
                    self._safe_float(item["metrics"]["cross_state_dependency"]), 4
                ),
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
            f"Training month: {month}. Assess logistics feasibility for each item. "
            "Use last_month_context when present to keep temporal continuity. "
            "Return ONLY a JSON array where each object has exactly: "
            "{\"state\": <string>, \"category\": <string>, "
            "\"feasibility\": \"STRONG|ADEQUATE|WEAK\", "
            "\"reasoning\": <string>, \"risk_flags\": [<string>, ...]}. "
            "One entry per input item in any order."
        )
        try:
            messages = build_analyst_prompt(json.dumps(batch_payload, indent=2), question)
            raw = query_llm(messages, model=self.model, max_tokens=4000)
            parsed_items = parse_batch_llm_response(raw, items)
            result: dict[tuple[str, str], tuple[str, str, list[str]]] = {}
            for item, parsed in zip(items, parsed_items):
                key = (item["state"], item["category"])
                if parsed is not None and isinstance(parsed, dict):
                    feasibility = str(parsed.get("feasibility", "WEAK")).upper()
                    if feasibility not in FEASIBILITY_VALUES:
                        feasibility = "WEAK"
                    reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided."
                    flags_raw = parsed.get("risk_flags", [])
                    risk_flags = [str(f) for f in flags_raw] if isinstance(flags_raw, list) else []
                else:
                    feasibility = "WEAK"
                    reasoning = "Fallback used because logistics LLM response parsing failed."
                    risk_flags = ["agent_failed"]
                result[key] = (feasibility, reasoning, risk_flags)
            return result
        except Exception:
            return {
                (item["state"], item["category"]): (
                    "WEAK",
                    "Fallback used because logistics LLM response parsing failed.",
                    ["agent_failed"],
                )
                for item in items
            }

    def _write_memory(self, month: str, state: str, category: str, output: LogisticsOutput) -> None:
        if self.memory is None or not hasattr(self.memory, "write_row"):
            return
        self.memory.write_row(
            state,
            category,
            month,
            log_avg_delivery_days=output["avg_delivery_days"],
            log_pct_on_time=output["pct_on_time"],
            log_freight_ratio=output["freight_ratio"],
            log_fastest_seller_state=output["fastest_seller_state"],
            log_reasoning=output["reasoning"],
        )

    def run(
        self,
        month: str,
        training_df: pd.DataFrame | None = None,
        states: list[str] | None = None,
        categories: list[str] | None = None,
        prev_memory: dict[tuple[str, str], dict] | None = None,
    ) -> list[LogisticsOutput]:
        joined = self._load_joined_data(training_df=training_df)
        delivered = joined[joined["order_status"] == "delivered"].copy()

        timestamp = datetime.now(timezone.utc).isoformat()
        focus_states = states if states is not None else FOCUS_STATES
        focus_categories = categories if categories is not None else FOCUS_CATEGORIES

        # --- Phase 1: compute all metrics (pandas only) ---
        all_items: list[dict[str, Any]] = []
        for state in focus_states:
            for category in focus_categories:
                metrics = self._compute_metrics(delivered, state, category)
                all_items.append({"state": state, "category": category, "metrics": metrics})

        # --- Phase 2: one batch LLM call for all pairs ---
        batch_results = self._llm_batch_assessment(
            month,
            all_items,
            prev_memory=prev_memory,
        )

        # --- Phase 3: assemble outputs ---
        outputs: list[LogisticsOutput] = []
        for item in all_items:
            state = item["state"]
            category = item["category"]
            metrics = item["metrics"]
            feasibility, reasoning, risk_flags = batch_results.get(
                (state, category),
                ("WEAK", "Fallback used because logistics LLM response parsing failed.", ["agent_failed"]),
            )

            output: LogisticsOutput = {
                "agent": self.AGENT_NAME,
                "timestamp": timestamp,
                "state": state,
                "category": category,
                "month": month,
                "avg_delivery_days": self._safe_float(metrics["avg_delivery_days"]),
                "pct_on_time": self._safe_float(metrics["pct_on_time"]),
                "freight_ratio": self._safe_float(metrics["freight_ratio"]),
                "fastest_seller_state": str(metrics["fastest_seller_state"]),
                "delivery_variance": self._safe_float(metrics["delivery_variance"]),
                "cross_state_dependency": self._safe_float(
                    metrics["cross_state_dependency"]
                ),
                "feasibility": feasibility,
                "reasoning": reasoning,
                "risk_flags": risk_flags,
            }
            self._write_memory(month, state, category, output)
            outputs.append(output)

        return outputs
