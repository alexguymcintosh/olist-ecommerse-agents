"""Supply quality agent: seller health and fulfillment confidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from utils.config import MIN_MONTHLY_ORDERS
from utils.data_loader import load_all
try:
    from utils.openrouter_client import RND_MODEL, build_analyst_prompt, parse_batch_llm_response, query_llm
except ModuleNotFoundError:  # pragma: no cover - local test fallback
    RND_MODEL = "deepseek/deepseek-v3.2"

    def build_analyst_prompt(data_summary: str, question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a business intelligence analyst."},
            {"role": "user", "content": f"DATA:\n{data_summary}\n\nQUESTION:\n{question}"},
        ]

    def query_llm(messages: list[dict[str, str]], model: str = RND_MODEL, max_tokens: int = 1000) -> str:
        del messages, model, max_tokens
        raise RuntimeError("LLM client unavailable in local environment.")

    def parse_batch_llm_response(raw: str, items: list[dict], **_: object) -> list[dict | None]:
        return [None] * len(items)
from utils.schema_agents import SupplyQualityOutput


class SupplyQualityAgent:
    """Compute seller-side health signals for state x category pairs."""

    AGENT_NAME = "supply_quality"

    def __init__(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        model: str = RND_MODEL,
    ) -> None:
        self.data = data if data is not None else load_all()
        self.model = model

    @staticmethod
    def _to_month_period(value: str | pd.Period) -> pd.Period:
        if isinstance(value, pd.Period):
            return value.asfreq("M")
        return pd.Period(str(value), freq="M")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            if np.isnan(parsed) or np.isinf(parsed):
                return default
            return parsed
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _map_churn_risk(churn_rate: float) -> str:
        if churn_rate > 0.3:
            return "HIGH"
        if churn_rate >= 0.1:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _extract_json_dict(raw: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")
        parsed = json.loads(raw[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not a JSON object.")
        return parsed

    @staticmethod
    def _ensure_month_column(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "month" not in out.columns:
            out["order_purchase_timestamp"] = pd.to_datetime(
                out["order_purchase_timestamp"], errors="coerce"
            )
            out["month"] = out["order_purchase_timestamp"].dt.to_period("M")
        else:
            out["month"] = out["month"].apply(SupplyQualityAgent._to_month_period)
        return out

    @staticmethod
    def _build_training_df_from_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        orders = data["orders"].copy()
        order_items = data["order_items"].copy()
        sellers = data["sellers"].copy()
        reviews = data["reviews"].copy()
        products = data["products"].copy()
        categories = data["categories"].copy()

        merged = (
            order_items.merge(orders, on="order_id", how="inner")
            .merge(sellers, on="seller_id", how="left")
            .merge(reviews, on="order_id", how="left")
            .merge(products, on="product_id", how="left")
            .merge(categories, on="product_category_name", how="left")
        )
        merged["order_purchase_timestamp"] = pd.to_datetime(
            merged["order_purchase_timestamp"], errors="coerce"
        )
        merged["order_delivered_customer_date"] = pd.to_datetime(
            merged.get("order_delivered_customer_date"), errors="coerce"
        )
        merged["price"] = pd.to_numeric(merged.get("price"), errors="coerce").fillna(0.0)
        merged["review_score"] = pd.to_numeric(
            merged.get("review_score"), errors="coerce"
        )
        merged["month"] = merged["order_purchase_timestamp"].dt.to_period("M")
        return merged

    def _enrich_training_df(self, training_df: pd.DataFrame) -> pd.DataFrame:
        """Attach item/seller/category fields required for supply metrics."""
        required_columns = {
            "seller_id",
            "seller_state",
            "price",
            "product_category_name_english",
            "review_score",
        }
        if required_columns.issubset(training_df.columns):
            return training_df.copy()

        required_tables = {"order_items", "sellers", "products", "categories", "reviews"}
        if not required_tables.issubset(self.data.keys()):
            return training_df.copy()

        base = training_df.copy()
        drop_columns = [
            "seller_id",
            "seller_state",
            "product_id",
            "price",
            "product_category_name",
            "product_category_name_english",
            "review_score",
        ]
        existing_drop_columns = [column for column in drop_columns if column in base.columns]
        if existing_drop_columns:
            base = base.drop(columns=existing_drop_columns)

        items = self.data["order_items"][["order_id", "seller_id", "product_id", "price"]].copy()
        sellers = self.data["sellers"][["seller_id", "seller_state"]].copy()
        products = self.data["products"][["product_id", "product_category_name"]].copy()
        categories = self.data["categories"][
            ["product_category_name", "product_category_name_english"]
        ].copy()
        reviews = self.data["reviews"][["order_id", "review_score"]].copy()

        enriched = (
            base.merge(items, on="order_id", how="left")
            .merge(sellers, on="seller_id", how="left")
            .merge(products, on="product_id", how="left")
            .merge(categories, on="product_category_name", how="left")
            .merge(reviews, on="order_id", how="left")
        )
        return enriched

    def _compute_weighted_review_score(
        self, scoped: pd.DataFrame, latest_month: pd.Period
    ) -> float:
        last_three = [latest_month - 2, latest_month - 1, latest_month]
        recent = scoped[scoped["month"].isin(last_three)].copy()
        recent = recent.dropna(subset=["review_score"])
        if recent.empty:
            return 3.0
        month_weights = {
            last_three[0]: 1.0,
            last_three[1]: 2.0,
            last_three[2]: 3.0,
        }
        recent["weight"] = recent["month"].map(month_weights).astype(float)
        weighted = np.average(recent["review_score"].astype(float), weights=recent["weight"])
        return self._safe_float(weighted, default=3.0)

    def _compute_avg_delivery_days(self, scoped: pd.DataFrame) -> float:
        delivered = scoped[scoped["order_status"] == "delivered"].copy()
        if delivered.empty:
            return 0.0
        deltas = (
            delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
        ).dt.days
        if deltas.dropna().empty:
            return 0.0
        return self._safe_float(deltas.mean(), default=0.0)

    def _compute_churn_rate(self, scoped: pd.DataFrame, latest_month: pd.Period) -> float:
        months_available = sorted(scoped["month"].dropna().unique())
        if len(months_available) < 7:
            return 0.0

        prior_window = {latest_month - 7, latest_month - 6, latest_month - 5}
        previous_month = latest_month - 1
        sellers_prior = set(
            scoped.loc[scoped["month"].isin(prior_window), "seller_id"].dropna().astype(str)
        )
        sellers_previous = set(
            scoped.loc[scoped["month"] == previous_month, "seller_id"].dropna().astype(str)
        )
        all_sellers = set(scoped["seller_id"].dropna().astype(str))
        denominator = max(len(all_sellers), 1)
        churned = sellers_prior - sellers_previous
        return self._safe_float(len(churned) / denominator, default=0.0)

    def _compute_top_seller_and_concentration(self, scoped: pd.DataFrame) -> tuple[str, float]:
        seller_count = int(scoped["seller_id"].nunique())
        if seller_count == 0:
            return "", 0.0

        scoped = scoped.copy()
        scoped["seller_id"] = scoped["seller_id"].astype(str)
        seller_orders = scoped.groupby("seller_id")["order_id"].count()
        top_seller_id = str(seller_orders.idxmax())

        top_revenue = self._safe_float(
            scoped.loc[scoped["seller_id"] == top_seller_id, "price"].sum()
        )
        total_revenue = max(self._safe_float(scoped["price"].sum(), default=0.0), 0.01)
        concentration = self._safe_float(top_revenue / total_revenue, default=0.0)
        return top_seller_id, concentration

    def _assess_supply(
        self,
        *,
        state: str,
        category: str,
        month: str,
        seller_count: int,
        avg_review_score: float,
        avg_delivery_days: float,
        churn_risk: str,
        churn_rate: float,
        top_seller_id: str,
        seller_concentration: float,
    ) -> tuple[str, str, list[str]]:
        data_summary = (
            f"state={state}, category={category}, month={month}\n"
            f"seller_count={seller_count}\n"
            f"avg_review_score={avg_review_score:.4f}\n"
            f"avg_delivery_days={avg_delivery_days:.4f}\n"
            f"churn_risk={churn_risk}\n"
            f"churn_rate={churn_rate:.4f}\n"
            f"top_seller_id={top_seller_id}\n"
            f"seller_concentration={seller_concentration:.4f}\n"
            "Return only JSON."
        )
        question = (
            "Assess supply confidence. Return ONLY JSON with keys: "
            '{"supply_confidence":"STRONG|ADEQUATE|WEAK","reasoning":"...","risk_flags":["..."]}'
        )

        try:
            messages = build_analyst_prompt(data_summary, question)
            raw = query_llm(messages, model=self.model, max_tokens=300)
            parsed = self._extract_json_dict(raw)
            confidence = str(parsed.get("supply_confidence", "WEAK")).upper()
            if confidence not in {"STRONG", "ADEQUATE", "WEAK"}:
                confidence = "WEAK"
            reasoning = str(parsed.get("reasoning", "LLM assessment unavailable."))
            risk_flags_raw = parsed.get("risk_flags", [])
            if isinstance(risk_flags_raw, list):
                risk_flags = [str(flag) for flag in risk_flags_raw]
            else:
                risk_flags = []
            return confidence, reasoning, risk_flags
        except Exception:
            return "WEAK", "Fallback: LLM response parse failed.", ["agent_failed"]

    def _assess_supply_batch(
        self,
        items: list[dict[str, Any]],
        month: str,
        prev_memory: dict[tuple[str, str], dict] | None = None,
    ) -> dict[tuple[str, str], tuple[str, str, list[str]]]:
        """Send all non-sparse state×category pairs in one LLM call.

        Returns {(state, category): (supply_confidence, reasoning, risk_flags)}.
        Falls back to ("WEAK", "...", ["agent_failed"]) for any missing item.
        """
        batch_payload = []
        for item in items:
            payload_item = {
                "state": item["state"],
                "category": item["category"],
                "month": month,
                "seller_count": item["seller_count"],
                "avg_review_score": round(item["avg_review_score"], 4),
                "avg_delivery_days": round(item["avg_delivery_days"], 4),
                "churn_risk": item["churn_risk"],
                "churn_rate": round(item["churn_rate"], 4),
                "top_seller_id": item["top_seller_id"],
                "seller_concentration": round(item["seller_concentration"], 4),
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
            "Assess supply confidence for each item. "
            "Use last_month_context when present to maintain historical continuity. "
            "Return ONLY a JSON array where each object has exactly: "
            "{\"state\": <string>, \"category\": <string>, "
            "\"supply_confidence\": \"STRONG|ADEQUATE|WEAK\", "
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
                    confidence = str(parsed.get("supply_confidence", "WEAK")).upper()
                    if confidence not in {"STRONG", "ADEQUATE", "WEAK"}:
                        confidence = "WEAK"
                    reasoning = str(parsed.get("reasoning", "LLM assessment unavailable."))
                    flags_raw = parsed.get("risk_flags", [])
                    risk_flags = [str(f) for f in flags_raw] if isinstance(flags_raw, list) else []
                else:
                    confidence = "WEAK"
                    reasoning = "Fallback: LLM response parse failed."
                    risk_flags = ["agent_failed"]
                result[key] = (confidence, reasoning, risk_flags)
            return result
        except Exception:
            return {
                (item["state"], item["category"]): (
                    "WEAK", "Fallback: LLM response parse failed.", ["agent_failed"]
                )
                for item in items
            }

    @staticmethod
    def _dedupe_flags(flags: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for flag in flags:
            if flag not in seen:
                seen.add(flag)
                ordered.append(flag)
        return ordered

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

        supply_confidence = str(
            previous.get("supply_confidence", previous.get("sq_supply_confidence", "UNKNOWN"))
        )
        churn_risk = str(previous.get("churn_risk", previous.get("sq_churn_risk", "UNKNOWN")))
        reasoning = str(previous.get("reasoning", previous.get("sq_reasoning", "n/a")))
        return (
            f"Last month: supply_confidence={supply_confidence}, "
            f"churn_risk={churn_risk}, reasoning={reasoning}"
        )

    def run(
        self,
        training_df: pd.DataFrame | None,
        states: list[str],
        categories: list[str],
        month: str,
        prev_memory: dict[tuple[str, str], dict] | None = None,
    ) -> list[SupplyQualityOutput]:
        if training_df is None:
            training_df = self._build_training_df_from_data(self.data)
        else:
            training_df = self._enrich_training_df(training_df)
        training_df = self._ensure_month_column(training_df)

        training_df["price"] = pd.to_numeric(training_df.get("price"), errors="coerce").fillna(0.0)
        training_df["review_score"] = pd.to_numeric(
            training_df.get("review_score"), errors="coerce"
        )
        training_df["order_purchase_timestamp"] = pd.to_datetime(
            training_df.get("order_purchase_timestamp"), errors="coerce"
        )
        training_df["order_delivered_customer_date"] = pd.to_datetime(
            training_df.get("order_delivered_customer_date"), errors="coerce"
        )

        latest_month = training_df["month"].max()
        if pd.isna(latest_month):
            latest_month = self._to_month_period(month) - 1
        latest_month = self._to_month_period(latest_month)

        timestamp = datetime.now(timezone.utc).isoformat()

        # --- Phase 1: compute all metrics without LLM ---
        all_metrics: list[dict[str, Any]] = []
        for state in states:
            for category in categories:
                scoped = training_df[
                    (training_df["seller_state"] == state)
                    & (training_df["product_category_name_english"] == category)
                ].copy()

                latest_scoped = scoped[scoped["month"] == latest_month]
                latest_order_count = int(latest_scoped["order_id"].nunique())
                seller_count = int(scoped["seller_id"].nunique())
                avg_review_score = self._compute_weighted_review_score(scoped, latest_month)
                avg_delivery_days = self._compute_avg_delivery_days(scoped)
                churn_rate = self._compute_churn_rate(scoped, latest_month)
                churn_risk = self._map_churn_risk(churn_rate)
                top_seller_id, seller_concentration = self._compute_top_seller_and_concentration(
                    scoped
                )
                is_sparse = latest_order_count < MIN_MONTHLY_ORDERS

                all_metrics.append(
                    {
                        "state": state,
                        "category": category,
                        "seller_count": seller_count,
                        "avg_review_score": self._safe_float(avg_review_score, default=3.0),
                        "avg_delivery_days": self._safe_float(avg_delivery_days, default=0.0),
                        "churn_risk": churn_risk,
                        "churn_rate": self._safe_float(churn_rate, default=0.0),
                        "top_seller_id": top_seller_id,
                        "seller_concentration": self._safe_float(seller_concentration, default=0.0),
                        "is_sparse": is_sparse,
                    }
                )

        # --- Phase 2: one batch LLM call for all non-sparse pairs ---
        non_sparse = [m for m in all_metrics if not m["is_sparse"]]
        if non_sparse:
            batch_results = self._assess_supply_batch(
                non_sparse,
                month,
                prev_memory=prev_memory,
            )
        else:
            batch_results = {}

        # --- Phase 3: assemble outputs ---
        outputs: list[SupplyQualityOutput] = []
        for m in all_metrics:
            state = m["state"]
            category = m["category"]
            risk_flags: list[str] = []

            if m["is_sparse"]:
                supply_confidence = "WEAK"
                reasoning = "Sparse latest-month volume; LLM assessment skipped."
                risk_flags.append("sparse_data")
            else:
                supply_confidence, reasoning, risk_flags = batch_results.get(
                    (state, category),
                    ("WEAK", "Fallback: LLM response parse failed.", ["agent_failed"]),
                )
                risk_flags = list(risk_flags)  # ensure mutable copy

            if m["seller_count"] == 0:
                risk_flags.append("critical_seller_gap")
            if m["churn_rate"] > 0.5:
                risk_flags.append("high_seller_churn")

            outputs.append(
                {
                    "agent": self.AGENT_NAME,
                    "timestamp": timestamp,
                    "state": state,
                    "category": category,
                    "month": month,
                    "seller_count": m["seller_count"],
                    "avg_review_score": m["avg_review_score"],
                    "avg_delivery_days": m["avg_delivery_days"],
                    "churn_risk": m["churn_risk"],
                    "churn_rate": m["churn_rate"],
                    "top_seller_id": m["top_seller_id"],
                    "seller_concentration": m["seller_concentration"],
                    "supply_confidence": supply_confidence,
                    "reasoning": reasoning,
                    "risk_flags": self._dedupe_flags(risk_flags),
                }
            )

        return outputs
