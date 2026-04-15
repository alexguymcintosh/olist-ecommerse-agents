"""Customer domain agent: demand, reviews, delivery, payments."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from utils.base_agent import BaseAgent


class CustomerAgent(BaseAgent):
    """Analyses orders, customers, reviews, and payments."""

    AGENT_NAME = "customer"

    def _prepare_data(self) -> pd.DataFrame:
        orders = self.data["orders"]
        customers = self.data["customers"]
        reviews = self.data["reviews"]
        payments = self.data["payments"]
        return (
            orders.merge(customers, on="customer_id")
            .merge(reviews, on="order_id", how="left")
            .merge(payments, on="order_id", how="left")
        )

    def _compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        # Order-level metrics: payments merge can duplicate rows per order.
        order_df = df.drop_duplicates(subset=["order_id"], keep="first")

        customer_order_counts = order_df.groupby("customer_id")["order_id"].nunique()
        if len(customer_order_counts) == 0:
            repeat_rate = 0.0
        else:
            repeat_rate = float((customer_order_counts > 1).sum() / len(customer_order_counts))

        review_scores = pd.to_numeric(order_df["review_score"], errors="coerce")
        avg_review_score = float(review_scores.mean()) if review_scores.notna().any() else 0.0
        if pd.isna(avg_review_score):
            avg_review_score = 0.0

        purchase = pd.to_datetime(order_df["order_purchase_timestamp"], errors="coerce")
        delivered = pd.to_datetime(order_df["order_delivered_customer_date"], errors="coerce")
        delivery_days = (delivered - purchase).dt.total_seconds() / 86400.0
        avg_delivery_days = float(delivery_days.mean()) if delivery_days.notna().any() else 0.0
        if pd.isna(avg_delivery_days):
            avg_delivery_days = 0.0

        est = pd.to_datetime(order_df["order_estimated_delivery_date"], errors="coerce")
        has_both = delivered.notna() & est.notna()
        if has_both.any():
            d_del = delivered[has_both].dt.normalize()
            d_est = est[has_both].dt.normalize()
            late = d_del > d_est
            pct_late_delivery = float(late.sum() / len(late))
        else:
            pct_late_delivery = 0.0

        pay = df["payment_type"].dropna()
        if pay.empty:
            top_payment_type = ""
        else:
            top_payment_type = str(pay.mode().iloc[0])

        return {
            "repeat_rate": repeat_rate,
            "avg_review_score": avg_review_score,
            "avg_delivery_days": avg_delivery_days,
            "pct_late_delivery": pct_late_delivery,
            "top_payment_type": top_payment_type,
        }

    def _build_question(self, metrics: dict[str, Any], sample_str: str) -> str:
        metrics_json = json.dumps(metrics, indent=2)
        return f"""Pre-computed metrics (authoritative — use only these numbers, do not infer from the sample):
{metrics_json}

Below is a sample of up to 50 rows for context (may be incomplete — do not compute metrics from it).

SAMPLE:
{sample_str}

Analyse the customer / order domain using the pre-computed metrics and return a JSON object with exactly these keys:
{{
  "insights": ["<3-5 concise bullet findings as plain English strings>"],
  "metrics": {metrics_json},
  "top_opportunity": "<one sentence>",
  "risk_flags": ["<0+ short strings for the Connector>"]
}}
The "metrics" value must echo the pre-computed JSON above exactly (same keys, numbers, and strings).
Return ONLY the JSON object. No prose, no markdown."""
