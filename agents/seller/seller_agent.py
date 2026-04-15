"""Seller domain agent: supply base, geography, revenue concentration."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from utils.base_agent import BaseAgent


class SellerAgent(BaseAgent):
    """Analyses sellers, order items, and reviews (seller-side context)."""

    AGENT_NAME = "seller"

    def _prepare_data(self) -> pd.DataFrame:
        order_items = self.data["order_items"]
        sellers = self.data["sellers"]
        reviews = self.data["reviews"]
        return order_items.merge(sellers, on="seller_id").merge(
            reviews, on="order_id", how="left"
        )

    def _compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        # Reviews join can duplicate order lines; metrics use one row per order line.
        if df.empty:
            return {
                "total_sellers": 0,
                "pct_sao_paulo": 0.0,
                "top_seller_revenue_concentration": 0.0,
            }

        line_df = df.drop_duplicates(subset=["order_id", "order_item_id"])
        price = pd.to_numeric(line_df["price"], errors="coerce").fillna(0.0)
        line_df = line_df.assign(_price=price)

        total_sellers = int(line_df["seller_id"].nunique())

        sellers_unique = line_df.drop_duplicates(subset=["seller_id"])
        state = sellers_unique["seller_state"].astype(str).str.upper()
        if len(sellers_unique) == 0:
            pct_sao_paulo = 0.0
        else:
            pct_sao_paulo = float((state == "SP").sum() / len(sellers_unique))

        by_seller = line_df.groupby("seller_id", sort=False)["_price"].sum()
        total_rev = float(by_seller.sum())
        n = len(by_seller)
        if total_rev <= 0.0 or n == 0:
            top_seller_revenue_concentration = 0.0
        else:
            k = max(1, math.ceil(0.2 * n))
            top_rev = float(by_seller.sort_values(ascending=False).head(k).sum())
            top_seller_revenue_concentration = top_rev / total_rev

        return {
            "total_sellers": total_sellers,
            "pct_sao_paulo": pct_sao_paulo,
            "top_seller_revenue_concentration": float(top_seller_revenue_concentration),
        }

    def _build_question(self, metrics: dict[str, Any], sample_str: str) -> str:
        metrics_json = json.dumps(metrics, indent=2)
        return f"""Pre-computed metrics (authoritative — use only these numbers, do not infer from the sample):
{metrics_json}

Below is a sample of up to 50 rows for context (may be incomplete — do not compute metrics from it).

SAMPLE:
{sample_str}

delivery time analysis belongs to CustomerAgent, do not analyse it here.

Analyse the seller / supply domain using the pre-computed metrics and return a JSON object with exactly these keys:
{{
  "insights": ["<3-5 concise bullet findings as plain English strings>"],
  "metrics": {metrics_json},
  "top_opportunity": "<one sentence>",
  "risk_flags": ["<0+ short strings for the Connector>"]
}}
The "metrics" value must echo the pre-computed JSON above exactly (same keys, numbers, and strings).
Return ONLY the JSON object. No prose, no markdown."""
