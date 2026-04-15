"""Product domain agent: catalog, categories, order-item revenue."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from utils.base_agent import BaseAgent


class ProductAgent(BaseAgent):
    """Analyses order items joined to products and category translations."""

    AGENT_NAME = "product"

    def _prepare_data(self) -> pd.DataFrame:
        order_items = self.data["order_items"]
        products = self.data["products"]
        categories = self.data["categories"]
        df = order_items.merge(products, on="product_id").merge(
            categories, on="product_category_name", how="left"
        )
        df = df.copy()
        df["product_category_name_english"] = df["product_category_name_english"].fillna(
            df["product_category_name"]
        )
        return df

    def _compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {
                "top_category": "",
                "top_category_revenue": 0.0,
                "total_categories": 0,
                "avg_order_value": 0.0,
            }

        cat_col = "product_category_name_english"
        work = df.assign(
            _price=pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        )
        by_cat = work.groupby(cat_col, dropna=False)["_price"].sum()
        by_cat = by_cat[by_cat.index.notna()]
        if by_cat.empty:
            top_category = ""
            top_category_revenue = 0.0
        else:
            top_category = str(by_cat.idxmax())
            top_category_revenue = float(by_cat.max())

        total_categories = int(work[cat_col].nunique(dropna=True))

        total_revenue = float(work["_price"].sum())
        n_orders = int(work["order_id"].nunique())
        avg_order_value = total_revenue / n_orders if n_orders else 0.0

        return {
            "top_category": top_category,
            "top_category_revenue": top_category_revenue,
            "total_categories": total_categories,
            "avg_order_value": float(avg_order_value),
        }

    def _build_question(self, metrics: dict[str, Any], sample_str: str) -> str:
        metrics_json = json.dumps(metrics, indent=2)
        return f"""Pre-computed metrics (authoritative — use only these numbers, do not infer from the sample):
{metrics_json}

Below is a sample of up to 50 rows for context (may be incomplete — do not compute metrics from it).

SAMPLE:
{sample_str}

Analyse the product / catalog domain using the pre-computed metrics and return a JSON object with exactly these keys:
{{
  "insights": ["<3-5 concise bullet findings as plain English strings>"],
  "metrics": {metrics_json},
  "top_opportunity": "<one sentence>",
  "risk_flags": ["<0+ short strings for the Connector>"]
}}
The "metrics" value must echo the pre-computed JSON above exactly (same keys, numbers, and strings).
Return ONLY the JSON object. No prose, no markdown."""
