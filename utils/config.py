"""Central focus parameters for the 5-agent architecture."""

from __future__ import annotations

import pandas as pd

FOCUS_STATES: list[str] = ["SP", "RJ", "MG", "RS", "PR"]
FOCUS_CATEGORIES: list[str] = [
    "health_beauty",
    "bed_bath_table",
    "sports_leisure",
    "watches_gifts",
    "computers_accessories",
]
MIN_MONTHLY_ORDERS: int = 10
TRAINING_WINDOW_MONTHS: int = 12
MAX_ITERATIONS: int = 13


def get_top_states(n: int, data: dict) -> list[str]:
    """Return top N states by total order volume, ranked largest first."""
    if n <= 0:
        return []

    orders = data.get("orders", pd.DataFrame()).copy()
    if orders.empty:
        return []

    if "customer_state" not in orders.columns:
        customers = data.get("customers", pd.DataFrame())
        if not customers.empty and {"customer_id", "customer_state"}.issubset(
            customers.columns
        ):
            orders = orders.merge(
                customers[["customer_id", "customer_state"]],
                on="customer_id",
                how="left",
            )

    if "customer_state" not in orders.columns or "order_id" not in orders.columns:
        return []

    ranked = (
        orders.dropna(subset=["customer_state"])
        .groupby("customer_state", dropna=True)["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    return [str(state) for state in ranked.head(n).index.tolist()]


def get_top_categories(n: int, data: dict) -> list[str]:
    """Return top N categories by total revenue, ranked largest first."""
    if n <= 0:
        return []

    order_items = data.get("order_items", pd.DataFrame()).copy()
    if order_items.empty:
        return []

    if "product_category_name_english" not in order_items.columns:
        products = data.get("products", pd.DataFrame())
        categories = data.get("categories", pd.DataFrame())
        if {"product_id", "product_category_name"}.issubset(products.columns):
            order_items = order_items.merge(
                products[["product_id", "product_category_name"]],
                on="product_id",
                how="left",
            )
        if {
            "product_category_name",
            "product_category_name_english",
        }.issubset(categories.columns):
            order_items = order_items.merge(
                categories[
                    ["product_category_name", "product_category_name_english"]
                ],
                on="product_category_name",
                how="left",
            )

    if "price" not in order_items.columns:
        return []

    order_items["price"] = pd.to_numeric(order_items["price"], errors="coerce").fillna(0.0)
    ranked = (
        order_items.dropna(subset=["product_category_name_english"])
        .groupby("product_category_name_english", dropna=True)["price"]
        .sum()
        .sort_values(ascending=False)
    )
    return [str(category) for category in ranked.head(n).index.tolist()]
