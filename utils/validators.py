"""Runtime validation for domain agent outputs."""

from __future__ import annotations

REQUIRED_KEYS = {"agent", "timestamp", "insights", "metrics", "top_opportunity", "risk_flags"}

REQUIRED_METRICS = {
    "customer": {"repeat_rate", "avg_review_score", "avg_delivery_days", "pct_late_delivery", "top_payment_type"},
    "product": {"top_category", "top_category_revenue", "total_categories", "avg_order_value"},
    "seller": {"total_sellers", "pct_sao_paulo", "top_seller_revenue_concentration"},
}


def validate_output(output: dict, agent_name: str) -> None:
    """Raises ValueError with a clear message if output is malformed."""
    missing = REQUIRED_KEYS - set(output.keys())
    if missing:
        raise ValueError(
            f"ValidationError: {agent_name}.run() missing required keys: {missing}\n"
            f"Expected: {REQUIRED_KEYS}\nGot: {set(output.keys())}"
        )
    agent_type = output.get("agent")
    if agent_type in REQUIRED_METRICS:
        metrics_missing = REQUIRED_METRICS[agent_type] - set(output["metrics"].keys())
        if metrics_missing:
            raise ValueError(
                f"ValidationError: {agent_name} metrics missing keys: {metrics_missing}"
            )
