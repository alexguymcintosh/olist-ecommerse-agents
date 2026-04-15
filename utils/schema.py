from __future__ import annotations

from typing import TypedDict


class CustomerMetrics(TypedDict):
    repeat_rate: float
    avg_review_score: float
    avg_delivery_days: float
    pct_late_delivery: float
    top_payment_type: str


class ProductMetrics(TypedDict):
    top_category: str
    top_category_revenue: float
    total_categories: int
    avg_order_value: float


class SellerMetrics(TypedDict):
    total_sellers: int
    pct_sao_paulo: float
    # Pareto definition: revenue share held by the top 20% of sellers.
    top_seller_revenue_concentration: float


class DomainAgentOutput(TypedDict):
    agent: str
    timestamp: str
    insights: list[str]
    metrics: dict
    top_opportunity: str
    risk_flags: list[str]


class PriorityAction(TypedDict):
    action: str
    agent: str
    urgency: str


class ConnectorOutput(TypedDict):
    timestamp: str
    cross_domain_insights: list[str]
    strategic_recommendation: str
    priority_actions: list[PriorityAction]
    briefing: str
