from __future__ import annotations

from typing import TypedDict


class SupplyQualityOutput(TypedDict):
    agent: str
    timestamp: str
    state: str
    category: str
    month: str
    seller_count: int
    avg_review_score: float
    avg_delivery_days: float
    churn_risk: str
    churn_rate: float
    top_seller_id: str
    seller_concentration: float
    supply_confidence: str
    reasoning: str
    risk_flags: list[str]


class CustomerReadinessOutput(TypedDict):
    agent: str
    timestamp: str
    state: str
    category: str
    month: str
    avg_spend: float
    order_volume_trend: float
    top_payment_type: str
    high_value_customer_count: int
    repeat_rate: float
    installment_pct: float
    readiness: str
    reasoning: str
    risk_flags: list[str]


class LogisticsOutput(TypedDict):
    agent: str
    timestamp: str
    state: str
    category: str
    month: str
    avg_delivery_days: float
    pct_on_time: float
    freight_ratio: float
    fastest_seller_state: str
    delivery_variance: float
    cross_state_dependency: float
    feasibility: str
    reasoning: str
    risk_flags: list[str]


class ConnectorDecision(TypedDict):
    state: str
    category: str
    month: str
    composite_score: float
    decision: str
    confidence: str
    urgency: str
    reasoning: str
    challenge: str
    most_predictive_agent: str
    risk_flags: list[str]


class ConnectorOutput(TypedDict):
    agent: str
    timestamp: str
    month: str
    decisions: list[ConnectorDecision]
    briefing: str
    follow_up_used: bool
    follow_up_agent: str | None
    follow_up_question: str | None
    follow_up_response: str | None
