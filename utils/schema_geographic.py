from __future__ import annotations

from typing import TypedDict


class GeographicMetrics(TypedDict):
    top5_states: list[str]
    top5_categories: list[str]
    growth_matrix: dict[str, dict[str, float]]
    momentum_scores: dict[str, dict[str, float]]
    seller_counts: dict[str, dict[str, int]]
    order_counts: dict[str, dict[str, int]]
    supply_gap_matrix: dict[str, dict[str, float | None]]


class Prediction(TypedDict):
    state: str
    category: str
    predicted_growth_pct: float
    confidence: str
    confidence_score: float
    reasoning: str


class SupplyGap(TypedDict):
    state: str
    category: str
    current_sellers: int
    current_month_order_count: int
    predicted_order_volume: float | None
    supply_gap_ratio: float | None
    supply_gap_severity: float


class RankedOpportunity(TypedDict):
    rank: int
    state: str
    category: str
    predicted_growth_pct: float
    current_sellers: int
    current_month_order_count: int
    predicted_order_volume: float | None
    supply_gap_ratio: float | None
    supply_gap_severity: float
    composite_score: float
    recommended_action: str
    urgency: str


class GeographicOutput(TypedDict):
    agent: str
    timestamp: str
    iteration: int
    training_window: tuple[str, str]
    prediction_month: str
    predictions: list[Prediction]
    supply_gaps: list[SupplyGap]
    ranked_opportunities: list[RankedOpportunity]
    metrics: GeographicMetrics
    risk_flags: list[str]


class WalkForwardIteration(TypedDict):
    iteration: int
    training_window: tuple[str, str]
    prediction_month: str
    previous_prediction: list[Prediction] | None
    predictions: list[Prediction]
    supply_gaps: list[SupplyGap]
    ranked_opportunities: list[RankedOpportunity]
    validation: dict


class WalkForwardResult(TypedDict):
    completed_iterations: int
    total_iterations: int
    iterations: list[WalkForwardIteration]
    aggregate_accuracy: dict
