"""Central focus parameters for the 5-agent architecture."""

from __future__ import annotations

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
