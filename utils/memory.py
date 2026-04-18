"""Shared SQLite memory access layer for all agents."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Final

import pandas as pd


SCHEMA_COLUMNS: Final[tuple[str, ...]] = (
    "state",
    "category",
    "month",
    "geo_predicted_growth",
    "geo_actual_growth",
    "geo_directional_accuracy",
    "geo_confidence",
    "geo_confidence_score",
    "geo_reasoning",
    "sq_seller_count",
    "sq_avg_review",
    "sq_avg_delivery_days",
    "sq_churn_risk",
    "sq_top_seller_id",
    "sq_reasoning",
    "cr_avg_spend",
    "cr_order_volume_trend",
    "cr_top_payment_type",
    "cr_high_value_customer_count",
    "cr_repeat_rate",
    "cr_reasoning",
    "log_avg_delivery_days",
    "log_pct_on_time",
    "log_freight_ratio",
    "log_fastest_seller_state",
    "log_delivery_variance",
    "log_reasoning",
    "conn_decision",
    "conn_confidence",
    "conn_reasoning",
    "conn_actual_outcome",
    "conn_most_predictive_agent",
)

CREATE_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS agent_memory (
    state TEXT NOT NULL,
    category TEXT NOT NULL,
    month TEXT NOT NULL,
    geo_predicted_growth REAL,
    geo_actual_growth REAL,
    geo_directional_accuracy INTEGER,
    geo_confidence TEXT,
    geo_confidence_score REAL,
    geo_reasoning TEXT,
    sq_seller_count INTEGER,
    sq_avg_review REAL,
    sq_avg_delivery_days REAL,
    sq_churn_risk TEXT,
    sq_top_seller_id TEXT,
    sq_reasoning TEXT,
    cr_avg_spend REAL,
    cr_order_volume_trend REAL,
    cr_top_payment_type TEXT,
    cr_high_value_customer_count INTEGER,
    cr_repeat_rate REAL,
    cr_reasoning TEXT,
    log_avg_delivery_days REAL,
    log_pct_on_time REAL,
    log_freight_ratio REAL,
    log_fastest_seller_state TEXT,
    log_delivery_variance REAL,
    log_reasoning TEXT,
    conn_decision TEXT,
    conn_confidence TEXT,
    conn_reasoning TEXT,
    conn_actual_outcome TEXT,
    conn_most_predictive_agent TEXT,
    PRIMARY KEY (state, category, month)
);
"""


class Memory:
    """SQLite table access for shared agent memory."""

    DEFAULT_DB = Path(__file__).resolve().parents[1] / "memory.db"

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else self.DEFAULT_DB
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(CREATE_TABLE_SQL)

    def table_columns(self) -> tuple[str, ...]:
        """Return table columns in declared order."""
        with sqlite3.connect(self._path) as conn:
            pragma_rows = conn.execute("PRAGMA table_info(agent_memory)").fetchall()
        return tuple(row[1] for row in pragma_rows)

    def write_row(self, state: str, category: str, month: str, **cols: Any) -> None:
        """Upsert one state-category-month row while preserving other columns."""
        invalid_columns = set(cols) - set(SCHEMA_COLUMNS)
        if invalid_columns:
            invalid_str = ", ".join(sorted(invalid_columns))
            raise ValueError(f"Unknown memory columns: {invalid_str}")

        existing = self.read_row(state=state, category=category, month=month) or {}
        merged = {**existing, **cols, "state": state, "category": category, "month": month}
        columns = tuple(merged.keys())
        placeholders = ", ".join("?" for _ in columns)
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO agent_memory ({', '.join(columns)}) VALUES ({placeholders})",
                [merged[column] for column in columns],
            )

    def read_row(self, state: str, category: str, month: str) -> dict[str, Any] | None:
        """Read one memory row by composite primary key."""
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM agent_memory WHERE state=? AND category=? AND month=?",
                (state, category, month),
            ).fetchone()
        return dict(row) if row else None

    def read_prev_row(self, state: str, category: str, month: str) -> dict[str, Any] | None:
        """Read memory row from the immediately preceding month."""
        prev_month = str(pd.Period(month, freq="M") - 1)
        return self.read_row(state=state, category=category, month=prev_month)

    def read_all(self) -> pd.DataFrame:
        """Return all memory rows ordered by key for deterministic reads."""
        with sqlite3.connect(self._path) as conn:
            return pd.read_sql(
                "SELECT * FROM agent_memory ORDER BY state, category, month",
                conn,
            )
