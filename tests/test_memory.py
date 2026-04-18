"""Tests for shared SQLite memory table."""

import sqlite3

from utils.memory import Memory, SCHEMA_COLUMNS


def test_memory_auto_creates_table(tmp_path) -> None:
    db_path = tmp_path / "memory_story2.db"
    memory = Memory(db_path)

    assert db_path.exists()
    assert memory.table_columns() == SCHEMA_COLUMNS

    with sqlite3.connect(db_path) as conn:
        table_info = conn.execute("PRAGMA table_info(agent_memory)").fetchall()
    pk_columns = [row[1] for row in table_info if row[5] > 0]
    assert pk_columns == ["state", "category", "month"]


def test_write_row_preserves_existing_columns(tmp_path) -> None:
    memory = Memory(tmp_path / "memory_story2.db")
    memory.write_row("SP", "health_beauty", "2018-01", geo_predicted_growth=0.11, geo_reasoning="r1")
    memory.write_row("SP", "health_beauty", "2018-01", sq_seller_count=8, sq_reasoning="r2")

    row = memory.read_row("SP", "health_beauty", "2018-01")
    assert row is not None
    assert row["geo_predicted_growth"] == 0.11
    assert row["geo_reasoning"] == "r1"
    assert row["sq_seller_count"] == 8
    assert row["sq_reasoning"] == "r2"


def test_read_row_returns_none_for_missing_key(tmp_path) -> None:
    memory = Memory(tmp_path / "memory_story2.db")
    assert memory.read_row("RJ", "computers_accessories", "2018-05") is None


def test_read_prev_row_returns_previous_month_or_none(tmp_path) -> None:
    memory = Memory(tmp_path / "memory_story2.db")
    memory.write_row("MG", "sports_leisure", "2018-03", conn_decision="hold")

    prev_row = memory.read_prev_row("MG", "sports_leisure", "2018-04")
    assert prev_row is not None
    assert prev_row["conn_decision"] == "hold"
    assert memory.read_prev_row("MG", "sports_leisure", "2018-03") is None


def test_read_all_returns_dataframe_with_32_columns(tmp_path) -> None:
    memory = Memory(tmp_path / "memory_story2.db")
    memory.write_row("RS", "bed_bath_table", "2018-02", cr_avg_spend=154.2)
    frame = memory.read_all()

    assert list(frame.columns) == list(SCHEMA_COLUMNS)
    assert frame.shape == (1, 32)
    assert frame.iloc[0]["cr_avg_spend"] == 154.2
