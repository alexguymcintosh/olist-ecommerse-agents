# Shared Memory Module — Implementation Stories

## Context
This story set is derived from:
- `CLAUDE.md`
- `docs/plans/5-agent-ceo-plan.md` section `utils/memory.py`

Scope is memory module planning only. No code is implemented by this document.

## Global Delivery Rules
- Python 3.12.
- SQLite table is single source of truth for cross-agent memory: `agent_memory`.
- Unit key is always `(state, category, month)` with a composite primary key.
- Writes must be non-destructive between agents (each agent updates only its own columns).
- Tests use `pytest` and filesystem-isolated databases via `tmp_path`.

## Story Dependency Order
1. Story 1 (schema)
2. Story 2 (Memory class + unit tests)
3. Story 3 (integration smoke)

---

## Story 1 — SQLite schema: create table and verify all 32 columns

### Goal
Define and auto-create the `agent_memory` SQLite table with the full agreed schema, then verify the table contains exactly 32 columns.

### Scope
- Implement schema creation in `utils/memory.py` using `CREATE TABLE IF NOT EXISTS agent_memory`.
- Include all required columns across PK + 5 agents.
- Add schema verification utility/assertion used by tests.

### Required Schema (32 Columns)
1. `state`
2. `category`
3. `month`
4. `geo_predicted_growth`
5. `geo_actual_growth`
6. `geo_directional_accuracy`
7. `geo_confidence`
8. `geo_confidence_score`
9. `geo_reasoning`
10. `sq_seller_count`
11. `sq_avg_review`
12. `sq_avg_delivery_days`
13. `sq_churn_risk`
14. `sq_top_seller_id`
15. `sq_reasoning`
16. `cr_avg_spend`
17. `cr_order_volume_trend`
18. `cr_top_payment_type`
19. `cr_high_value_customer_count`
20. `cr_repeat_rate`
21. `cr_reasoning`
22. `log_avg_delivery_days`
23. `log_pct_on_time`
24. `log_freight_ratio`
25. `log_fastest_seller_state`
26. `log_delivery_variance`
27. `log_reasoning`
28. `conn_decision`
29. `conn_confidence`
30. `conn_reasoning`
31. `conn_actual_outcome`
32. `conn_most_predictive_agent`

### Acceptance Criteria
1. Instantiating `Memory` auto-creates `agent_memory` when DB file is empty.
2. Table primary key is `(state, category, month)`.
3. `PRAGMA table_info(agent_memory)` returns exactly 32 columns in schema contract.
4. No extra or missing columns are present.

### Test Specification
- Add schema-focused tests in `tests/` for:
  - table auto-creation
  - PK correctness
  - exact 32-column existence and names

---

## Story 2 — Memory class: full implementation + 5 tests using `tmp_path`

### Goal
Implement the full `Memory` class interface in `utils/memory.py` for safe upserts, point reads, previous-month reads, and full-table DataFrame reads.

### Scope
- Implement:
  - `__init__(db_path: str | Path | None = None)`
  - `_create_table_if_not_exists()`
  - `write_row(state, category, month, **cols)`
  - `read_row(state, category, month)`
  - `read_prev_row(state, category, month)`
  - `read_all()`
- Preserve non-overwritten columns during upsert (read-merge-write pattern).
- Keep exception behavior explicit (do not swallow SQLite operational failures).

### Functional Requirements
- `write_row()` updates only provided columns while preserving existing values for other agents.
- `read_row()` returns `dict | None`.
- `read_prev_row()` computes previous month using pandas Period arithmetic and returns `None` when not found.
- `read_all()` returns DataFrame ordered by `state, category, month`.
- `DEFAULT_DB` path remains project-root aligned when `db_path` is omitted.

### Acceptance Criteria
1. All public methods above are implemented and callable.
2. Upserts are non-destructive across agent column subsets.
3. `read_prev_row()` is safe for first-iteration lookups (returns `None`).
4. `read_all()` includes all schema columns with `NaN` for unwritten fields.
5. Behavior is deterministic across repeated runs with fresh temp databases.

### Test Specification (exactly 5 tests; all use `tmp_path`)
1. `test_memory_auto_creates_table(tmp_path)`
2. `test_write_row_preserves_existing_columns(tmp_path)`
3. `test_read_row_returns_none_for_missing_key(tmp_path)`
4. `test_read_prev_row_returns_previous_month_or_none(tmp_path)`
5. `test_read_all_returns_dataframe_with_32_columns(tmp_path)`

---

## Story 3 — Integration smoke: write all 5-agent columns, `read_all` returns complete row

### Goal
Prove end-to-end memory interoperability by simulating writes from all 5 agents into one `(state, category, month)` key and validating complete-row retrieval.

### Scope
- Add one integration smoke test module for memory only.
- Sequentially call `write_row()` with each agent's column subset:
  - Geographic subset
  - Supply Quality subset
  - Customer Readiness subset
  - Logistics subset
  - Connector subset
- Validate combined row via `read_row()` and `read_all()`.

### Acceptance Criteria
1. After five partial writes, one row exists for the target key.
2. Row contains populated values from all five agent subsets.
3. No previously written agent fields are lost after later writes.
4. `read_all()` returns one complete row with all 32 columns available.
5. Smoke test runs against isolated temp DB and leaves no repository artifacts.

### Test Specification
- Add integration test:
  - `test_memory_smoke_all_agent_columns_roundtrip(tmp_path)`
    - Arrange one key (`state`, `category`, `month`)
    - Act with five `write_row()` calls (one per agent subset)
    - Assert full-row completeness from both `read_row()` and `read_all()`

---

## Definition of Done (Memory Module Stories)
- Story 1, Story 2, and Story 3 acceptance criteria all pass.
- Exactly five unit tests exist for Story 2, each using `tmp_path`.
- Integration smoke verifies full multi-agent write compatibility on shared row state.
- No agent implementation work is included; memory module only.
