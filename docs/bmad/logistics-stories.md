# Logistics Agent — Implementation Stories

## Context
This story set is derived from:
- `CLAUDE.md`
- `docs/plans/5-agent-ceo-plan.md` (section: `agents/logistics/logistics_agent.py (Agent 4)`)

Scope is logistics-only implementation planning. No code is implemented by this document.

## Global Delivery Rules
- Python 3.12.
- Pandas computes all logistics metrics before LLM usage.
- All LLM calls must route through `utils/openrouter_client.py`.
- `seller_state` is supply side and `customer_state` is demand side; never mix.
- Use `datetime.now(timezone.utc)` for timestamps.

## Story Dependency Order
1. Story 1 (`utils/schema_agents.py` LogisticsOutput contract tests)
2. Story 2 (`agents/logistics/logistics_agent.py` implementation + unit tests)
3. Story 3 (integration smoke with real data + mocked LLM)

---

## Story 1 — LogisticsOutput schema tests in `utils/schema_agents.py`

### Goal
Lock down the `LogisticsOutput` TypedDict contract with tests only, so downstream implementation has a stable output shape.

### Scope
- Write tests for `LogisticsOutput` TypedDict only.
- Do not modify logistics agent runtime logic in this story.
- Validate required keys and expected runtime value shapes used by Agent 4.

### Requirements
- Import target from `utils.schema_agents`.
- Verify a valid logistics payload includes all required fields:
  - `agent`, `timestamp`, `state`, `category`, `month`
  - `avg_delivery_days`, `pct_on_time`, `freight_ratio`
  - `fastest_seller_state`, `delivery_variance`, `cross_state_dependency`
  - `feasibility`, `reasoning`, `risk_flags`
- Ensure `risk_flags` is a list and `feasibility` is constrained to `STRONG|ADEQUATE|WEAK` by test fixtures/contracts.

### Acceptance Criteria
1. Test module exists and targets only `LogisticsOutput`.
2. Tests fail when a required `LogisticsOutput` key is omitted from a candidate payload.
3. Tests pass for a complete, valid logistics payload fixture.
4. Story introduces no implementation code in `agents/logistics/logistics_agent.py`.

### Test Specification
Create `tests/logistics/test_schema_logistics_output.py` with:
- `test_logistics_output_valid_payload_has_all_required_keys()`
- `test_logistics_output_missing_required_key_is_detected()`
- `test_logistics_output_feasibility_fixture_values_are_valid()`
- `test_logistics_output_risk_flags_is_list()`

---

## Story 2 — Implement `agents/logistics/logistics_agent.py` + 9 tests

### Goal
Implement Agent 4 end-to-end so it produces 25 logistics outputs (5 states × 5 categories) with NaN-safe metrics and LLM-narrated feasibility.

### Scope
- Implement `agents/logistics/logistics_agent.py` fully.
- Use joins and metrics exactly as specified in CEO plan.
- Persist logistics memory fields for each `(state, category, month)`.
- Add exactly 9 unit tests in `tests/logistics/test_logistics_agent.py`.

### Requirements
- Data join chain:
  - `orders -> order_items -> sellers -> customers -> products -> categories`
- Filter to delivered orders before time-based computations:
  - `order_status == "delivered"`
- Compute metrics with explicit guards:
  - `avg_delivery_days`: delivered only, scoped to customer demand side.
  - `pct_on_time`: delivered date <= estimated date, default `0.0` when no delivered rows.
  - `freight_ratio`: `mean(freight_value) / max(mean(price), 0.01)`.
  - `fastest_seller_state`: lowest mean delivery days route to customer state, default `""` when unavailable.
  - `delivery_variance`: `0.0` when fewer than 2 delivered observations.
  - `cross_state_dependency`: share where `seller_state != customer_state`.
- Generate one LLM call per state-category and parse:
  - `feasibility`, `reasoning`, `risk_flags`
- Fallback on LLM parse failure:
  - `risk_flags` includes `"agent_failed"` and output remains schema-valid.
- Write memory fields:
  - `log_avg_delivery_days`, `log_pct_on_time`, `log_freight_ratio`, `log_fastest_seller_state`, `log_reasoning`

### Acceptance Criteria
1. `run(month=...)` returns 25 logistics outputs for focus states/categories.
2. Every output has valid `feasibility` in `{STRONG, ADEQUATE, WEAK}`.
3. No output field is NaN; guard logic covers all sparse-data cases.
4. `delivery_variance` is `0.0` for `<2` delivered rows.
5. LLM parse failure path returns safe fallback with `risk_flags: ["agent_failed"]`.
6. Memory writes occur per state-category with logistics columns only.

### Test Specification (exactly 9 tests)
Create `tests/logistics/test_logistics_agent.py`:
- `test_run_returns_25_outputs_for_focus_grid()`
- `test_delivered_filter_applied_before_time_metrics()`
- `test_avg_delivery_days_scoped_to_customer_state_and_category()`
- `test_pct_on_time_defaults_zero_when_no_delivered_orders()`
- `test_freight_ratio_uses_price_floor_guard()`
- `test_fastest_seller_state_defaults_empty_when_no_delivered_rows()`
- `test_delivery_variance_is_zero_with_single_delivered_order()`
- `test_cross_state_dependency_uses_seller_vs_customer_state()`
- `test_llm_parse_failure_returns_agent_failed_risk_flag()`

---

## Story 3 — Integration smoke: real data + mocked LLM + no-NaN outputs

### Goal
Add a lightweight integration smoke test that runs LogisticsAgent on real project data with mocked LLM responses and validates shape/safety guarantees.

### Scope
- Integration test only for logistics path.
- Use real dataset loading path (`utils/data_loader.py`) and actual joins.
- Mock OpenRouter client response to deterministic valid JSON.

### Requirements
- Execute LogisticsAgent for one target month using real loaded data.
- Mock LLM output for each state-category request.
- Assert output count equals 25.
- Assert no NaN across all output fields in all 25 rows.
- Assert each output is contract-compatible with `LogisticsOutput`.

### Acceptance Criteria
1. Smoke test uses real data tables (not synthetic-only fixtures).
2. LLM dependency is fully mocked; test remains deterministic and offline.
3. Exactly 25 outputs are produced.
4. All outputs are NaN-free and feasibility-valid.
5. Test runtime remains smoke-level (fast enough for regular CI execution).

### Test Specification
Create `tests/integration/test_logistics_smoke.py`:
- `test_logistics_smoke_real_data_mocked_llm_returns_25_outputs_without_nan()`
