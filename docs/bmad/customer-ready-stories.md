# Customer Readiness Agent — Implementation Stories

## Context
This story set is derived from:
- `CLAUDE.md`
- `docs/plans/5-agent-ceo-plan.md` (Agent 3 section)

Scope is planning only. Do not implement code in this document.

## Global Rules
- Python 3.12.
- Pandas computes all metrics; LLM only narrates pre-computed facts.
- All LLM calls go through `utils/openrouter_client.py`.
- Use `datetime.now(timezone.utc)` (never `utcnow()`).
- Never hardcode focus states or categories; read from config.
- Keep tests isolated to the correct test folders.

## Story Dependency Order
1. Story 1 (`CustomerReadinessOutput` TypedDict tests)
2. Story 2 (`CustomerReadinessAgent` full implementation + unit tests)
3. Story 3 (integration smoke with real data + mocked LLM)

---

## Story 1 — `utils/schema_agents.py` tests for `CustomerReadinessOutput` TypedDict only

### Goal
Define the contract quality gate for `CustomerReadinessOutput` by adding tests first, scoped only to this TypedDict.

### Scope
- Add test module: `tests/customer_ready/test_schema_customer_readiness.py`.
- Validate only the `CustomerReadinessOutput` shape and field-level expectations.
- Do not add tests for SupplyQuality, Logistics, or Connector schemas in this story.

### Requirements
- `CustomerReadinessOutput` must include:
  - `agent`, `timestamp`, `state`, `category`, `month`
  - `avg_spend`, `order_volume_trend`, `top_payment_type`
  - `high_value_customer_count`, `repeat_rate`, `installment_pct`
  - `readiness`, `reasoning`, `risk_flags`
- `readiness` must be constrained to `HIGH | MEDIUM | LOW`.
- Numeric fields must be typed consistently for downstream connector use.
- Schema tests should fail clearly if field names drift from Agent 3 plan.

### Acceptance Criteria
1. Tests exist and are limited to `CustomerReadinessOutput`.
2. Tests enforce presence of all required fields listed above.
3. Tests enforce readiness enum values (`HIGH`, `MEDIUM`, `LOW`).
4. No non-Agent-3 schema coverage is added in this story.

### Test Specification
- `test_customer_readiness_output_required_keys()`
- `test_customer_readiness_output_readiness_enum()`
- `test_customer_readiness_output_numeric_fields_are_declared()`

---

## Story 2 — `agents/customer_ready/customer_ready_agent.py` full implementation + 9 tests

### Goal
Implement the Customer Readiness Agent end-to-end for all 25 state-category pairs with robust metric guards and fallback behavior.

### Scope
- Fully implement `agents/customer_ready/customer_ready_agent.py`.
- Use `utils/schema_agents.py` `CustomerReadinessOutput`.
- Read/write memory columns for Agent 3 only.
- Add/complete exactly 9 unit tests in `tests/customer_ready/test_customer_ready_agent.py`.

### Implementation Requirements
- Build join pipeline per plan:
  - `orders -> customers -> order_items -> products -> categories -> payments_agg`
- Aggregate payments to order level before joining:
  - `payments_agg = payments.groupby("order_id")["payment_value"].sum()...`
- Compute `installment_pct` from raw `payments` in a separate pass, credit-card only.
- Collapse to `per_order` before money metrics to prevent duplicated item rows from inflating spend.
- Compute metrics before LLM:
  - `avg_spend` from `per_order["total_payment_value"].mean()`
  - `order_volume_trend` from monthly distinct `order_id` counts
  - `top_payment_type` mode by state (default `credit_card` when empty)
  - `high_value_customer_count` from per-customer avg order value above dataset median
  - `repeat_rate` from customers with >1 distinct order_id in scoped pair
  - `installment_pct` from credit-card installments `>= 6`
- Enforce "last 3 months" temporal window from training max month.
- LLM returns JSON with `readiness`, `reasoning`, `risk_flags`; parse safely.
- Fallback on parse/model failure returns valid output with `risk_flags: ["agent_failed"]`.
- Always return exactly 25 outputs (`len(FOCUS_STATES) * len(FOCUS_CATEGORIES)`).
- Write memory fields:
  - `cr_avg_spend`, `cr_order_volume_trend`, `cr_top_payment_type`
  - `cr_high_value_customer_count`, `cr_repeat_rate`, `cr_reasoning`

### Acceptance Criteria
1. Agent returns 25 outputs for one run month.
2. Every output has readiness in `{HIGH, MEDIUM, LOW}`.
3. `repeat_rate == 0.0` is treated as valid signal, not error.
4. `installment_pct` excludes non-credit-card payments.
5. No payment double-counting from installment rows.
6. LLM parse failure never raises; fallback output is emitted.
7. Memory writes target only Agent 3 columns.
8. No hardcoded state/category names in logic.
9. All 9 story tests pass.

### Test Specification (exactly 9)
1. `test_run_returns_25_outputs_for_focus_pairs()`
2. `test_readiness_values_constrained_to_high_medium_low()`
3. `test_payments_are_aggregated_per_order_before_join()`
4. `test_money_metrics_use_per_order_not_order_item_rows()`
5. `test_installment_pct_uses_credit_card_only()`
6. `test_order_volume_trend_uses_distinct_order_count()`
7. `test_repeat_rate_zero_is_valid_output()`
8. `test_llm_parse_failure_returns_agent_failed_flag()`
9. `test_memory_write_maps_to_customer_readiness_columns()`

---

## Story 3 — Integration smoke (real data, mocked LLM, assert 25 outputs and no NaN)

### Goal
Provide a lightweight integration confidence check that the Customer Readiness Agent runs against real project data with deterministic mocked LLM responses.

### Scope
- Add integration smoke test file:
  - `tests/integration/test_customer_ready_smoke.py`
- Load real dataset via project data loader path.
- Mock `utils/openrouter_client.py` responses for deterministic JSON.
- Run one month window and assert output quality.

### Requirements
- Use real Olist tables for joins and metric computation.
- Mock LLM response with valid `readiness`, `reasoning`, and `risk_flags`.
- Execute full run path of `CustomerReadinessAgent` (not isolated private methods).
- Assert exactly 25 outputs are returned.
- Assert no NaN in required numeric fields:
  - `avg_spend`, `order_volume_trend`, `high_value_customer_count`, `repeat_rate`, `installment_pct`
- Assert no NaN/empty contract breaks for key categorical fields:
  - `state`, `category`, `month`, `readiness`

### Acceptance Criteria
1. Smoke test executes with real data and mocked LLM only.
2. Exactly 25 outputs are produced.
3. Required numeric fields contain no NaN in all outputs.
4. Readiness values remain within `{HIGH, MEDIUM, LOW}`.
5. Test remains deterministic across repeated runs.

### Test Specification
- `test_customer_ready_smoke_real_data_mocked_llm_25_outputs_no_nan()`

---

## Definition of Done (These 3 Stories)
- Story 1 schema tests pass and remain scoped to `CustomerReadinessOutput` only.
- Story 2 agent implementation is complete and all 9 specified tests pass.
- Story 3 integration smoke passes with real data, mocked LLM, 25 outputs, and no NaN.
- No connector/supply/logistics implementation is included in this story set.
