# Supply Quality Agent — Implementation Stories

## Context
This story set is derived from:
- `CLAUDE.md`
- `docs/plans/5-agent-ceo-plan.md` section `4. agents/supply_quality/supply_quality_agent.py (Agent 2)`

Scope is planning only. No code is implemented by this document.

## Global Rules
- Python 3.12.
- Follow focus parameters from `utils/config.py`; never hardcode state/category lists in agent logic.
- Pandas computes all metrics; LLM only narrates pre-computed facts.
- All LLM calls route through `utils/openrouter_client.py`.
- Use `datetime.now(timezone.utc)` for timestamps.
- Keep seller-side logic on `seller_state` (never `customer_state`).

## Story Dependency Order
1. Story 1 (`utils/schema_agents.py` test contract lock for `SupplyQualityOutput`)
2. Story 2 (`agents/supply_quality/supply_quality_agent.py` implementation + 9 tests)
3. Story 3 (integration smoke: real data, mocked LLM, 25 outputs, no NaN)

---

## Story 1 — `utils/schema_agents.py` contract tests (`SupplyQualityOutput` only)

### Goal
Lock the schema contract for `SupplyQualityOutput` with tests only, so Agent 2 implementation can proceed against a stable output shape.

### Scope
- Add tests that validate `SupplyQualityOutput` keys and expected value types.
- Test only `SupplyQualityOutput` from `utils/schema_agents.py`.
- Do not add tests for `CustomerReadinessOutput`, `LogisticsOutput`, or connector schemas in this story.
- No implementation changes to agent logic in this story.

### Requirements
- Import path under test: `from utils.schema_agents import SupplyQualityOutput`.
- Validate required fields:
  - `agent`, `timestamp`, `state`, `category`, `month`
  - `seller_count`, `avg_review_score`, `avg_delivery_days`
  - `churn_risk`, `churn_rate`, `top_seller_id`, `seller_concentration`
  - `supply_confidence`, `reasoning`, `risk_flags`
- Validate canonical enum domains in test fixtures:
  - `churn_risk`: `HIGH|MEDIUM|LOW`
  - `supply_confidence`: `STRONG|ADEQUATE|WEAK`
- Confirm type hints are importable and stable (TypedDict introspection is acceptable).

### Acceptance Criteria
1. Tests cover `SupplyQualityOutput` only and pass.
2. Tests fail if required `SupplyQualityOutput` fields are removed or renamed.
3. No runtime dependency imports are introduced by this story.

### Test Specification
Create `tests/supply_quality/test_schema_supply_quality_output.py`:
- `test_supply_quality_output_required_keys()`
- `test_supply_quality_output_enum_domains_in_fixture()`
- `test_supply_quality_output_type_hints_are_exposed()`

---

## Story 2 — `agents/supply_quality/supply_quality_agent.py` full implementation + 9 tests

### Goal
Implement Agent 2 end-to-end so it computes seller health metrics with pandas, calls the LLM for confidence narration, and returns one `SupplyQualityOutput` per state-category pair.

### Scope
- Implement `SupplyQualityAgent` in `agents/supply_quality/supply_quality_agent.py`.
- Implement `run(...) -> list[SupplyQualityOutput]` with exactly `len(states) * len(categories)` outputs.
- Use pre-joined training DataFrame input and compute all required metrics before LLM call.
- Enforce sparse-data sentinel path and hard risk flags.
- Include exactly 9 unit tests for this story.

### Requirements
- Metrics and behavior align with CEO plan:
  - `seller_count`
  - `avg_review_score` (last 3 months weighted; sentinel `3.0` when no reviews)
  - `avg_delivery_days` (delivered orders only)
  - `churn_rate` and `churn_risk` thresholds (`>0.3` HIGH, `0.1-0.3` MEDIUM, `<0.1` LOW)
  - `top_seller_id`
  - `seller_concentration` with safe divide guard
- LLM prompt uses pre-computed metrics only; parse JSON into:
  - `supply_confidence`, `reasoning`, `risk_flags`
- Hard flags:
  - always add `critical_seller_gap` when `seller_count == 0`
  - always add `high_seller_churn` when `churn_rate > 0.5`
- Sparse handling:
  - if latest-month pair volume `< MIN_MONTHLY_ORDERS`, skip LLM and return sentinel with `risk_flags: ["sparse_data"]`
- Failure handling:
  - malformed/empty LLM output must return fallback with `risk_flags: ["agent_failed"]` without raising
- No memory writes inside the agent (handled externally by orchestrator).

### Acceptance Criteria
1. Returns exactly 25 outputs for 5 states x 5 categories.
2. Every output conforms to `SupplyQualityOutput`.
3. Sparse pairs produce sentinel output and skip LLM call.
4. `seller_count == 0` always includes `critical_seller_gap`.
5. LLM parse failure returns fallback output with `agent_failed` risk flag.
6. Exactly 9 unit tests are present and passing for this story.

### Test Specification (exactly 9 tests)
Create `tests/supply_quality/test_supply_quality_agent.py`:
- `test_run_returns_25_outputs_for_focus_grid()`
- `test_output_matches_supply_quality_output_contract()`
- `test_avg_review_score_uses_sentinel_when_no_reviews()`
- `test_churn_risk_threshold_mapping()`
- `test_seller_count_zero_adds_critical_seller_gap()`
- `test_churn_rate_above_half_adds_high_seller_churn()`
- `test_sparse_pair_returns_sparse_data_and_skips_llm()`
- `test_llm_parse_failure_returns_agent_failed_fallback()`
- `test_top_seller_and_concentration_safe_guard()`

---

## Story 3 — Integration smoke (real data, mocked LLM, assert 25 outputs no NaN)

### Goal
Validate a thin end-to-end smoke path for Agent 2 using real project data inputs and a mocked LLM response, proving deterministic execution and clean numeric outputs.

### Scope
- Add one integration smoke test under `tests/integration/`.
- Use real loaded dataset tables/path used by project loaders.
- Mock LLM client response to avoid network/model variability.
- Execute Agent 2 across focus states/categories for one prediction month.

### Requirements
- Test uses real data ingestion path (not synthetic-only fixture).
- LLM response is mocked to a valid JSON object for `supply_confidence`, `reasoning`, and `risk_flags`.
- Assert output count is exactly `25`.
- Assert no NaN values in numeric output fields expected to be numeric in each returned object:
  - `seller_count`
  - `avg_review_score`
  - `avg_delivery_days`
  - `churn_rate`
  - `seller_concentration`
- Assert test does not perform external LLM API calls.

### Acceptance Criteria
1. Integration smoke runs with real data and mocked LLM.
2. Agent returns exactly 25 outputs.
3. Numeric fields in outputs contain no NaN values.
4. Test passes reliably across repeated runs.

### Test Specification
Create `tests/integration/test_supply_quality_smoke.py`:
- `test_supply_quality_smoke_real_data_mocked_llm_25_outputs_no_nan()`
