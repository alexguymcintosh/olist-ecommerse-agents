# Connector Agent - Implementation Stories

## Context
This story set is derived from:
- `CLAUDE.md`
- `docs/plans/5-agent-ceo-plan.md` (Agent 5 connector section)

Scope is planning only. Do not implement code in this document.

## Global Hard Requirements (apply to all stories)
- Connector does **not** subclass any base class.
- Composite score formula is exactly:
  - `(geo * 0.35 + supply * 0.25 + customer * 0.20 + logistics * 0.20) * geo_predicted_growth_pct`
- If `composite_score <= 0`, force `decision="no_action"` and `urgency="LOW"`.
- `prev_month=None` on first iteration must run without error.
- Follow-up is triggered only when top composite score is strictly greater than 2x runner-up.
- If all 4 agent signals failed, connector still returns 25 decisions.
- Memory write is called exactly 25 times per connector run (once per pair).

## Story Dependency Order
1. Story 1 (`utils/schema_agents.py` TypedDict test coverage)
2. Story 2 (`agents/connector/connector_agent.py` implementation + 9 tests)
3. Story 3 (integration smoke for connector orchestration behavior)

---

## Story 1 - `utils/schema_agents.py`: tests for `ConnectorDecision` and `ConnectorOutput` TypedDicts

### Goal
Create schema contract tests that lock the connector output shape before connector implementation.

### Scope
- Add test module: `tests/connector/test_schema_connector.py`.
- Validate only `ConnectorDecision` and `ConnectorOutput` TypedDicts.
- No connector runtime logic in this story.

### Requirements
- `ConnectorDecision` must include:
  - `state`, `category`, `month`, `composite_score`
  - `decision`, `confidence`, `urgency`
  - `reasoning`, `challenge`, `most_predictive_agent`, `risk_flags`
- `ConnectorOutput` must include:
  - `agent`, `timestamp`, `month`, `decisions`, `briefing`
  - `follow_up_used`, `follow_up_agent`, `follow_up_question`, `follow_up_response`
- `decisions` is typed as `list[ConnectorDecision]`.
- Follow-up fields are nullable (`str | None`) where specified.

### Acceptance Criteria
1. Tests enforce presence of all required keys in `ConnectorDecision`.
2. Tests enforce presence of all required keys in `ConnectorOutput`.
3. Tests enforce that `decisions` is declared as `list[ConnectorDecision]`.
4. Tests are isolated to connector schema definitions only.

### Test Specification
- `test_connector_decision_required_keys()`
- `test_connector_output_required_keys()`
- `test_connector_output_decisions_typed_as_list_of_connector_decision()`
- `test_connector_output_follow_up_fields_allow_none()`

---

## Story 2 - `agents/connector/connector_agent.py`: full implementation + 9 tests from plan

### Goal
Implement connector decisioning for all 25 focus pairs with deterministic scoring, robust fallbacks, optional follow-up, and memory persistence.

### Scope
- Fully implement `agents/connector/connector_agent.py`.
- Use `GeographicOutput` from `utils/schema_geographic.py`.
- Use `SupplyQualityOutput`, `CustomerReadinessOutput`, `LogisticsOutput`, `ConnectorOutput` from `utils/schema_agents.py`.
- Add/complete 9 tests in `tests/connector/test_connector_agent.py` exactly as defined in plan.

### Implementation Requirements
- Class remains standalone (`ConnectorAgent`) and does not inherit from any base class.
- `run(...)` accepts `prev_month: str | None = None` and handles first iteration safely.
- Build one decision per state x category pair (25 total).
- Composite calculation uses exactly:
  - `geo_signal = geo["confidence_score"] * 0.35`
  - `supply_signal = SUPPLY_SCORE[supply_confidence] * 0.25`
  - `customer_signal = READINESS_SCORE[readiness] * 0.20`
  - `logistics_signal = LOGISTICS_SCORE[feasibility] * 0.20`
  - `composite = (geo_signal + supply_signal + customer_signal + logistics_signal) * geo["predicted_growth_pct"]`
- Apply decision override:
  - If `composite_score <= 0`: `decision="no_action"`, `urgency="LOW"`.
- Parse LLM JSON safely per pair; on parse failure:
  - keep computed `composite_score`
  - add connector fallback details with `risk_flags` including `"connector_failed"`.
- If input agent outputs include `risk_flags` containing `"agent_failed"`, propagate risk context but still produce a decision.
- Sort final `decisions` descending by `composite_score`.
- Follow-up behavior:
  - evaluate only after 25 decisions are ranked
  - trigger only when top score `> 2 * max(runner_up, 0.001)`
  - if fewer than 2 decisions, no follow-up
  - execute at most one follow-up for top-ranked pair.
- Memory persistence:
  - call `memory.write_row(...)` once per decision (25 calls/run)
  - write connector columns only (`conn_decision`, `conn_confidence`, `conn_reasoning`, `conn_most_predictive_agent`).

### Acceptance Criteria
1. Connector returns exactly 25 `ConnectorDecision` items.
2. Decisions are sorted in descending `composite_score`.
3. `prev_month=None` path executes without exception.
4. All-agents-failed input still returns 25 decisions.
5. Composite formula matches required weights exactly.
6. Follow-up triggers only when top score is strictly > 2x runner-up.
7. Memory write called exactly 25 times in one run.
8. Parse failure preserves `composite_score` and emits fallback risk flag.
9. `composite_score <= 0` always maps to `no_action` + `LOW` urgency.

### Test Specification (exactly 9 from plan)
1. `test_run_returns_25_decisions()`
2. `test_decisions_sorted_by_composite_score()`
3. `test_prev_month_none_first_iteration()`
4. `test_all_agents_failed_still_runs()`
5. `test_composite_score_weights()`
6. `test_follow_up_triggered_when_top_dominates()`
7. `test_follow_up_not_triggered_when_close()`
8. `test_memory_write_called_per_decision()`
9. `test_llm_parse_failure_preserves_composite()`

---

## Story 3 - Integration smoke: mocked 4-agent outputs, sorted 25 decisions, follow-up behavior

### Goal
Provide a deterministic connector integration smoke test covering end-to-end orchestration with mocked upstream agent outputs.

### Scope
- Add integration test file: `tests/integration/test_connector_smoke.py`.
- Mock all 4 domain agent outputs (geographic, supply, customer, logistics).
- Mock connector LLM responses for deterministic decision JSON and follow-up text.
- Execute connector `run(...)` with realistic 5x5 grid inputs.

### Requirements
- Input setup includes full 25 state x category combinations.
- Output asserts:
  - exactly 25 decisions are returned
  - decisions are sorted by `composite_score` descending
- Follow-up coverage includes both branches:
  - triggered when top score dominates (`> 2x` runner-up)
  - not triggered when scores are close
- First-iteration safety:
  - call `run(..., prev_month=None)` with no crash
- Failure-tolerance coverage:
  - when all 4 mocked agent outputs are failed, connector still emits 25 decisions
- Persistence coverage:
  - memory write invocation count equals 25 in a single run.

### Acceptance Criteria
1. Smoke test uses mocked 4-agent outputs and deterministic mocked LLM.
2. Exactly 25 connector decisions are always produced.
3. Composite ordering is validated end-to-end (descending).
4. Follow-up logic is validated for both trigger and no-trigger scenarios.
5. `prev_month=None` scenario is explicitly tested.
6. All-agent-failure scenario still yields 25 decisions.
7. Memory writes are asserted at exactly 25 calls per run.

### Test Specification
- `test_connector_smoke_25_decisions_sorted_with_follow_up_paths()`

---

## Definition of Done (All 3 Stories)
- Story 1 locks connector TypedDict contracts with focused schema tests.
- Story 2 delivers full connector behavior and the 9 plan-defined unit tests.
- Story 3 validates connector integration flow with mocked upstream signals and write-count guarantees.
- Document remains planning-only; no implementation work is performed here.
