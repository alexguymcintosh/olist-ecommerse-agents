# Geographic Demand Agent v2 — Implementation Stories

## Context
This story set is derived from:
- `CLAUDE.md`
- `docs/specs/geographic-agent-ceo-plan.md`

Scope is v2 implementation planning only. No code is implemented by this document.

## Global Delivery Rules
- Python 3.12.
- Keep v1 contracts intact (`utils/schema.py`, existing v1 agents, `run_all.py` default path).
- Pandas computes metrics; LLM only consumes pre-computed facts.
- All LLM calls route through `utils/openrouter_client.py`.
- Tests are required for each story before story completion.

## Story Dependency Order
1. Story 1 (`temporal_window`)  
2. Story 2 (`GeographicAgent`)  
3. Story 3 (`walk_forward.py`)  
4. Story 4 (13-iteration integration)  
5. Story 5 (accuracy scoring + validation)

---

## Story 1 — `temporal_window()` in `utils/data_loader.py`

### Goal
Add a window helper that safely slices monthly data and reports actual month coverage for gap detection.

### Scope
- Add `temporal_window(df: pd.DataFrame, start_month: str, end_month: str) -> tuple[pd.DataFrame, int]` to `utils/data_loader.py`.
- Function returns:
  - filtered DataFrame where `month` is within inclusive `[start_month, end_month]`
  - distinct month count in filtered result
- Preserve all existing public functions in `data_loader.py`.

### Requirements
- Accept month strings in `YYYY-MM`.
- Compare against `pd.Period(..., freq="M")`.
- Do not mutate caller DataFrame.
- No LLM usage.

### Acceptance Criteria
1. `temporal_window()` returns `(filtered_df, month_count)` with correct types.
2. Date boundaries are inclusive.
3. Missing-month windows return smaller `month_count` (no silent assumption of continuity).
4. Existing `data_loader` behavior remains unchanged for other functions.

### Test Specification
Create `tests/test_data_loader_temporal_window.py`:
- `test_temporal_window_inclusive_bounds()`
  - Given months `2016-09..2016-12`, window `2016-10..2016-11` returns exactly 2 months.
- `test_temporal_window_reports_missing_month_count()`
  - Given months `2016-10` and `2016-12` only, window `2016-10..2016-12` returns `month_count == 2`.
- `test_temporal_window_returns_dataframe_and_int()`
  - Asserts return signature behavior at runtime.
- `test_temporal_window_does_not_mutate_input()`
  - Verifies input DataFrame unchanged.

---

## Story 2 — GeographicAgent core

### Goal
Implement `agents/geographic/geographic_agent.py` to compute demand momentum, supply gaps, and ranked opportunities for top state/category pairs.

### Scope
- Implement GeographicAgent core methods:
  - `_load_geographic_data()`
  - `_compute_growth_matrix()`
  - `_identify_top5_states()`
  - `_identify_top5_categories()`
  - `_score_confidence()`
  - `_compute_supply_gaps()`
  - `_rank_opportunities()`
  - `run() -> GeographicOutput`
- Use `utils/schema_geographic.py` for output shapes.

### Requirements
- Join chain:
  - orders -> customers (`customer_state`, demand side)
  - orders -> order_items
  - order_items -> products
  - products -> category translation CSV loaded with `encoding="utf-8-sig"`
  - order_items -> sellers (`seller_state`, supply side)
- Assert post-join null ratio for `product_category_name_english < 1%`.
- Growth handling:
  - continuity check before `pct_change()`
  - reindex missing periods with `0`
  - replace `[inf, -inf]` with `NaN`
  - zero-order or sparse month (`<10` orders) => `NaN` growth + LOW confidence
- Supply formula:
  - `predicted_order_volume = current_month_order_count * (1 + predicted_growth)`
  - `supply_gap_ratio = predicted_order_volume / max(current_sellers, 1)`
- Opportunity ranking uses:
  - `predicted_growth_pct * confidence_score * supply_gap_severity`

### Acceptance Criteria
1. GeographicAgent returns valid `GeographicOutput` with populated `predictions`, `supply_gaps`, and `ranked_opportunities`.
2. Supply-side seller counts are computed by `seller_state`, not `customer_state`.
3. No `inf` values propagate to output growth-related structures.
4. LOW confidence is assigned for sparse/invalid growth conditions.
5. Composite score calculation uses numeric `confidence_score`.

### Test Specification
Create `tests/test_geographic_agent.py` (or extend if exists):
- `test_load_geographic_data_uses_utf8_sig_and_low_null_category_ratio()`
- `test_load_geographic_data_uses_seller_state_for_supply_counts()`
- `test_compute_growth_matrix_handles_missing_month_with_reindex()`
- `test_compute_growth_matrix_replaces_inf_with_nan()`
- `test_sparse_month_sets_low_confidence()`
- `test_supply_gap_ratio_formula()`
- `test_rank_opportunities_uses_confidence_score_numeric()`
- `test_run_returns_geographic_output_contract()`

---

## Story 3 — `walk_forward.py` orchestrator

### Goal
Implement walk-forward orchestration that runs monthly prediction iterations, carries previous predictions, and writes structured results.

### Scope
- Create/complete `walk_forward.py`.
- Build rolling training windows via `temporal_window()`.
- Call GeographicAgent for each iteration.
- Include `previous_prediction` in each `WalkForwardIteration` after iteration 1.
- Emit `WalkForwardResult` JSON file in `outputs/`.
- Support CLI:
  - `--start`
  - `--end`
  - `--validate`
  - `--ranked`
- Support env flag:
  - `WALK_FORWARD_VERBOSE=1` for verbose logs/prompts.

### Requirements
- End-to-end schedule supports 13 iterations (from 25 raw months and 12-month training lookback).
- Per-iteration payload includes:
  - `training_window: tuple[str, str]`
  - `prediction_month`
  - `previous_prediction`
  - `predictions`, `supply_gaps`, `ranked_opportunities`
  - `validation` (may be placeholder in this story; fully specified in Story 5)
- JSON output must be serializable with standard library `json`.

### Acceptance Criteria
1. CLI execution with `--start 1 --end 3` completes successfully and writes JSON output.
2. Iteration objects include `training_window` and `previous_prediction` fields as specified.
3. `completed_iterations` equals the number of executed iterations.
4. No schema violations against `WalkForwardResult`.

### Test Specification
Create `tests/test_walk_forward.py`:
- `test_walk_forward_runs_three_iterations()`
- `test_walk_forward_carries_previous_prediction()`
- `test_walk_forward_emits_training_window_tuple()`
- `test_walk_forward_json_serializable()`
- `test_walk_forward_respects_start_end_bounds()`

---

## Story 4 — Integration test: 13 iterations end-to-end

### Goal
Validate complete v2 loop across all 13 walk-forward iterations with real dataset wiring.

### Scope
- Add integration test(s) for full run.
- Validate output contracts and iteration count over complete range.
- Confirm output artifact integrity.

### Requirements
- Full run executes 13 iterations in sequence.
- Output file exists and loads as JSON.
- Iteration array length matches `completed_iterations`.
- Every iteration has required keys and nested list shapes.
- System remains additive to v1 behavior.

### Acceptance Criteria
1. End-to-end run completes without crash.
2. `completed_iterations == 13`.
3. `len(result["iterations"]) == 13`.
4. Every iteration includes:
   - `training_window` tuple
   - `prediction_month`
   - `predictions`, `supply_gaps`, `ranked_opportunities`
5. Result remains parseable and schema-compatible with `WalkForwardResult`.

### Test Specification
Create `tests/test_walk_forward_integration.py`:
- `test_walk_forward_full_13_iterations_end_to_end()`
  - Runs full range
  - Asserts `completed_iterations == 13`
  - Asserts output file exists and loads
- `test_walk_forward_iteration_payload_shapes()`
  - Checks required keys for each iteration
- `test_v1_path_unchanged_after_v2_integration()`
  - Sanity checks existing v1 entrypoint still imports/runs baseline path

---

## Story 5 — Accuracy scoring and validation

### Goal
Implement deterministic scoring of predictions vs actuals and persist validation metrics per iteration and aggregate.

### Scope
- Add validation/scoring functions used by `walk_forward.py`.
- Compute per-item metrics:
  - `predicted_growth`
  - `actual_growth`
  - `directional_accuracy`
  - `pct_error`
- Compute aggregate metrics in `aggregate_accuracy`.
- Handle sparse/invalid actuals safely (NaN-aware).

### Requirements
- Validation executes when `--validate` is enabled and actual month data is available.
- No divide-by-zero / inf propagation.
- Directional accuracy logic:
  - true if sign(predicted_growth) == sign(actual_growth)
- Percent error logic:
  - safe computation with guard for near-zero denominator.
- Aggregate metrics include at least:
  - average directional accuracy
  - average percent error
  - count of scored predictions

### Acceptance Criteria
1. Per-iteration `validation` is populated with numeric/scalar values for scored rows.
2. `aggregate_accuracy` is populated and stable across repeated runs on same data.
3. Sparse or invalid rows are skipped or marked safely without crashing.
4. Validation remains JSON-serializable.

### Test Specification
Create `tests/test_accuracy_scoring.py`:
- `test_directional_accuracy_sign_match()`
- `test_pct_error_safe_division_guard()`
- `test_validation_handles_nan_and_sparse_rows()`
- `test_aggregate_accuracy_computation()`
- `test_validation_payload_json_serializable()`

---

## Definition of Done (All 5 Stories)
- All story acceptance criteria pass.
- All new test modules pass with `pytest`.
- v1 baseline tests remain passing.
- No deviations from schema contracts in `utils/schema_geographic.py`.
- No direct API calls bypassing `utils/openrouter_client.py` where LLM usage exists.
