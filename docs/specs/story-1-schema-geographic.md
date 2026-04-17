# Story 1: Create `utils/schema_geographic.py`

## Story Goal
Create a new schema-only module, `utils/schema_geographic.py`, that defines all Geographic v2 TypedDict contracts from the CEO plan with zero runtime dependencies.

## Scope
Implement only schema definitions in `utils/schema_geographic.py` using Python TypedDicts.

## Required Definitions
The file must include `from __future__ import annotations` at the top.

Define all of the following TypedDicts:

1. `GeographicMetrics`
2. `Prediction`
3. `SupplyGap`
4. `RankedOpportunity`
5. `GeographicOutput`
6. `WalkForwardIteration`
7. `WalkForwardResult`

## Required Fields
- `Prediction` includes both:
  - `confidence: str`
  - `confidence_score: float`
- `GeographicOutput` includes:
  - `training_window: tuple[str, str]`
- `WalkForwardIteration` includes:
  - `training_window: tuple[str, str]`
  - `previous_prediction: list[Prediction] | None`

## Constraints
- Zero runtime dependencies.
- Schema-only change (no business logic, no orchestration changes, no agent code changes).

## Acceptance Criteria
1. Every TypedDict listed above can be instantiated with all required fields.
2. `json.dumps()` on each TypedDict instance succeeds without error.
3. `confidence_score` is a float (not a string).
4. `training_window` is a tuple (not a single string).
