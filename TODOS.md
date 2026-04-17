# Deferred / post-MVP backlog

## Geographic v2 — ConnectorAgent extension (from geographic-agent-ceo-plan MVP cut)

- Add `geographic_output: GeographicOutput | None` to `ConnectorOutput`.
- When `geographic_output` is present, include geographic predictions and ranked opportunities in briefing synthesis.
- **Acceptance (when picked up):** `run_all.py --geographic --save` produces combined output with geographic section.

```python
# Target shape (reference)
class ConnectorOutput(TypedDict):
    timestamp: str
    cross_domain_insights: list[str]
    strategic_recommendation: str
    priority_actions: list[PriorityAction]
    briefing: str
    geographic_output: GeographicOutput | None
```

## Geographic v2 — schema typing follow-up

- `WalkForwardIteration.validation` is currently `dict` (untyped for MVP).
- `WalkForwardResult.aggregate_accuracy` is currently `dict` (untyped for MVP).
- Post-MVP: replace both with explicit TypedDict contracts in `utils/schema_geographic.py` and update tests accordingly.

## Geographic v2 — LLM cost/latency optimization

- `_predict_next_month_growth()` currently makes 25 LLM calls per iteration (5x5 grid).
- Consider batching predictions into one LLM call per iteration to reduce cost and latency.
