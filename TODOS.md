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

---

## 5-Agent System — From CEO Plan 2026-04-18

### P1: walk_forward_full.py per-iteration error recovery

**What:** Wrap each iteration in `try/except`. On failure, write error state to memory and continue to next iteration.
**Why:** 1,300 LLM calls over ~22 minutes. Any network blip or rate limit aborts all iterations with no recovery.
**Pros:** Partial results better than nothing. Missing iterations logged clearly.
**Cons:** Silent failures can make accuracy numbers misleading (missing iterations excluded from aggregate). Add an iteration-level `status` column to memory to track failed iterations.
**Effort:** S → S with CC. **Priority: P1** — address before running full 13-iteration experiment.

### P2: Memory run/version key

**What:** Add `run_id TEXT` column to `agent_memory` table. Pass a run identifier when creating `Memory(db_path="memory_runX.db")`.
**Why:** Re-running with different models or prompts overwrites the same `(state, category, month)` rows. Experiment comparison is impossible today.
**Workaround now:** Pass different `db_path` per run (e.g., `Memory("memory_run2.db")`).
**Pros:** Enables experiment comparison. Clean separation between runs.
**Cons:** Breaks existing schema — requires migration or fresh DB per change.
**Effort:** M → S with CC. **Priority: P2.**

### P2: Batch LLM calls for agents 2-4

**What:** Instead of 25 LLM calls per agent per iteration, make 1 LLM call per agent with all 25 state×category contexts. Return JSON array of 25 assessments.
**Why:** Current: 100 LLM calls/iteration × 13 = 1,300 total (~22 min). Batching domain agents: ~42 calls total (~40 min saved).
**Pros:** 3-5× faster. Fewer API calls. Cheaper.
**Cons:** Larger prompts. More complex JSON parsing (`[{}, {}, ...]` response). Deviates from geographic agent per-pair pattern.
**Start here:** Implement for one agent first (e.g., supply_quality), validate output matches per-call version, then replicate.
**Effort:** M → S with CC. **Priority: P2.**

### P3: Connector auto-adjusting composite weights

**What:** After 3+ iterations, read `conn_most_predictive_agent` history from memory. If one agent is consistently flagged as most predictive, increase its composite weight by +5%, reduce others proportionally.
**Why:** Fixed weights don't adapt. Logistics may consistently outperform geographic signal in certain states.
**Caveat:** `most_predictive_agent` is LLM-assigned narration, not measurement. Wait until `conn_actual_outcome` is validated before trusting it for weight calibration.
**Effort:** M → S with CC. **Priority: P3** — revisit after 13 iterations of data.

### P3: Volume-weighted composite score

**What:** Add order_count term: `composite += (order_count / max_order_count_in_grid) × 0.10`, reduce geo/supply/customer/logistics weights by 2.5% each.
**Why:** A tiny segment with noisy +200% growth can outrank a large profitable segment with +15% growth. Volume weighting prevents low-signal pairs from dominating the ranking.
**When to act:** If low-volume pairs consistently top the rankings in practice after the full 13-iteration run.
**Effort:** S → XS with CC. **Priority: P3.**
