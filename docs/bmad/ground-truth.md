# Ground Truth State Document
**Audit date:** 2026-04-19 (updated after VIZ-01/02/03 implementation)  
**Scope:** Full codebase vs CLAUDE.md claims and visualisation-stories.md spec

---

## 1. Overall Status: What CLAUDE.md Claims vs Reality

CLAUDE.md has been updated to reflect current reality. All previous `[ ]` items were already implemented. CLAUDE.md now also lists the visualisation scripts added in the VIZ-01/02/03 session.

| CLAUDE.md claim | Actual status |
|---|---|
| `[x] utils/config.py` | **DONE** — `utils/config.py` exists with all focus parameters + `get_top_states` / `get_top_categories` helpers |
| `[x] utils/memory.py` | **DONE** — `utils/memory.py` exists with full SQLite layer |
| `[x] utils/schema_agents.py` | **DONE** — `utils/schema_agents.py` exists with all 5 TypedDicts |
| `[x] Supply Quality Agent` | **DONE** — `agents/supply_quality/supply_quality_agent.py` fully implemented |
| `[x] Customer Readiness Agent` | **DONE** — `agents/customer_ready/customer_ready_agent.py` fully implemented |
| `[x] Logistics Agent` | **DONE** — `agents/logistics/logistics_agent.py` fully implemented |
| `[x] Connector Agent` | **DONE** — `agents/connector/connector_agent.py` fully implemented |
| `[x] walk_forward_full.py` | **DONE** — 5-agent orchestrator; instrumented with `agent_timings` in VIZ-01 |
| `[x] dashboard.py` | **DONE** — real-time ops dashboard (VIZ-01); port 5001 ⚠️ spec says 8765 |
| `[x] memory_viz.py` | **DONE** — memory visualisation report (VIZ-02) |
| `[x] perf_viz.py` | **DONE** — performance visualisation report (VIZ-03) |
| `[x] All 84 tests passing` | **VERIFIED** — 84/84 pass after all VIZ changes |
| `[ ] Integration test — all 5 agents × 13 iterations` | **PARTIALLY DONE** — integration smoke tests exist and pass; full 13-iteration live run not done (only 5 iterations exist) |

---

## 2. File Structure: Actual vs Intended

### Actual root-level files (non-hidden, non-pycache)

```
CLAUDE.md
TODOS.md
conftest.py
dashboard.py          ← VIZ-01: real-time ops dashboard (port 5001)
memory_viz.py         ← VIZ-02: memory visualisation report (NEW)
perf_viz.py           ← VIZ-03: performance visualisation report (NEW)
pytest.ini
requirements.txt
run_tests.py
visualise.py          ← legacy, pre-VIZ session, not part of VIZ spec
walk_forward.py       ← geo-only version, matches CLAUDE.md
walk_forward_full.py  ← 5-agent version; instrumented with agent_timings (VIZ-01)
```

### Agents directory — all 5 exist

```
agents/__init__.py
agents/geographic/geographic_agent.py    ← 17 KB
agents/supply_quality/supply_quality_agent.py  ← 22 KB
agents/customer_ready/customer_ready_agent.py  ← 18 KB
agents/logistics/logistics_agent.py      ← 17 KB
agents/connector/connector_agent.py      ← 21 KB
```

### Utils directory — all exist

```
utils/config.py         ← focus params + get_top_states/categories
utils/data_loader.py    ← load_all + temporal_window
utils/memory.py         ← SQLite Memory class
utils/openrouter_client.py  ← query_llm, build_analyst_prompt, parse_batch_llm_response
utils/schema_agents.py  ← SupplyQualityOutput, CustomerReadinessOutput, LogisticsOutput, ConnectorDecision, ConnectorOutput
utils/schema_geographic.py  ← GeographicMetrics, Prediction, SupplyGap, RankedOpportunity, GeographicOutput, WalkForwardIteration, WalkForwardResult
```

### Tests directory — actual structure differs from CLAUDE.md

CLAUDE.md lists `tests/geographic/ — complete` but **no such folder exists**. Geographic tests are at the root of `tests/`.

**Actual test layout:**
```
tests/
├── test_accuracy_scoring.py          ← root level (not in tests/geographic/)
├── test_data_loader_temporal_window.py
├── test_geographic_agent.py
├── test_memory.py
├── test_schema_geographic.py
├── test_walk_forward.py
├── supply_quality/
│   ├── test_schema_supply_quality_output.py
│   └── test_supply_quality_agent.py
├── customer_ready/
│   ├── test_schema_customer_readiness.py
│   └── test_customer_ready_agent.py
├── logistics/
│   ├── test_schema_logistics_output.py
│   └── test_logistics_agent.py
├── connector/
│   ├── test_schema_connector.py
│   └── test_connector_agent.py
└── integration/
    ├── test_connector_smoke.py
    ├── test_customer_ready_smoke.py
    ├── test_logistics_smoke.py
    ├── test_memory_integration.py
    └── test_supply_quality_smoke.py
```

**Expected layout per CLAUDE.md:**
```
tests/geographic/       ← DOES NOT EXIST (tests are at root level)
tests/supply_quality/   ← exists ✓
tests/customer_ready/   ← exists ✓
tests/logistics/        ← exists ✓
tests/connector/        ← exists ✓
tests/integration/      ← exists ✓
```

### Docs directory

```
docs/bmad/
├── connector-stories.md
├── customer-ready-stories.md
├── logistics-stories.md
├── memory-stories.md
├── supply-quality-stories.md
├── v2-stories.md
├── implementation-artifacts/   ← directory (empty or contains files)
├── planning-artifacts/         ← directory
└── test-artifacts/             ← directory
docs/plans/
├── 5-agent-ceo-plan.md
└── 5-agent-eng-review.md
docs/specs/
├── 2026-04-14-ceo-plan.md
├── geographic-agent-ceo-plan.md
├── geographic-agent-eng-review.md
└── story-1-schema-geographic.md
```

### Outputs directory

```
outputs/
├── dashboard.html
├── dashboard_full.html    ← git untracked
├── perf_report.html       ← generated by perf_viz.py (VIZ-03)
├── walk_forward_full_2026-04-19-12-30.json
└── iterations/
    ├── 2017-09.json  ← has agent_timings (VIZ-01 instrumented run)
    ├── 2017-10.json  ← has agent_timings (VIZ-01 instrumented run)
    ├── 2017-12.json  ← has agent_timings (VIZ-01 instrumented run)
    ├── 2018-01.json  ← NO agent_timings (pre-VIZ-01 run, key absent)
    └── 2018-02.json  ← NO agent_timings (pre-VIZ-01 run, key absent)
```

**Note:** November 2017 is still missing from `iterations/`. 2018-01 and 2018-02 pre-date VIZ-01 instrumentation and contain no `agent_timings` key. All three visualisation scripts handle this correctly (graceful degradation).

### Memory report outputs (runtime, not committed)

```
memory_report_2026-04-19-13-17.html  ← generated by memory_viz.py (VIZ-02)
```

### Memory databases (runtime, not committed)

```
memory_2026-04-19-12-19.db  ← 2-iteration full 5×5 run
memory_2026-04-19-12-29.db  ← 1-iteration 1×1 smoke run
memory.db                   ← listed as modified in git status but FILE DOES NOT EXIST
                               (git tracks its deletion, shown as " M" in status)
```

---

## 3. What Is Actually Implemented

### Agent 1 — Geographic Agent (`agents/geographic/geographic_agent.py`)

**Implemented:** Full spec + one optimization beyond spec.

- `_load_geographic_data()` — joins orders/customers/order_items/products/categories/sellers, adds `month` period, enforces `< 3%` null category ratio (hard stop)
- `_compute_growth_matrix()` — monthly revenue, MoM growth, inf guard, missing-month reindex, momentum (3-month mean), order counts, sparse flags
- `_score_confidence()` — HIGH/MEDIUM/LOW based on order count ≥ 10 and growth magnitude
- `_compute_supply_gaps()` — predicted_order_volume / max(sellers, 1)
- `_rank_opportunities()` — composite = predicted_growth × confidence_score × supply_gap_severity
- `_predict_next_month_growth()` — single-pair LLM call (kept for backwards compat, not used in `run()`)
- `_predict_batch_growth()` — **batch LLM call for all 25 pairs** (beyond original spec which described per-pair calls)
- `run()` — orchestrates full pipeline, returns `GeographicOutput`

**Deviations from spec:**
- Uses batch LLM call (1 call per iteration instead of 25) — optimisation added beyond spec
- Does NOT use `_identify_top5_states` / `_identify_top5_categories` in `run()` — uses `FOCUS_STATES` / `FOCUS_CATEGORIES` from config instead
- `_identify_top5_states` and `_identify_top5_categories` exist but are dead code in production path

### Agent 2 — Supply Quality Agent (`agents/supply_quality/supply_quality_agent.py`)

**Implemented:** Full spec + batch optimisation.

- Pandas computes: seller count, weighted avg review score (last 3 months), avg delivery days, churn risk/rate, top seller id, seller concentration
- Sparse detection: latest_order_count < MIN_MONTHLY_ORDERS skips LLM
- `_assess_supply_batch()` — 1 LLM call for all non-sparse pairs
- Includes `last_month_context` in LLM payload when prev_memory provided
- `critical_seller_gap` flag when seller_count == 0
- `high_seller_churn` flag when churn_rate > 0.5

**Key difference from CLAUDE.md spec:** Agent accepts `training_df` from walk-forward orchestrator (can also build its own from `self.data` if `None`). The `_enrich_training_df()` method handles partially-joined dataframes.

**Memory columns written by `walk_forward_full.py`:**  
`sq_seller_count, sq_avg_review, sq_avg_delivery_days, sq_churn_risk, sq_top_seller_id, sq_reasoning`  
Note: `churn_rate` and `seller_concentration` are in `SupplyQualityOutput` TypedDict but are **NOT written to memory** (not in memory schema either).

### Agent 3 — Customer Readiness Agent (`agents/customer_ready/customer_ready_agent.py`)

**Implemented:** Full spec + batch optimisation.

- Pandas computes: avg_spend (per-order dedup), order_volume_trend (MoM % on distinct orders), top_payment_type, high_value_customer_count (above median spend), repeat_rate, installment_pct (credit card orders with ≥ 6 installments)
- `_assess_batch()` — 1 LLM call for all 25 pairs
- Accepts `llm_client` injection for testability
- `_write_memory()` writes to memory if `self.memory` is set

**Key difference:** `installment_pct` is in the TypedDict output but **NOT written to memory** (memory schema doesn't include it, matches CLAUDE.md).

**Memory columns written:** `cr_avg_spend, cr_order_volume_trend, cr_top_payment_type, cr_high_value_customer_count, cr_repeat_rate, cr_reasoning`

### Agent 4 — Logistics Agent (`agents/logistics/logistics_agent.py`)

**Implemented:** Full spec + batch optimisation + one extra metric.

- Pandas computes: avg_delivery_days, pct_on_time, freight_ratio, fastest_seller_state, delivery_variance (std dev), cross_state_dependency
- `_llm_batch_assessment()` — 1 LLM call for all 25 pairs
- `_write_memory()` writes to memory if `self.memory` is set
- **Extra metric:** `delivery_variance` is computed AND written to memory as `log_delivery_variance`

**Memory columns written:** `log_avg_delivery_days, log_pct_on_time, log_freight_ratio, log_fastest_seller_state, log_delivery_variance, log_reasoning`

### Agent 5 — Connector Agent (`agents/connector/connector_agent.py`)

**Implemented:** Full spec with one deviation in follow-up loop behaviour.

- Does NOT subclass any base class ✓
- `_composite_score()` — `(geo × 0.35 + supply × 0.25 + customer × 0.20 + logistics × 0.20) × predicted_growth`
- Reads `prev_month` memory for outcome feedback text
- `_batch_connector_decisions()` — 1 LLM call for all pairs
- Follow-up loop: triggers when top composite score > 2× runner-up; generates follow-up question and LLM response
- `memory.write_row()` called per decision

**Deviation from spec — follow-up loop:**  
CLAUDE.md says: *"Agent re-runs focused analysis on that subset."*  
Reality: The follow-up question is sent **directly to the LLM**, not to the actual domain agent object. The domain agent is NOT re-instantiated or re-run. The follow-up is purely an additional LLM inference call.

**Deviation from spec — briefing:**  
CLAUDE.md says: *"Rich terminal narrative"*  
Reality: briefing is a 3-line text of form `"1. SP x health_beauty -> recruit_sellers (15.23)"`

**Memory columns written:** `conn_decision, conn_confidence, conn_reasoning, conn_most_predictive_agent`  
Note: `conn_actual_outcome` is written by `walk_forward_full.py` validation pass (not by connector itself).

---

## 4. Memory Schema: Actual vs CLAUDE.md

CLAUDE.md documents this schema:

```sql
CREATE TABLE agent_memory (
 state TEXT, category TEXT, month TEXT,
 geo_predicted_growth REAL, geo_actual_growth REAL,
 geo_directional_accuracy INTEGER, geo_confidence TEXT,
 geo_confidence_score REAL, geo_reasoning TEXT,
 sq_seller_count INTEGER, sq_avg_review REAL,
 sq_avg_delivery_days REAL, sq_churn_risk TEXT,
 sq_top_seller_id TEXT, sq_reasoning TEXT,
 cr_avg_spend REAL, cr_order_volume_trend REAL,
 cr_top_payment_type TEXT, cr_high_value_customer_count INTEGER,
 cr_repeat_rate REAL, cr_reasoning TEXT,
 log_avg_delivery_days REAL, log_pct_on_time REAL,
 log_freight_ratio REAL, log_fastest_seller_state TEXT,
 log_reasoning TEXT,          ← CLAUDE.md has this IMMEDIATELY after fastest_seller_state
 conn_decision TEXT, conn_confidence TEXT,
 conn_reasoning TEXT, conn_actual_outcome TEXT,
 conn_most_predictive_agent TEXT,
 PRIMARY KEY (state, category, month)
);
```

**Actual schema in `utils/memory.py`:**

```sql
CREATE TABLE agent_memory (
 state TEXT NOT NULL, category TEXT NOT NULL, month TEXT NOT NULL,
 geo_predicted_growth REAL, geo_actual_growth REAL,
 geo_directional_accuracy INTEGER, geo_confidence TEXT,
 geo_confidence_score REAL, geo_reasoning TEXT,
 sq_seller_count INTEGER, sq_avg_review REAL,
 sq_avg_delivery_days REAL, sq_churn_risk TEXT,
 sq_top_seller_id TEXT, sq_reasoning TEXT,
 cr_avg_spend REAL, cr_order_volume_trend REAL,
 cr_top_payment_type TEXT, cr_high_value_customer_count INTEGER,
 cr_repeat_rate REAL, cr_reasoning TEXT,
 log_avg_delivery_days REAL, log_pct_on_time REAL,
 log_freight_ratio REAL, log_fastest_seller_state TEXT,
 log_delivery_variance REAL,  ← EXTRA COLUMN — NOT in CLAUDE.md schema
 log_reasoning TEXT,
 conn_decision TEXT, conn_confidence TEXT,
 conn_reasoning TEXT, conn_actual_outcome TEXT,
 conn_most_predictive_agent TEXT,
 PRIMARY KEY (state, category, month)
);
```

**Discrepancy:** `log_delivery_variance REAL` is in the actual schema but missing from CLAUDE.md. This column IS populated by logistics agent and walk_forward_full.py.

---

## 5. Tests: Coverage and Pass Status

**All 84 tests pass.** Elapsed: ~31 seconds (data-loading dominated).

### Test coverage by area

| Area | Files | Tests | Status |
|---|---|---|---|
| Accuracy scoring | `test_accuracy_scoring.py` | 5 | All pass |
| Data loader | `test_data_loader_temporal_window.py` | 4 | All pass |
| Geographic agent | `test_geographic_agent.py` | 8 | All pass |
| Memory | `test_memory.py` | 5 | All pass |
| Schema geographic | `test_schema_geographic.py` | 2 | All pass |
| Walk forward (geo-only) | `test_walk_forward.py` | 5 | All pass |
| Supply quality schema | `supply_quality/test_schema_supply_quality_output.py` | 3 | All pass |
| Supply quality agent | `supply_quality/test_supply_quality_agent.py` | 8 | All pass |
| Customer readiness schema | `customer_ready/test_schema_customer_readiness.py` | 3 | All pass |
| Customer readiness agent | `customer_ready/test_customer_ready_agent.py` | 9 | All pass |
| Logistics schema | `logistics/test_schema_logistics_output.py` | 4 | All pass |
| Logistics agent | `logistics/test_logistics_agent.py` | 9 | All pass |
| Connector schema | `connector/test_schema_connector.py` | 4 | All pass |
| Connector agent | `connector/test_connector_agent.py` | 9 | All pass |
| Integration — connector | `integration/test_connector_smoke.py` | 1 | Pass |
| Integration — customer ready | `integration/test_customer_ready_smoke.py` | 1 | Pass |
| Integration — logistics | `integration/test_logistics_smoke.py` | 1 | Pass |
| Integration — memory | `integration/test_memory_integration.py` | 1 | Pass |
| Integration — supply quality | `integration/test_supply_quality_smoke.py` | 1 | Pass |

### What is NOT tested

- `walk_forward_full.py` — no dedicated unit tests (only covered by integration smoke tests for individual agents)
- `dashboard.py` — not tested (VIZ-01; port mismatch with spec also uncaught by tests)
- `memory_viz.py` — not tested (VIZ-02)
- `perf_viz.py` — not tested (VIZ-03)
- `visualise.py` — not tested (legacy)
- `config.get_top_states` / `config.get_top_categories` — not tested
- Connector follow-up loop LLM path — tested via mock; actual LLM integration not tested
- 13-iteration full end-to-end run — not in test suite
- `agent_timings` field in iteration JSON — no test asserts its presence or shape

---

## 6. Walk-Forward Runs: Actual State

### walk_forward.py (geo-only, legacy)
- Runs GeographicAgent in isolation
- 13 iterations Sep 2017 → Aug 2018
- No memory integration, no connector
- Tests pass

### walk_forward_full.py (5-agent, new)
- Orchestrates all 5 agents in sequence per iteration
- Accepts `--n-states`, `--n-categories`, `--max-iterations`, `--run-id`, `--iterations`
- Uses dynamic state/category selection (`get_top_states` / `get_top_categories`) rather than hardcoded `FOCUS_STATES` / `FOCUS_CATEGORIES`
- Per-iteration error recovery (try/except + continue) — **already implements TODOS.md P1**
- Run-isolated DB via `--run-id` — **already implements TODOS.md P2 (memory version key)**
- Writes per-iteration JSON to `outputs/iterations/`
- Validates previous iteration's connector decisions vs actuals
- **VIZ-01 addition:** emits `agent_timings` dict in each iteration JSON — `wall_seconds` (float or null) and `llm_calls` (int) per agent; error iterations write `null`/`0`

### Real run history

| DB file | Iterations | Grid | Months covered |
|---|---|---|---|
| `memory_2026-04-19-12-19.db` | 2 | 5 states × 5 categories | 2017-09, 2017-10 |
| `memory_2026-04-19-12-29.db` | 1 | 1 state × 1 category | 2017-09 |

The `outputs/iterations/` folder contains 5 iteration outputs:  
`2017-09, 2017-10, 2017-12, 2018-01, 2018-02` — **2017-11 is missing** (skipped, reason unknown from artifacts alone).

### Validation accuracy (from iteration outputs)

| Validated month | Validated | Correct | Incorrect | Accuracy |
|---|---|---|---|---|
| 2017-09 | N/A (first iteration) | — | — | — |
| 2017-10 | 9 | 7 | 2 | 77.8% |
| 2017-12 | 9 | 5 | 4 | 55.6% |
| 2018-01 | 9 | 4 | 5 | 44.4% |
| 2018-02 | 9 | 7 | 2 | 77.8% |

**Note:** `geo_actual_growth` and `geo_directional_accuracy` columns are **not populated** in the 2-iteration DB (`memory_2026-04-19-12-19.db`) — only `conn_actual_outcome` is written back. The `geo_directional_accuracy` write path exists in `walk_forward_full.py` but the field remains NULL in the checked DB.

---

## 7. TODOS.md vs Reality

Several TODOS.md items are **already implemented** but the file hasn't been updated:

| TODOS.md item | Priority | Status |
|---|---|---|
| Per-iteration error recovery in walk_forward_full.py | P1 | **DONE** — try/except in main loop |
| Memory run/version key (`--run-id`) | P2 | **DONE** — `--run-id` flag + `memory_{run_id}.db` |
| Batch LLM calls for agents 2-4 | P2 | **DONE** — all domain agents use batch calls |
| Connector auto-adjusting composite weights | P3 | Not done |
| Volume-weighted composite score | P3 | Not done |
| Geographic v2 ConnectorOutput extension | post-MVP | Not done |
| WalkForwardIteration schema typing | post-MVP | Not done |

---

## 8. Spec Deviations by Agent

### Geographic Agent

| Spec | Reality |
|---|---|
| 25 per-pair LLM calls per iteration | 1 batch LLM call per iteration |
| `_identify_top5_states` drives states | Dead code — `FOCUS_STATES` config used in `run()` |

### Supply Quality Agent

| Spec | Reality |
|---|---|
| Per-pair LLM calls | 1 batch call (non-sparse pairs only) |
| Output includes: `agent, timestamp, state, category, month, seller_count, avg_review_score, avg_delivery_days, churn_risk, churn_rate, top_seller_id, seller_concentration, supply_confidence, reasoning, risk_flags` | Matches spec ✓ |

### Customer Readiness Agent

| Spec | Reality |
|---|---|
| `AGENT_NAME = "customer_readiness"` | Matches ✓ |
| `installment_pct` in output schema | Present in TypedDict, NOT written to memory ✓ |
| Per-pair LLM calls | 1 batch call |

### Logistics Agent

| Spec | Reality |
|---|---|
| Output includes: `avg_delivery_days, pct_on_time, freight_ratio, fastest_seller_state, delivery_variance, cross_state_dependency` | All present ✓ |
| Memory writes: `log_avg_delivery_days, log_pct_on_time, log_freight_ratio, log_fastest_seller_state, log_reasoning` | Extra: `log_delivery_variance` also written (not in CLAUDE.md schema) |

### Connector Agent

| Spec | Reality |
|---|---|
| "Agent re-runs focused analysis on that subset" (follow-up loop) | Follow-up is a direct LLM call — domain agent NOT re-run |
| "Rich terminal narrative" for briefing | Simple 3-line text summary |
| Composite = `(geo × 0.35 + supply × 0.25 + customer × 0.20 + logistics × 0.20) × geo_predicted_growth` | Matches ✓ |
| Does NOT subclass any base class | Correct ✓ |

---

## 9. Missing Items and Open Flags

1. **No `tests/geographic/` folder** — CLAUDE.md directory structure lists it, doesn't exist. Geographic tests live at `tests/` root.
2. **No `run_all.py`** — referenced in TODOS.md (`run_all.py --geographic --save`), does not exist.
3. **`visualise.py`** — legacy file at root; not part of VIZ spec, not tested.
4. **`walk_forward_full.py`** — not in CLAUDE.md directory structure but now listed in Current Status.
5. **Full 13-iteration run has never completed** — only 5 iterations in `outputs/iterations/`, with 2017-11 missing.
6. **`conn_actual_outcome` not backfilled** — the 2-iteration memory DB has no outcome, directional accuracy, or actual growth recorded (requires 3rd iteration to validate 2nd).
7. **`openai` package** in `requirements.txt` (`openai==1.30.0`) — not used anywhere in the codebase. All LLM calls go through `requests` directly to OpenRouter. Dead dependency.
8. **⚠️ PORT MISMATCH — `dashboard.py`:** Spec (`visualisation-stories.md`) says `localhost:8765`. Implementation defaults to `5001`. Nothing in the test suite catches this. Needs reconciliation before team-wide use.
9. **`agent_timings` absent from 2018-01 and 2018-02 iteration JSONs** — these were produced before VIZ-01 instrumentation. All three viz scripts handle this correctly, but re-running those iterations would populate the field.
10. **No tests for any VIZ script** — `dashboard.py`, `memory_viz.py`, `perf_viz.py` are all untested.

---

## 10. Quick Reference: What Passes, What Works, What Doesn't

| Component | Implemented | Tests | Live Run |
|---|---|---|---|
| Geographic Agent | ✅ | ✅ 8/8 | ✅ used in walk_forward_full |
| Supply Quality Agent | ✅ | ✅ 8/8 | ✅ used in walk_forward_full |
| Customer Readiness Agent | ✅ | ✅ 9/9 | ✅ used in walk_forward_full |
| Logistics Agent | ✅ | ✅ 9/9 | ✅ used in walk_forward_full |
| Connector Agent | ✅ | ✅ 9/9 | ✅ used in walk_forward_full |
| Memory layer | ✅ | ✅ 5/5 | ✅ 3 DBs exist |
| Config | ✅ | ✅ (indirect) | ✅ |
| walk_forward.py (geo) | ✅ | ✅ 5/5 | — |
| walk_forward_full.py (5-agent) | ✅ | ❌ no direct tests | ✅ 5 iterations ran |
| Integration smoke tests | ✅ | ✅ 5/5 | — |
| 13-iteration full run | ❌ not done | ❌ | ❌ |
| dashboard.py (VIZ-01) | ✅ | ❌ | ⚠️ port 5001 (spec: 8765) |
| memory_viz.py (VIZ-02) | ✅ | ❌ | ✅ report generated |
| perf_viz.py (VIZ-03) | ✅ | ❌ | ✅ report generated |
| visualise.py | exists (legacy) | ❌ | unknown |
| TODOS.md accuracy | stale | — | — |
| CLAUDE.md current status | **updated** | — | — |
