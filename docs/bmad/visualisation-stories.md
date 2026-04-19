# Visualisation Stories
**Author:** John (Product Manager)  
**Date:** 2026-04-19  
**Source context:** `docs/bmad/ground-truth.md`, `CLAUDE.md`  
**Output location:** `docs/bmad/`

---

## Foundational constraint (all three stories)

**Compatibility gate:** After implementing each story, the following command must complete without error and produce a valid `outputs/walk_forward_full_*.json` file:

```
python walk_forward_full.py --n-states 1 --n-categories 1 --iterations 2
```

No story may modify the agent logic, memory schema, or connector output contract. All three scripts are **readers** or **lightweight emitters** only. `walk_forward_full.py` may be extended to emit additional telemetry fields (Story 1), but existing fields must remain unchanged.

---

## Story 1 — Real-Time Ops Dashboard

**ID:** VIZ-01  
**File:** `dashboard.py` (rewrite of existing file)  
**Priority:** P1 — needed while the 13-iteration run is in progress

---

### User story

As a team member watching a walk-forward run in another terminal,  
I want a browser tab that refreshes itself every 5 seconds and shows exactly what each agent did this iteration,  
so that I can catch a failing iteration or a poor decision without tailing log files.

---

### Context

`dashboard.py` already exists at project root but its current state is unknown and untested (ground-truth §9). This story replaces it entirely.

The primary data source is `outputs/iterations/YYYY-MM.json`. Each file is written by `walk_forward_full.py` at the end of each iteration. The dashboard must not touch the memory DB.

**Current gap:** The iteration JSON contains `validation`, `top_connector_decisions`, and `agent_signal_summary` but does **not** contain per-agent timing or API call counts. Story 1 includes a small instrumentation addition to `walk_forward_full.py` to emit this data. The instrumentation must be additive-only — no existing fields may be renamed or removed.

---

### Scope

**Part A — Instrument `walk_forward_full.py`**

Add an `agent_timings` key to each iteration JSON file. Shape:

```json
"agent_timings": {
  "geographic":        { "wall_seconds": 4.2,  "llm_calls": 1 },
  "supply_quality":    { "wall_seconds": 3.1,  "llm_calls": 1 },
  "customer_readiness":{ "wall_seconds": 2.8,  "llm_calls": 1 },
  "logistics":         { "wall_seconds": 2.5,  "llm_calls": 1 },
  "connector":         { "wall_seconds": 5.1,  "llm_calls": 1 }
}
```

- `wall_seconds` — elapsed time from agent `.run()` call to return, measured with `time.perf_counter()`
- `llm_calls` — hardcoded to `1` for all batch agents (all agents now use a single batch LLM call per iteration); this is a count, not a timer — accuracy over false precision
- If an agent raises an exception (error iteration), write `{ "wall_seconds": null, "llm_calls": 0 }` for that agent

The `_write_iteration_output` helper must accept and pass through `agent_timings`. All call sites in `run_walk_forward_full` must pass the dict.

**Part B — Rewrite `dashboard.py`**

A single-file Python web server (stdlib `http.server` only — no Flask, no FastAPI). Serves one HTML page. The page polls `GET /data` every 5 seconds via `fetch`. The server reads `outputs/iterations/` on each `/data` request and returns JSON.

**UI layout — terminal style, no charts, no CSS frameworks:**

```
OLIST WALK-FORWARD  |  last updated: 2026-04-19 14:23:05 UTC
─────────────────────────────────────────────────────────────

ITER 5  2018-01  [OK]

  AGENT              CALLS   TIME     SIGNAL
  geographic           1     4.2s     HIGH / MEDIUM / LOW (geo confidence distribution)
  supply_quality       1     3.1s     STRONG: 3  ADEQUATE: 12  WEAK: 10
  customer_readiness   1     2.8s     HIGH: 6  MEDIUM: 14  LOW: 5
  logistics            1     2.5s     STRONG: 8  ADEQUATE: 11  WEAK: 6
  connector            1     5.1s     —

  TOP DECISION:   SP × health_beauty  →  recruit_sellers  [HIGH]  score: 18.43

  VALIDATION (prev iteration):
    validated: 9   correct: 7   incorrect: 2   accuracy: 77.8%  [GREEN]

─────────────────────────────────────────────────────────────

ITER 4  2017-12  [OK]
  ...
```

**Colour flags (HTML `<span>` with inline `color` style only — no classes):**

| Condition | Colour |
|---|---|
| `[OK]` status | green |
| `[ERROR]` status | red |
| Confidence/feasibility HIGH or STRONG | green |
| Confidence/feasibility MEDIUM or ADEQUATE | orange |
| Confidence/feasibility LOW or WEAK | red |
| Accuracy ≥ 70% | green |
| Accuracy 50–69% | orange |
| Accuracy < 50% | red |
| No validation yet (first iteration) | grey |

**Signal distribution** in the agent row is computed from `agent_signal_summary` in the iteration JSON. Geographic row uses geo_confidence counts. Supply row uses supply_confidence counts. Customer row uses customer_confidence counts. Logistics row uses logistics_confidence counts. Connector row shows `—`.

**Polling:** JavaScript `setInterval` calling `fetch('/data')` every 5000 ms. On success, re-render the full page body. On fetch failure, append `[POLL ERROR]` in red to the header — do not clear existing content.

**Launch:** `python dashboard.py` starts server on `localhost:8765`. Prints `Dashboard: http://localhost:8765` to stdout.

**No dependencies beyond Python stdlib and what is already in `requirements.txt`.**

---

### Acceptance criteria

1. `python dashboard.py` starts without error and prints the URL.
2. Visiting `http://localhost:8765` in a browser renders at least one iteration row when `outputs/iterations/` contains at least one JSON file.
3. With no files in `outputs/iterations/`, the page renders: `No iterations yet.` in grey.
4. Each iteration row shows the correct `[OK]` or `[ERROR]` status with the correct colour.
5. `agent_timings` appears in iteration JSON files produced after this story ships — verified by running `python walk_forward_full.py --n-states 1 --n-categories 1 --iterations 2` and inspecting the output JSON.
6. Existing iteration JSON files without `agent_timings` are rendered gracefully — timing cells show `—` rather than crashing.
7. Validation row shows `No validation` in grey for iteration 1 (no previous month to validate).
8. `python -m pytest tests/` still passes 84/84 after all changes.
9. **Compatibility gate:** `python walk_forward_full.py --n-states 1 --n-categories 1 --iterations 2` completes and produces a valid JSON output.

---

### Out of scope

- Charts of any kind
- Historical trend lines in the dashboard (that is Story 3's job)
- Filtering or sorting iterations
- Authentication
- Any modification to agent logic or memory schema

---

## Story 2 — Memory Visualisation Report

**ID:** VIZ-02  
**File:** `memory_viz.py` (new file at project root)  
**Priority:** P2 — run after a multi-iteration run to inspect learning patterns

---

### User story

As a researcher reviewing what the system learned across 13 iterations,  
I want to open a single HTML file and see, for each state×category pair, how accuracy evolved, which agent was most trusted, and what decisions were made each month,  
so that I can identify which pairs the system calls well and which it consistently gets wrong.

---

### Context

The memory DB (`memory_*.db`) is the single source of truth for the system's learning history. After a full 13-iteration run each state×category×month row contains:

- `geo_directional_accuracy` (1/0/NULL)
- `conn_actual_outcome` (correct/incorrect/unknown/NULL)
- `conn_most_predictive_agent` (geographic/supply_quality/customer_readiness/logistics)
- `conn_decision` (free-text action)
- `conn_confidence` (HIGH/MEDIUM/LOW)
- All agent signal columns

The script is run on-demand by the researcher, not during the live walk-forward. It reads whatever DB is pointed to and writes a static HTML file alongside it.

**Ground-truth note:** `geo_actual_growth` and `geo_directional_accuracy` were NULL in the 2-iteration test DB (ground-truth §6). The script must handle NULL values gracefully throughout — never crash on missing data.

---

### Scope

**CLI:**

```
python memory_viz.py --db memory_2026-04-19-12-19.db
# writes: memory_report_2026-04-19-12-19.html

python memory_viz.py --db memory_2026-04-19-12-19.db --out custom_report.html
```

If `--db` is omitted, the script scans the project root for the most recently modified `memory_*.db` file and uses that.

**HTML output — three sections, no JavaScript, no external resources:**

**Section 1 — Accuracy Trend Table**

One row per state×category pair. Columns are months (sorted ascending). Each cell shows:

```
correct    → green background,  text: ✓
incorrect  → red background,    text: ✗
unknown    → grey background,   text: ?
(NULL)     → white background,  text: —
```

Below the table: summary line `X/Y pairs validated. Overall accuracy: Z%` (counting only correct+incorrect rows, ignoring unknown and NULL).

**Section 2 — Most Predictive Agent Per Pair**

One row per state×category pair. Shows the most frequently cited `conn_most_predictive_agent` across all months for that pair (mode). If all NULL, shows `—`. Colour-code by agent:

| Agent | Colour |
|---|---|
| geographic | blue |
| supply_quality | orange |
| customer_readiness | green |
| logistics | purple |
| — (no data) | grey |

Below the table: bar summary showing count of pairs dominated by each agent. Plain HTML `<div>` bars — character widths only, no SVG, no canvas.

**Section 3 — Decision History Per Pair**

One subsection per state×category pair (H3 heading: `SP × health_beauty`). Inside: a plain table with columns `month | decision | confidence | outcome`. Rows sorted by month ascending. Confidence coloured HIGH=green / MEDIUM=orange / LOW=red. Outcome coloured correct=green / incorrect=red / unknown=grey / —=grey.

**Report header:** shows DB path, total rows, months covered, states, categories, generation timestamp.

**Dependencies:** only `utils/memory.py` (existing), Python stdlib. No new packages.

---

### Acceptance criteria

1. `python memory_viz.py --db memory_2026-04-19-12-19.db` runs without error and produces a `.html` file.
2. The HTML file opens in a browser without JavaScript errors (the file contains no JavaScript).
3. Section 1 renders one row per unique state×category pair found in the DB.
4. Section 1 cells are correctly coloured based on `conn_actual_outcome` value.
5. A DB where `conn_actual_outcome` is entirely NULL (e.g. a 1-iteration run) renders all cells as `—` without crashing.
6. Section 2 shows the correct modal agent per pair (ties broken alphabetically).
7. Section 3 shows one subsection per pair with all months in the DB for that pair.
8. `--out` flag correctly overrides the output filename.
9. Running with no `--db` flag and multiple `memory_*.db` files present picks the most recently modified one and prints which DB was selected to stdout.
10. `python -m pytest tests/` still passes 84/84.
11. **Compatibility gate:** `python walk_forward_full.py --n-states 1 --n-categories 1 --iterations 2` completes without error.

---

### Out of scope

- Live updates or polling (this is a static snapshot tool)
- Reading from `outputs/iterations/` (that is Story 3's job)
- Modifying the memory schema
- Comparison between two DBs
- Any chart, SVG, or canvas element

---

## Story 3 — Performance Visualisation Report

**ID:** VIZ-03  
**File:** `perf_viz.py` (new file at project root)  
**Priority:** P2 — run after a multi-iteration run to review system performance  
**Dependency:** Story 1 (VIZ-01) must ship first — `agent_timings` must be present in iteration JSON files for timing rows to render; the script degrades gracefully when the field is absent

---

### User story

As a developer tuning the walk-forward pipeline,  
I want a single HTML file that shows, per iteration, how long each agent took, how many LLM calls were made, and whether the connector was right or wrong,  
so that I can spot slow iterations, identify which agents consume the most wall time, and see whether accuracy tracks with or against signal quality.

---

### Context

`outputs/iterations/YYYY-MM.json` files are the sole data source. Each file is one iteration. The script aggregates across all files it finds.

**Available in current iteration JSON (no Story 1 dependency):**
- `iteration`, `prediction_month`, `status`, `training_window`
- `validation` — `validated`, `correct`, `incorrect`, `unknown`
- `top_connector_decisions` — `decision`, `confidence`, `composite_score`, `most_predictive_agent`
- `agent_signal_summary` — per-pair signal levels

**Available only after Story 1 ships:**
- `agent_timings` — `wall_seconds` and `llm_calls` per agent

The script must work without `agent_timings` present (graceful degradation: timing rows show `—`). When `agent_timings` is present, full detail renders.

---

### Scope

**CLI:**

```
python perf_viz.py
# reads all outputs/iterations/*.json
# writes: outputs/perf_report.html

python perf_viz.py --dir outputs/iterations --out custom_perf.html
```

**HTML output — four sections, no JavaScript, no external resources:**

**Section 1 — Iteration Summary Table**

One row per iteration JSON file, sorted by `prediction_month` ascending.

Columns: `month | status | wall_time_total | llm_calls_total | validated | correct | incorrect | accuracy`

- `wall_time_total` — sum of all `agent_timings[*].wall_seconds`; shows `—` if `agent_timings` absent
- `llm_calls_total` — sum of all `agent_timings[*].llm_calls`; shows `—` if absent
- `accuracy` — `correct / (correct + incorrect)` as percentage; shows `—` if `validated == 0`
- `status` coloured: `ok` = green, `error` = red
- `accuracy` coloured: ≥ 70% = green, 50–69% = orange, < 50% = red, `—` = grey

**Section 2 — Per-Agent Timing Table** *(renders only when ≥ 1 iteration has `agent_timings`)*

One row per agent. Columns: `agent | avg_seconds | min_seconds | max_seconds | total_llm_calls`

Computed across all iterations that have `agent_timings`. If an agent has `wall_seconds: null` in some iterations (error iterations), those iterations are excluded from that agent's stats.

A plain HTML `<div>` bar chart below the table: one row per agent, bar width proportional to `avg_seconds`, max bar = 100% of column width. No SVG, no canvas. Character widths only.

**Section 3 — Correct / Incorrect Trend**

Plain HTML table: one row per iteration (sorted by month). Columns: `month | correct | incorrect | accuracy | trend`

The `trend` column shows:
- `↑` in green if accuracy is higher than the previous iteration
- `↓` in red if lower
- `→` in grey if equal or no prior iteration

Below the table: `Best iteration: YYYY-MM (X%)` and `Worst validated iteration: YYYY-MM (X%)`. These lines are computed only from iterations where `validated > 0`.

**Section 4 — Top Connector Decisions Across All Iterations**

A flat table of the top 10 decisions by `composite_score` across all iterations. Columns: `rank | month | state | category | decision | confidence | composite_score | most_predictive_agent`.

Confidence coloured HIGH=green / MEDIUM=orange / LOW=red.

**Report header:** shows source directory, files read count, date range covered, generation timestamp.

**Dependencies:** only Python stdlib, no new packages.

---

### Acceptance criteria

1. `python perf_viz.py` runs without error when `outputs/iterations/` contains at least one JSON file.
2. The HTML file opens in a browser without JavaScript errors.
3. Section 1 shows one row per JSON file found, sorted correctly by month.
4. `accuracy` column correctly computes and colour-codes from `validation` data.
5. Section 2 is absent when no iteration JSON contains `agent_timings` (not rendered at all — no empty table).
6. Section 2 renders correctly with accurate avg/min/max stats when `agent_timings` is present (verified by running Story 1's instrumented `walk_forward_full.py` first).
7. Section 3 trend arrows are correct: `↑` when accuracy improves, `↓` when it drops, `→` when equal.
8. Section 3 correctly identifies best and worst validated iteration.
9. Section 4 shows top 10 decisions by composite score with correct ranking.
10. Running against the existing 5 iteration files in `outputs/iterations/` (from ground-truth §6) produces a valid, non-crashing report with Sections 1, 3, 4 populated and Section 2 absent (no `agent_timings` in those files yet).
11. `python -m pytest tests/` still passes 84/84.
12. **Compatibility gate:** `python walk_forward_full.py --n-states 1 --n-categories 1 --iterations 2` completes without error.

---

### Out of scope

- Reading from the memory DB (that is Story 2's job)
- Live updates or polling (static snapshot)
- Filtering by state or category
- Modifying iteration JSON schema beyond what Story 1 already adds
- Any chart, SVG, or canvas element

---

## Story dependency map

```
VIZ-01  (dashboard.py + walk_forward_full.py instrumentation)
   └─► VIZ-03  (perf_viz.py — reads agent_timings emitted by VIZ-01)

VIZ-02  (memory_viz.py)
   └─► independent — reads memory DB only
```

Recommended implementation order: VIZ-01 → VIZ-02 → VIZ-03.  
VIZ-02 may be built in parallel with VIZ-01 since it has no dependency.

---

## Data source reference

| Data | Location | Available now |
|---|---|---|
| Iteration status, validation, decisions | `outputs/iterations/YYYY-MM.json` | ✅ |
| Agent signal distribution | `outputs/iterations/YYYY-MM.json` → `agent_signal_summary` | ✅ |
| Per-agent wall time and LLM call count | `outputs/iterations/YYYY-MM.json` → `agent_timings` | ❌ after VIZ-01 |
| Accuracy history per pair | `memory_*.db` → `conn_actual_outcome` | ✅ (after ≥ 2 iterations) |
| Most predictive agent per pair | `memory_*.db` → `conn_most_predictive_agent` | ✅ |
| Full decision history per pair | `memory_*.db` → `conn_decision`, `conn_confidence` | ✅ |
