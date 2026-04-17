# Engineering Review: Geographic Demand Agent v2

Generated: 2026-04-17  
Reviewer: Claude Code (plan-ceo-review skill, engineering pass)  
Source plan: `docs/specs/geographic-agent-ceo-plan.md`  
Data verified against: actual Olist CSVs in `data/`

---

## Summary

Five areas reviewed with real data queries. Three are blockers — they will produce
wrong outputs silently with no exception raised. Two are significant gaps that will
surface as confusing bugs during testing.

```
CRITICAL (wrong output, no error)  : 3
SIGNIFICANT (bug during testing)   : 4
SCHEMA GAP (missing field)         : 3
TEST GAPS (will break in prod)     : 6
```

---

## 1. Data Join Correctness

### 1A. CRITICAL — `customer_state` vs `seller_state` for supply gap is undefined

The plan's supply gap uses "current_sellers" for a (state × category) pair but never
specifies which state dimension to use.

Verified from actual data for health_beauty × RJ:

```
Sellers IN RJ (seller_state='RJ')  who sell health_beauty:  42 sellers
Sellers SERVING RJ customers (any seller_state): 219 sellers
```

These differ by 5x. Using seller_state gives 42. Using customer_state as the filter
gives 219. The plan never says which.

The correct definition for this project's business question (geographic mismatch between
supply and demand) is **sellers IN the state** (`seller_state`), not sellers who happen
to have fulfilled an order there. Using the wrong one would show "RJ is well-served" when
in fact it's being served by SP sellers from across the country.

**Required fix:** Add to the plan's Key Design Decisions section:

> **Q: Which 'state' do we use for seller supply?**  
> A: `seller_state` from the sellers table. This counts sellers physically located in
> that state. Do NOT filter by customer_state on the seller join — that counts any
> seller who ever shipped to that state, which is the wrong unit for gap analysis.

**Join chain for `_load_geographic_data()`:**
```
orders
  JOIN customers ON customer_id          → customer_state (demand side)
  JOIN order_items ON order_id           → price, product_id, seller_id
  JOIN products ON product_id            → product_category_name
  JOIN categories ON product_category_name → product_category_name_english
  JOIN sellers ON seller_id              → seller_state (supply side, for gap calc)
```

Supply gap `current_sellers` = `seller_id.nunique()` WHERE `seller_state = focus_state`.
This is a SEPARATE query from the revenue/growth query (which uses `customer_state`).

### 1B. WARNING — 1,627 rows have null English category after join (1.4% data loss)

```python
products['product_category_name'].isna() → 610 products (no category at all)
After join with categories table → 1,627 null product_category_name_english rows
```

The category translation CSV has a BOM character (`\ufeff`) on the first column name.
Depending on how `pd.read_csv()` is called, this may cause the join to silently fail
for the first column.

**Required fix:** Load categories with `encoding='utf-8-sig'` to strip the BOM:
```python
cats = pd.read_csv(DATA_DIR / 'product_category_name_translation.csv', encoding='utf-8-sig')
```

Add to `_load_geographic_data()` acceptance criteria: "assert NaN category rows < 1%
after join (should be ~0.6% from products with no category — not 1.4%)."

### 1C. OK — Focus category names match exactly

All five focus categories exist verbatim in `product_category_name_english`:
`health_beauty`, `bed_bath_table`, `sports_leisure`, `watches_gifts`, `computers_accessories`. No mapping needed.

---

## 2. Edge Cases in `_compute_growth_matrix()`

### 2A. CRITICAL — November 2016 is missing from the dataset

The dataset jumps directly from October 2016 to December 2016. November 2016 has zero
orders. Verified:

```
2016-09:    4 orders
2016-10:  324 orders
[2016-11: MISSING]
2016-12:    1 order
2017-01:  800 orders
```

Impact on `pct_change()` or any MoM calculation:

If `_compute_growth_matrix()` calls `.pct_change()` on a Period-indexed Series, pandas
compares the adjacent rows in the DataFrame regardless of the actual time gap. So
Oct 2016 → Dec 2016 looks like a 1-month comparison but is actually 2 months. For the
health_beauty × SP series, this produces a growth rate of ~+267% for that step where
the real MoM rate should be roughly half that.

**Required fix:** After building the monthly revenue Series, check for period continuity
before computing growth rates:

```python
def _check_period_continuity(series: pd.Series) -> list[str]:
    """Returns list of missing months in a Period-indexed Series."""
    if len(series) < 2:
        return []
    full_range = pd.period_range(series.index.min(), series.index.max(), freq='M')
    missing = [str(p) for p in full_range if p not in series.index]
    return missing

# In _compute_growth_matrix():
missing = _check_period_continuity(monthly_revenue)
if missing:
    # Option A: fill with 0 and mark confidence=LOW for adjacent months
    monthly_revenue = monthly_revenue.reindex(
        pd.period_range(monthly_revenue.index.min(), monthly_revenue.index.max(), freq='M'),
        fill_value=0.0
    )
    # Option B: exclude the anomalous growth step from momentum calculation
```

This affects every training window that includes Oct-Dec 2016 (iterations 1-3).

### 2B. CRITICAL — 105 of 625 (state×category×month) combos have zero orders

```
Missing combos: 105 / 625 = 16.8% of the grid
Combos with < 5 orders: 33
```

`pct_change()` behavior:
- `0 → N`: produces `inf` (divide by zero)
- `N → 0`: produces `-1.0` (-100% growth, which looks valid but means "no orders that month")
- `0 → 0`: produces `NaN`

Any `inf` or `NaN` in the growth matrix will silently propagate through the momentum
calculation and supply gap formula without raising an exception. The LLM prompt will
receive an `inf` value as a number in its context, which is undefined behavior.

**Required fix:**
```python
# In _compute_growth_matrix(), after pct_change():
growth = monthly_revenue.pct_change()
growth = growth.replace([np.inf, -np.inf], np.nan)

# Flag cells with insufficient data
def _growth_is_reliable(monthly_revenue: pd.Series, min_orders: int = 5) -> bool:
    return (monthly_revenue > 0).sum() >= 3 and monthly_revenue.max() >= min_orders
```

Add to acceptance criteria: "given a (state×category) with 0 orders in 3 of 12 training
months, `_compute_growth_matrix()` returns `NaN` growth (not `inf`), and `_score_confidence()`
returns `LOW`."

### 2C. WARNING — Early months (Sep-Dec 2016) are pre-scale noise, not signal

```
Sep 2016: 4 total orders (entire dataset)
Oct 2016: 324 orders
Dec 2016: 1 order
Jan 2017: 800 orders
```

Any training window including these months will see growth rates of +8000% (Sep→Oct)
and -99.7% (Oct→Dec) that have nothing to do with actual demand trends. Including
this period inflates early momentum scores and corrupts the first 3 iterations.

**Required fix (choose one):** Add to Key Design Decisions:

> Option A: Exclude months with total dataset orders < 500 from training windows.
> Hard-codes Sep-Dec 2016 as warmup/excluded.
>
> Option B: Use a min_monthly_orders guard: any (state×category×month) with < 10
> orders gets NaN growth for that step (already handled by 2B fix above).

Option B is simpler and generalizes. Option A is more principled but requires a
dataset-level parameter.

---

## 3. Walk-Forward Loop Correctness

### 3A. CRITICAL — The plan says 25 iterations but there are only 13

```
Total months in dataset: 25
Months needed for initial training window: 12
Usable prediction months: 25 - 12 = 13 (not 25)
```

The plan states: "Iterations 1-25 map to months 13-37 (Jan 2017 – Dec 2018)."

This is wrong on two levels:

**Level 1 — wrong month count.** You cannot get 25 predictions from a 25-month dataset
with a 12-month rolling window. You get 13.

**Level 2 — wrong month labels.** The plan says "month 13 = Jan 2017" which implies the
dataset starts Jan 2016. The dataset starts Sep 2016. Actual month 13 = **Oct 2017**.

Correct walk-forward mapping (actual):
```
Iteration  1: training 2016-09 to 2017-09 (11 actual months, Nov missing), predict 2017-10
Iteration  2: training 2016-10 to 2017-10, predict 2017-11
...
Iteration 13: training 2017-09 to 2018-08, predict 2018-09
```

Iteration 13 predicts Sep 2018, which has only 1 order total — effectively corrupt data.
Usable iterations: 12 (iterations 1-12, predicting Oct 2017 through Aug 2018).

**Required fix:** Update the plan's walk_forward.py section:

```python
# Replace the current loop spec:
# for iteration T from 13 to 37:

# With:
months = sorted(df['month'].unique())  # 25 calendar months
usable_months = [m for m in months if monthly_order_count[m] >= MIN_ORDERS]  # exclude sparse ends
# With 12-month window, iterations = len(usable_months) - 12
# Expected: ~12 usable iterations, predicting Oct 2017 through Aug 2018
```

Update the acceptance criterion: "walk_forward.py --start 1 --end 3 runs 3 iterations" is
correct, but the integration test assertion needs changing from `completed_iterations == 25`
to `completed_iterations == 12` (or whatever the actual count is after sparse-month filtering).

### 3B. WARNING — `temporal_window()` using calendar range includes the Nov 2016 gap

If `temporal_window(df, '2016-09', '2017-08')` is implemented as:
```python
df[df['month'].between(start, end)]
```
...it returns 11 months of data (no Nov 2016) while the caller expects 12. The training
window silently has 11 months, not 12. The momentum calculation (slope of last 3 MoM
growth rates) may return only 2 rates for the first iteration if the missing month falls
in the last 3 months.

**Required fix:** `temporal_window()` should return the filtered DataFrame AND a `month_count`
so the caller can assert it got the expected number of months:

```python
def temporal_window(df: pd.DataFrame, start_month: str, end_month: str
                    ) -> tuple[pd.DataFrame, int]:
    """Returns (filtered_df, distinct_month_count). Month count may be less than
    calendar range if data has gaps."""
    mask = (df['month'] >= pd.Period(start_month, 'M')) & \
           (df['month'] <= pd.Period(end_month, 'M'))
    filtered = df[mask]
    month_count = filtered['month'].nunique()
    return filtered, month_count
```

---

## 4. Schema Completeness

### 4A. `WalkForwardIteration` is not in `schema_geographic.py`

The TypedDict is defined inline in the walk_forward.py section of the plan but is not
listed in the `utils/schema_geographic.py` build step. Add it to Step 1 alongside the
other TypedDicts. It needs an import in `walk_forward.py` from `utils.schema_geographic`.

### 4B. `GeographicMetrics.training_end_month` captures half the window

The plan has `training_end_month: str` in `GeographicOutput` (and in `WalkForwardIteration`).
This captures when the training window ends but not when it starts. Debugging a bad
prediction in iteration 3 requires knowing both bounds.

**Required fix:** Change to `training_window: tuple[str, str]` (start_month, end_month).
Or add `training_start_month: str` as a separate field. Either works; pick one and be
consistent across `GeographicOutput` and `WalkForwardIteration`.

### 4C. `Prediction.confidence` is a string but composite score formula needs a number

The plan's `OpportunityRanker` computes:
```
composite_score = predicted_growth × confidence × supply_gap_severity
```

`confidence` is typed as `str` ("HIGH" / "MEDIUM" / "LOW"). The multiplication will
fail at runtime with `TypeError: can't multiply sequence by non-int of type 'float'`.

**Required fix:** Add `confidence_score: float` to the `Prediction` TypedDict alongside
`confidence: str`. The mapping (HIGH=1.0, MEDIUM=0.6, LOW=0.3 — or whatever scale
is chosen) belongs in `_score_confidence()`. The composite score formula uses
`confidence_score`, not `confidence`.

---

## 5. Test Coverage Gaps

### 5A. No test for missing month in training window

**Test needed (unit):**
```python
def test_compute_growth_matrix_handles_missing_month():
    """Nov 2016 is missing from dataset. Growth calc must not produce inf or skip silently."""
    # Given: monthly revenue series with Oct 2016 = 100, Dec 2016 = 150 (Nov missing)
    # When: _compute_growth_matrix() runs on this window
    # Then: growth for the Oct→Dec step is NaN (not inf, not +50%)
    # Then: confidence for that (state×category) is LOW
```

### 5B. No test for zero-order month producing inf growth

**Test needed (unit):**
```python
def test_compute_growth_matrix_zero_order_month():
    """pct_change() on 0 → N produces inf. Must be caught."""
    # Given: monthly revenue = [0, 0, 500, 600, 0, 700]
    # When: _compute_growth_matrix() runs
    # Then: no inf values in output
    # Then: months with 0 orders produce NaN growth, not inf
```

### 5C. No test asserting correct iteration count

**Test needed (integration):**
```python
def test_walk_forward_iteration_count():
    """25 months - 12 for training window = 13 max iterations. Sparse ends excluded."""
    result = run_walk_forward(df)
    assert result['completed_iterations'] == 12  # or whatever the exact count is
    assert result['completed_iterations'] != 25  # guard against the plan's wrong assertion
```

### 5D. No test for `customer_state` vs `seller_state` join

**Test needed (unit):**
```python
def test_load_geographic_data_uses_correct_state_dimensions():
    """Revenue/growth uses customer_state. Seller supply uses seller_state."""
    data = geographic_agent._load_geographic_data()
    # health_beauty × RJ demand comes from customer_state='RJ' rows
    # health_beauty × RJ supply counts seller_state='RJ' unique sellers (should be 42)
    # NOT sellers who shipped to RJ (that would be 219)
    rj_sellers = geographic_agent._get_seller_count('RJ', 'health_beauty', data)
    assert rj_sellers == 42  # not 219
```

### 5E. No test for BOM-corrupted category join

**Test needed (unit):**
```python
def test_category_join_no_bom_corruption():
    """Categories CSV has BOM char on first column. Must use utf-8-sig encoding."""
    data = data_loader.load_all()
    # After full join, null english category should be < 1% (from products with no category)
    # If BOM corrupts the join key, ALL rows would have null english category
    null_pct = df['product_category_name_english'].isna().mean()
    assert null_pct < 0.01
```

### 5F. No test for confidence_score numeric type before composite score multiply

**Test needed (unit):**
```python
def test_opportunity_ranker_composite_score_numeric():
    """confidence is str, confidence_score must be float. TypeError if wrong field used."""
    prediction = Prediction(
        state='SP', category='health_beauty',
        predicted_growth_pct=0.12,
        confidence='HIGH',
        confidence_score=1.0,
        reasoning='...'
    )
    ranker = OpportunityRanker()
    score = ranker._composite_score(prediction, supply_gap_severity=2.5)
    assert isinstance(score, float)
    assert score > 0
```

---

## Confirmed Correct

- Focus category names match CSV exactly (no mapping needed)
- Join keys are unambiguous: `customer_id` links orders↔customers, `order_id` links orders↔items, `product_id` links items↔products, `seller_id` links items↔sellers
- `customer_state` from the customers table is the right demand-side state (0 NaN values)
- `price` column in order_items is the revenue signal (not `freight_value`)
- The 5 focus states (SP, RJ, MG, RS, PR) are the correct top-5 by customer order volume

---

## Changes Required in the Plan

```
1. [CRITICAL] Update walk-forward iteration count: 13 usable (not 25). Update
   month-to-date mapping. Remove "Iterations 1-25 map to months 13-37 (Jan 2017 – Dec 2018)."
   Replace with: "Iterations 1-12, predicting Oct 2017 through Aug 2018."
   Integration test assertion: completed_iterations == 12, not == 25.

2. [CRITICAL] Add to Key Design Decisions: seller_state (not customer_state)
   defines supply-side seller count for gap calculation. Add test 5D above.

3. [CRITICAL] Add to Step 2 acceptance criteria: given zero-order month in
   training window, growth = NaN (not inf). Add pct_change() inf guard to spec.

4. [SIGNIFICANT] Add to _load_geographic_data() spec: load categories CSV with
   encoding='utf-8-sig' to strip BOM. Add assertion: null category < 1% post-join.

5. [SIGNIFICANT] Add to temporal_window() spec: return (df, month_count) tuple.
   Caller asserts month_count == expected_months (catches Nov 2016 gap silently
   giving 11 months instead of 12).

6. [SIGNIFICANT] Add to _compute_growth_matrix() spec: check period continuity
   before pct_change(). Fill missing periods with 0, mark those months LOW confidence.

7. [SIGNIFICANT] Add to Key Design Decisions: early months (Sep-Dec 2016) have
   <5 orders per category/state. min_monthly_orders guard (suggest 10) sets growth=NaN.

8. [SCHEMA] Add WalkForwardIteration to schema_geographic.py Step 1 build list.

9. [SCHEMA] Change training_end_month: str to training_window: tuple[str, str]
   in GeographicOutput and WalkForwardIteration.

10. [SCHEMA] Add confidence_score: float to Prediction TypedDict alongside
    confidence: str. Update composite score formula to use confidence_score.

11. [TESTS] Add tests 5A-5F above to Step 2 and Step 3 acceptance criteria.
    Current acceptance criteria ("mock data, mock LLM, no error") will pass
    with all these bugs present.
```
