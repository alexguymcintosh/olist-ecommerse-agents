# Olist AI Club — Geographic Demand Intelligence System

## Project
Multi-agent e-commerce intelligence system built on the Olist Brazilian dataset (100k orders, 2016-2018). Teaching vehicle for agentic problem-solving skills transferable to any industry. The system predicts demand, validates supply, and recommends actions — learning from its own mistakes month by month.

## Problem Statement
For each state × category × month: predict demand growth, identify seller supply gaps, validate with buyer and logistics signals, and recommend the single best action Olist should take.

## Focus Parameters
FOCUS_STATES = ["SP", "RJ", "MG", "RS", "PR"]
FOCUS_CATEGORIES = ["health_beauty", "bed_bath_table", "sports_leisure", "watches_gifts", "computers_accessories"]
MIN_MONTHLY_ORDERS = 10
TRAINING_WINDOW_MONTHS = 12
MAX_ITERATIONS = 13

## The 5-Agent Architecture

### Unit of Analysis
state × category × month — every agent, every metric, every memory entry is anchored to this triplet.

### Shared Memory
Single SQLite table agent_memory with primary key (state, category, month). Each agent writes to its own columns. Connector reads the full row.

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
    log_reasoning TEXT,
    conn_decision TEXT, conn_confidence TEXT,
    conn_reasoning TEXT, conn_actual_outcome TEXT,
    conn_most_predictive_agent TEXT,
    PRIMARY KEY (state, category, month)
);
```

## Agent 1 — Geographic Agent — COMPLETE

**Purpose:** Demand signal. Predicts which state-category pairs will grow next month.

**Input:** orders + customers (customer_state) + order_items + products + categories + sellers (seller_state)

**Pandas computes:** Monthly revenue per state × category. MoM growth rate with inf guard and missing period reindex. Momentum score (slope of last 3 MoM rates). Supply gap ratio = predicted_order_volume / max(current_sellers, 1). Confidence: HIGH/MEDIUM/LOW based on order count and momentum consistency.

**LLM question:** Given pre-computed momentum and supply gap, predict next month growth % and reasoning.

**Output:** GeographicOutput TypedDict — predictions, supply_gaps, ranked_opportunities

**Memory written:** geo_predicted_growth, geo_actual_growth, geo_directional_accuracy, geo_confidence, geo_confidence_score, geo_reasoning

**Walk-forward:** 13 iterations, Sep 2017 → Aug 2018. Previous prediction fed into next LLM call for temporal coherence.

## Agent 2 — Supply Quality Agent

**Purpose:** Seller health signal. Validates whether sellers in a state-category can fulfil predicted demand.

**Input:** sellers + order_items + reviews + orders

**Pandas computes:** Seller count per seller_state × category. Avg review score weighted last 3 months. Avg delivery days per state × category. Churn risk: sellers active 6 months ago but not last month / total sellers. Top seller id by order volume. Seller concentration: revenue share of top seller.

**LLM question:** Given seller count, review scores, delivery performance, and churn risk — assess supply confidence for this state-category.

**Output schema:**
- agent, timestamp, state, category, month
- seller_count, avg_review_score, avg_delivery_days
- churn_risk (HIGH/MEDIUM/LOW), churn_rate, top_seller_id
- seller_concentration, supply_confidence (STRONG/ADEQUATE/WEAK)
- reasoning, risk_flags

**Memory written:** sq_seller_count, sq_avg_review, sq_avg_delivery_days, sq_churn_risk, sq_top_seller_id, sq_reasoning

**Tuning target:** If geo predicts HIGH growth but supply_confidence = WEAK — connector flags high-risk opportunity. Seller count of 0 = critical gap hard flag.

## Agent 3 — Customer Readiness Agent

**Purpose:** Buyer signal. Validates whether customers in a state are ready to spend in the predicted category.

**Input:** orders + customers (customer_state) + order_items + products + categories + payments

**Pandas computes:** Avg payment value per state last 3 months. Order volume trend per state MoM. Top payment type per state. High-value customer count: customers with avg order above dataset median. Repeat rate per state. Installment behaviour: % using 6+ installments.

**LLM question:** Given spend trends, payment behaviour, and repeat rate — assess whether customers in this state are ready to grow spending in this category next month.

**Output schema:**
- agent, timestamp, state, category, month
- avg_spend, order_volume_trend, top_payment_type
- high_value_customer_count, repeat_rate, installment_pct
- readiness (HIGH/MEDIUM/LOW), reasoning, risk_flags

**Memory written:** cr_avg_spend, cr_order_volume_trend, cr_top_payment_type, cr_high_value_customer_count, cr_repeat_rate, cr_reasoning

**Tuning target:** If geo predicts growth but readiness = LOW — connector recommends marketing spend not seller recruitment.

## Agent 4 — Logistics Agent

**Purpose:** Delivery feasibility signal. Validates whether the supply chain can support predicted demand growth.

**Input:** orders + order_items + sellers (seller_state) + customers (customer_state)

**Pandas computes:** Avg delivery days by seller_state → customer_state route. % on-time per state × category. Freight cost ratio: avg freight / avg price. Fastest seller cluster: seller_state with lowest avg delivery to this customer_state. Delivery variance: std dev of delivery days. Cross-state dependency: % of orders served by out-of-state sellers.

**LLM question:** Given delivery performance, freight costs, and cross-state dependency — assess logistics feasibility and identify the fastest fulfilment path.

**Output schema:**
- agent, timestamp, state, category, month
- avg_delivery_days, pct_on_time, freight_ratio
- fastest_seller_state, delivery_variance, cross_state_dependency
- feasibility (STRONG/ADEQUATE/WEAK), reasoning, risk_flags

**Memory written:** log_avg_delivery_days, log_pct_on_time, log_freight_ratio, log_fastest_seller_state, log_reasoning

**Tuning target:** If geo predicts RJ growth but cross_state_dependency = 0.95 — connector recommends recruiting local RJ sellers not increasing SP inventory.

## Agent 5 — Connector Agent

**Purpose:** Decision maker. Reads all 4 agent outputs for same state × category × month. Makes one ranked decision. Stores reasoning. Learns from previous month's outcome.

**Input:** All 4 agent outputs + previous month's connector memory row

**Does NOT subclass any base class. Standalone interface.**

**Composite score (pandas computes):**
composite = (geo_confidence_score × 0.35 + supply_quality_score × 0.25 + customer_readiness_score × 0.20 + logistics_score × 0.20) × geo_predicted_growth

**LLM question:** Given all 4 agent signals and last month's outcome — what is the single best action for Olist in this state-category next month? Challenge your own recommendation: what could go wrong?

**Back and forth loop:** Connector can send one follow-up question to one agent before finalising. Agent re-runs focused analysis on that subset. Connector gets refined signal before deciding.

**Output schema:**
- agent, timestamp, month
- decisions: list of ConnectorDecision ranked by composite_score
- briefing: Rich terminal narrative
- follow_up_used, follow_up_agent, follow_up_question, follow_up_response

**ConnectorDecision fields:** state, category, month, composite_score, decision, confidence, urgency, reasoning, challenge, most_predictive_agent, risk_flags

**Memory written:** conn_decision, conn_confidence, conn_reasoning, conn_actual_outcome, conn_most_predictive_agent

**Learning loop:** Each month connector reads previous month's outcome and most_predictive_agent before deciding. Adjusts composite weights if one agent is consistently more accurate.

## Full System Flow Per Iteration

Month N arrives → all 4 domain agents run in parallel on training window → each agent reads its own memory columns → each agent writes new signal to memory table → connector reads full memory row for each state × category → connector reads previous month's outcome → optional follow-up to one agent → connector makes ranked decisions → connector writes to memory → month N+1 validates predictions against actuals → accuracy scores written back → loop continues.

## Directory Structure

agents/geographic/geographic_agent.py — complete
agents/supply_quality/supply_quality_agent.py
agents/customer_ready/customer_ready_agent.py
agents/logistics/logistics_agent.py
agents/connector/connector_agent.py
utils/data_loader.py — complete
utils/openrouter_client.py — complete
utils/schema_geographic.py — complete
utils/schema_agents.py — new shared schemas for agents 2-5
utils/memory.py — SQLite read/write for all agents
utils/config.py — focus parameters
walk_forward.py — complete
tests/geographic/ — complete
tests/supply_quality/
tests/customer_ready/
tests/logistics/
tests/connector/
tests/integration/
docs/specs/ — history
docs/bmad/ — history
docs/plans/ — new agent plans
outputs/

## Tech Stack
- Language: Python 3.12
- LLM: OpenRouter API via utils/openrouter_client.py
- RnD model: deepseek/deepseek-v3.2
- Production model: anthropic/claude-sonnet-4-5
- Claude Code + G-Stack: architectural decisions
- BMAD in Cursor: story-by-story implementation
- Cursor + Opus: parallel agent builds
- Memory: SQLite via utils/memory.py
- Tests: pytest, separate folder per agent

## Hard Rules
1. Pandas computes ALL metrics — LLM only narrates pre-computed facts
2. All LLM calls go through utils/openrouter_client.py
3. Focus parameters in config.py — one file change shifts entire analysis
4. Each agent writes to its own columns in shared SQLite memory table
5. Never hardcode state or category names in agent logic — always read from config
6. Tests in separate folder per agent — never mix test files across agents
7. Use datetime.now(timezone.utc) — never utcnow()
8. Connector never subclasses any base class
9. All agents run independently — connector is the only one that reads others
10. seller_state = supply side, customer_state = demand side — never mix
11. Categories CSV requires encoding=utf-8-sig
12. Null ratio for product_category_name_english: allow up to 3%

## Key Data Facts
- 96k customers, 3095 sellers, 32951 products, 73 categories
- 25 months: Sep 2016 – Oct 2018, 13 usable walk-forward iterations
- Nov 2016 missing from dataset — period reindex required before pct_change
- 16.8% of state × category × month combos have zero orders — inf guard required
- 0% repeat customer rate — biggest business problem
- Delivery speed = number 1 review driver (4.4 stars under 1 week vs 2.2 stars over 4 weeks)
- 60% sellers in SP, customers nationwide — core geographic mismatch
- Top revenue: health_beauty $1.26M, watches_gifts $1.2M

## Current Status
- [x] Geographic Agent complete and tested
- [x] walk_forward.py complete and tested (13 iterations)
- [x] Shared memory schema designed
- [x] Directory structure defined
- [x] utils/config.py
- [x] utils/memory.py
- [x] utils/schema_agents.py
- [x] Supply Quality Agent
- [x] Customer Readiness Agent
- [x] Logistics Agent
- [x] Connector Agent
- [x] walk_forward_full.py — 5-agent orchestrator (not in original directory spec)
- [x] dashboard.py — real-time ops dashboard, port 5001 (VIZ-01)
- [x] memory_viz.py — memory visualisation report (VIZ-02)
- [x] perf_viz.py — performance visualisation report (VIZ-03)
- [x] All 84 tests passing
- [ ] Integration test — all 5 agents × 13 iterations (only 5 iterations run to date)

## Repo
github.com/alexguymcintosh/olist-ecommerse-agents

## Team
AI Club — transparent development in Discord. All work shared. Everyone maintains full system.