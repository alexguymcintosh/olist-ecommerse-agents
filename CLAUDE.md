# Olist AI Club — Geographic Demand Agent System

## Project
Multi-agent e-commerce intelligence system built on the Olist Brazilian dataset (100k orders, 2016-2018). Teaching vehicle for agentic problem-solving skills transferable to any industry.

## Problem Statement
Predict which product category will grow in which Brazilian state next month, identify the seller supply gap, and recommend what Olist should do about it.

## Focus Parameters (change these to adjust analysis scope)
FOCUS_STATES = ["SP", "RJ", "MG", "RS", "PR"]  # top 5 by order volume
FOCUS_CATEGORIES = ["health_beauty", "bed_bath_table", "sports_leisure", "watches_gifts", "computers_accessories"]  # top 5 by revenue

## Architecture — Current MVP (v1)
Four domain agents producing structured JSON outputs:
- CustomerAgent: repeat rate, delivery, reviews, payments
- ProductAgent: category revenue, volume, avg order value
- SellerAgent: seller count, SP concentration, Pareto concentration
- ConnectorAgent: cross-domain synthesis and briefing

Status: complete, tested, run_all.py working end to end.

## Architecture — Next MVP (v2 Geographic Demand Agent)
Unit of analysis: state × category × month (25 pairs: 5 states × 5 categories)
Prediction: which category grows in which state next month
Walk-forward loop: run month 1→2, store result, run month 2→3 with memory, repeat across 25 months
Memory: SQLite — stores prediction, actual, error, reasoning per month per state-category pair
Build order:
1. feature_table.py — state × category × month aggregations from full dataset
2. seller_gap.py — demand vs seller supply per state-category
3. GeographicAgent — wraps feature table + seller gap, produces structured JSON
4. memory.py — SQLite read/write for prediction history
5. walk_forward.py — loop orchestrator, runs GeographicAgent month by month
6. Connector ranking — ranks opportunities by gap size × revenue potential

No ML yet. Pure pandas + LLM reasoning + memory loop.

## Tech Stack
- Language: Python 3.12
- LLM: MiniMax M2.7 via Ollama
  - ANTHROPIC_AUTH_TOKEN=ollama
  - ANTHROPIC_BASE_URL=http://localhost:11434
  - ANTHROPIC_MODEL=minimax-m2.7:cloud
- Claude Code + G-Stack: architectural decisions (/plan-ceo-review, /plan-eng-review, /engineer, /qa, /ship)
- BMAD in Cursor: story-by-story implementation
- Codex: second opinion and boilerplate
- Cursor: implementation
- Data: utils/data_loader.py — always sample before sending to LLM
- All LLM calls: utils/openrouter_client.py — never hardcode API calls

## Hard Rules
1. Pandas computes ALL metrics from full DataFrame — LLM only narrates pre-computed facts
2. All LLM calls go through utils/openrouter_client.py
3. Focus parameters live in config.py — change one file, entire analysis shifts
4. Each agent produces structured JSON consumed by the next layer
5. Tests required before any agent marked complete
6. Use datetime.now(timezone.utc) — never utcnow()

## Key Data Facts
- 96k customers, 3095 sellers, 32951 products, 73 categories, 25 months (Sep 2016 - Oct 2018)
- $16M revenue, avg order $161
- 0% repeat customer rate — biggest business problem
- Delivery speed = #1 review driver (4.4 stars <1wk vs 2.2 stars >4wk)
- 60% sellers in SP, customers nationwide — core geographic mismatch
- Top revenue: health_beauty $1.26M, watches_gifts $1.2M

## Current Status
- [x] EDA complete
- [x] OpenRouter connected via n8n
- [x] v1 MVP complete — all 4 agents running, tests passing, run_all.py working
- [x] Dashboard visualisation (outputs/dashboard.html)
- [ ] config.py with focus parameters
- [ ] feature_table.py
- [ ] seller_gap.py
- [ ] GeographicAgent
- [ ] memory.py (SQLite)
- [ ] walk_forward.py
- [ ] Connector ranking

## Repo
github.com/alexguymcintosh/olist-ecommerse-agents

## Team
AI Club — transparent development in Discord. All work shared. Everyone maintains full system.
