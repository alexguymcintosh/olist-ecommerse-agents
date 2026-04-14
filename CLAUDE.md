# Olist AI Club — Agent System

## Project
Multi-agent e-commerce analysis system built on the Olist Brazilian dataset (100k orders, 2016-2018). Four agents: Customer, Product, Seller, Connector. Teaching vehicle for agentic problem-solving skills transferable to any industry.

## Mission
Build domain-expert agents that independently analyse their data slice, then discover how to connect them into a Connector Agent that surfaces cross-domain business insights.

## Architecture
- Customer Agent: demand signal (orders, customers, reviews, payments)
- Product Agent: catalog intelligence (products, categories, order_items)
- Seller Agent: supply performance (sellers, order_items, reviews)
- Connector Agent: NOT YET DEFINED — emerges from domain agent outputs

## Stack
- Language: Python 3.9+
- LLM calls: utils/openrouter_client.py (single entry point, swap model for RnD vs prod)
- RnD model: deepseek/deepseek-v3.2 via OpenRouter (cheap, fast)
- Prod model: anthropic/claude-sonnet-4-5 via OpenRouter
- Data: utils/data_loader.py (always sample before sending to LLM)
- Visual flows: n8n (separate, mirrors Python agents)
- Planning: BMAD docs in /docs/bmad
- Workflow: G-Stack slash commands for role-based development

## Key Data Facts
- 96k customers, 3095 sellers, 32951 products, 73 categories
- $16M total revenue, avg order $161
- 0% repeat customer rate — biggest business problem
- Delivery speed is #1 review score driver (4.4 stars <1wk vs 2.2 stars >4wk)
- 60% of sellers concentrated in São Paulo state
- Top revenue: health_beauty ($1.26M), watches_gifts ($1.2M)

## Development Rules
1. Always sample data before sending to LLM (max 100 rows for analysis, 50 for LLM context)
2. All LLM calls go through utils/openrouter_client.py — never hardcode API calls in agents
3. Use RND_MODEL for exploration, PROD_MODEL only for final agent builds
4. Each agent must produce a structured JSON output that the Connector Agent can consume
5. Write tests in /tests before marking any agent as complete
6. Every student maintains a full working copy of all 4 agents

## G-Stack Usage
- /plan-ceo-review — use when defining what an agent should actually do
- /plan-eng-review — use when designing agent data flow and output schema
- /engineer — use when implementing agent logic
- /qa — use when testing agent outputs
- /ship — use when committing completed agent work

## Current Status
- [x] Data loaded and EDA complete
- [x] OpenRouter connected via n8n (tested)
- [x] Project structure scaffolded
- [ ] Customer Agent MVP
- [ ] Product Agent MVP
- [ ] Seller Agent MVP
- [ ] Connector Agent (defined after domain MVPs)

## Team
AI Club — transparent development in Discord. All work shared. Everyone maintains full system.
