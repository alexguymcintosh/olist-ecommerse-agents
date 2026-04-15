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

## Build Order (from CEO/Eng plan)
1. utils/schema.py — TypedDicts, zero runtime deps, build FIRST
2. utils/validators.py — validate_output()
3. utils/base_agent.py — BaseAgent with shared run() logic
4. Domain agents: Customer → Product → Seller (each with tests)
5. run_all.py — orchestrator
6. ConnectorAgent — last, after all domain agents pass tests

## Stack
- Language: Python 3.9+
- LLM calls: utils/openrouter_client.py (single entry point, swap model for RnD vs prod)
- RnD model: deepseek/deepseek-v3.2 via OpenRouter (cheap, fast)
- Prod model: anthropic/claude-sonnet-4-5 via OpenRouter
- Data: utils/data_loader.py (always sample before sending to LLM)
- Visual flows: n8n (separate, mirrors Python agents)
- Planning: BMAD docs in /docs/bmad, specs in /docs/specs
- Workflow: G-Stack slash commands for role-based development
- Testing: pytest with conftest.py fixtures + pytest.ini

## Hard Rules (non-negotiable)
1. Pandas computes ALL numeric metrics from the FULL DataFrame — LLM only narrates pre-computed facts
2. All LLM calls go through utils/openrouter_client.py — never hardcode API calls in agents
3. Use RND_MODEL for exploration, PROD_MODEL only for final agent builds
4. Each agent produces a structured DomainAgentOutput (see schema.py) for the Connector to consume
5. Write tests before marking any agent complete — pytest must be green
6. Every student maintains a full working copy of all 4 agents
7. Use datetime.now(timezone.utc) — never datetime.utcnow() (deprecated Python 3.12+)
8. ConnectorAgent does NOT subclass BaseAgent — different interface
9. Add __init__.py to all agent and utils folders

## Key Data Facts
- 96k customers, 3095 sellers, 32951 products, 73 categories
- $16M total revenue, avg order $161
- 0% repeat customer rate — biggest business problem
- Delivery speed is #1 review score driver (4.4 stars <1wk vs 2.2 stars >4wk)
- 60% of sellers concentrated in São Paulo state
- Top revenue: health_beauty ($1.26M), watches_gifts ($1.2M)
- NaN risk: delivery dates can be null (handle in CustomerAgent)
- NaN risk: product category join can produce null top_category (fillna in ProductAgent)

## Output Schema (utils/schema.py)
DomainAgentOutput keys: agent, timestamp, insights, metrics, top_opportunity, risk_flags
ConnectorOutput keys: timestamp, cross_domain_insights, strategic_recommendation, priority_actions, briefing

## G-Stack Usage
- /plan-ceo-review — scope and strategy decisions
- /plan-eng-review — architecture, data flow, edge cases
- /engineer — implement agent logic
- /qa — test agent outputs
- /ship — commit completed work

## BMAD Workflow
- @bmad-agent-pm — write one story at a time
- @bmad-agent-dev — implement + test that story
- pytest — confirm green before next story
- Never implement more than one story at a time

## Current Status
- [x] EDA complete — all key signals identified
- [x] OpenRouter connected via n8n (tested)
- [x] Project structure scaffolded
- [x] CEO plan complete (docs/specs/ceo-plan.md)
- [x] Eng review complete — all gaps resolved
- [ ] utils/schema.py
- [ ] utils/validators.py
- [ ] utils/base_agent.py
- [ ] Customer Agent MVP + tests
- [ ] Product Agent MVP + tests
- [ ] Seller Agent MVP + tests
- [ ] run_all.py
- [ ] Connector Agent (defined after domain MVPs)

## Team
AI Club — transparent development in Discord. All work shared. Everyone maintains full system.