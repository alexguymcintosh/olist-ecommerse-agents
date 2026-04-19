"""Connector agent: decision synthesis across domain agent outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from utils.config import FOCUS_CATEGORIES, FOCUS_STATES
from utils.openrouter_client import (
    RND_MODEL,
    build_analyst_prompt,
    parse_batch_llm_response,
    query_llm,
)
from utils.schema_agents import (
    ConnectorDecision,
    ConnectorOutput,
    CustomerReadinessOutput,
    LogisticsOutput,
    SupplyQualityOutput,
)
from utils.schema_geographic import Prediction

SUPPLY_SCORE = {"STRONG": 1.0, "ADEQUATE": 0.5, "WEAK": 0.0}
READINESS_SCORE = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
LOGISTICS_SCORE = {"STRONG": 1.0, "ADEQUATE": 0.5, "WEAK": 0.0}
PREDICTIVE_AGENTS = {"geographic", "supply_quality", "customer_readiness", "logistics"}


class ConnectorAgent:
    """Make ranked state-category decisions from all 4 agent signals."""

    AGENT_NAME = "connector"

    def __init__(self, memory: Any, model: str = RND_MODEL) -> None:
        self.memory = memory
        self.model = model

    @staticmethod
    def _extract_json_dict(raw: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")
        parsed = json.loads(raw[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not a JSON object.")
        return parsed

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _dedupe_flags(flags: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for flag in flags:
            if flag not in seen:
                seen.add(flag)
                ordered.append(flag)
        return ordered

    def _composite_score(
        self,
        geo: Prediction,
        supply: SupplyQualityOutput,
        customer: CustomerReadinessOutput,
        logistics: LogisticsOutput,
    ) -> float:
        geo_signal = self._safe_float(geo.get("confidence_score"), 0.0) * 0.35
        supply_signal = SUPPLY_SCORE.get(str(supply.get("supply_confidence", "")).upper(), 0.0) * 0.25
        customer_signal = READINESS_SCORE.get(str(customer.get("readiness", "")).upper(), 0.0) * 0.20
        logistics_signal = LOGISTICS_SCORE.get(str(logistics.get("feasibility", "")).upper(), 0.0) * 0.20
        predicted_growth = self._safe_float(geo.get("predicted_growth_pct"), 0.0)
        return (geo_signal + supply_signal + customer_signal + logistics_signal) * predicted_growth

    def _build_prompt(
        self,
        *,
        month: str,
        state: str,
        category: str,
        composite_score: float,
        geo: Prediction,
        supply: SupplyQualityOutput,
        customer: CustomerReadinessOutput,
        logistics: LogisticsOutput,
        prev_outcome_text: str,
    ) -> list[dict[str, str]]:
        summary = (
            f"Multi-signal analysis for {category} in {state}, month {month}:\n\n"
            f"COMPOSITE SCORE: {composite_score:.2f}\n"
            f"GEO: predicted_growth={self._safe_float(geo.get('predicted_growth_pct')):+.1%}, "
            f"confidence={geo.get('confidence', 'LOW')} ({self._safe_float(geo.get('confidence_score')):.2f})\n"
            f"SUPPLY: {supply.get('supply_confidence', 'WEAK')} - {supply.get('reasoning', '') or 'Supply metrics unavailable (agent failed)'}\n"
            f"CUSTOMER: {customer.get('readiness', 'LOW')} - {customer.get('reasoning', '') or 'Customer metrics unavailable (agent failed)'}\n"
            f"LOGISTICS: {logistics.get('feasibility', 'WEAK')} - {logistics.get('reasoning', '') or 'Logistics metrics unavailable (agent failed)'}\n\n"
            f"{prev_outcome_text}"
        )
        question = (
            "What is the single best action Olist should take? Challenge your own recommendation - what could go wrong?\n\n"
            "Return JSON:\n"
            "{"
            '"decision":"...",'
            '"confidence":"HIGH|MEDIUM|LOW",'
            '"urgency":"HIGH|MEDIUM|LOW",'
            '"reasoning":"...",'
            '"challenge":"...",'
            '"most_predictive_agent":"geographic|supply_quality|customer_readiness|logistics"'
            "}"
        )
        return build_analyst_prompt(summary, question)

    def _fallback_decision(
        self,
        *,
        month: str,
        state: str,
        category: str,
        composite_score: float,
        risk_flags: list[str],
    ) -> ConnectorDecision:
        decision = "no_action" if composite_score <= 0 else "review_opportunity"
        urgency = "LOW"
        return {
            "state": state,
            "category": category,
            "month": month,
            "composite_score": composite_score,
            "decision": decision,
            "confidence": "LOW",
            "urgency": urgency,
            "reasoning": "Fallback: connector LLM response parse failed.",
            "challenge": "Model response unavailable; validate manually.",
            "most_predictive_agent": "geographic",
            "risk_flags": self._dedupe_flags(risk_flags + ["connector_failed"]),
        }

    def _parse_connector_response(
        self,
        *,
        parsed: dict[str, Any],
        month: str,
        state: str,
        category: str,
        composite_score: float,
        risk_flags: list[str],
    ) -> ConnectorDecision:
        confidence = str(parsed.get("confidence", "LOW")).upper()
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            confidence = "LOW"
        urgency = str(parsed.get("urgency", "LOW")).upper()
        if urgency not in {"HIGH", "MEDIUM", "LOW"}:
            urgency = "LOW"
        most_predictive_agent = str(parsed.get("most_predictive_agent", "geographic"))
        if most_predictive_agent not in PREDICTIVE_AGENTS:
            most_predictive_agent = "geographic"

        decision = str(parsed.get("decision", "no_action"))
        reasoning = str(parsed.get("reasoning", "No reasoning provided."))
        challenge = str(parsed.get("challenge", "No challenge provided."))

        if composite_score <= 0:
            decision = "no_action"
            urgency = "LOW"

        return {
            "state": state,
            "category": category,
            "month": month,
            "composite_score": composite_score,
            "decision": decision,
            "confidence": confidence,
            "urgency": urgency,
            "reasoning": reasoning,
            "challenge": challenge,
            "most_predictive_agent": most_predictive_agent,
            "risk_flags": self._dedupe_flags(risk_flags),
        }

    def _batch_connector_decisions(
        self, month: str, items: list[dict[str, Any]]
    ) -> dict[tuple[str, str], ConnectorDecision]:
        batch_payload: list[dict[str, Any]] = []
        for item in items:
            batch_payload.append(
                {
                    "state": item["state"],
                    "category": item["category"],
                    "month": month,
                    "composite_score": round(item["composite_score"], 4),
                    "geo_predicted_growth_pct": round(
                        self._safe_float(item["geo"].get("predicted_growth_pct"), 0.0), 4
                    ),
                    "geo_confidence": str(item["geo"].get("confidence", "LOW")),
                    "geo_confidence_score": round(
                        self._safe_float(item["geo"].get("confidence_score"), 0.0), 4
                    ),
                    "supply_confidence": str(
                        item["supply"].get("supply_confidence", "WEAK")
                    ),
                    "customer_readiness": str(
                        item["customer"].get("readiness", "LOW")
                    ),
                    "logistics_feasibility": str(
                        item["logistics"].get("feasibility", "WEAK")
                    ),
                    "supply_reasoning": str(item["supply"].get("reasoning", "")),
                    "customer_reasoning": str(item["customer"].get("reasoning", "")),
                    "logistics_reasoning": str(item["logistics"].get("reasoning", "")),
                    "prev_outcome": item["prev_outcome_text"],
                }
            )

        question = (
            "Assess the best action for each item. "
            "Return ONLY a JSON array where each object has exactly: "
            '{"state": <string>, "category": <string>, "decision": <string>, '
            '"confidence": "HIGH|MEDIUM|LOW", "urgency": "HIGH|MEDIUM|LOW", '
            '"reasoning": <string>, "challenge": <string>, '
            '"most_predictive_agent": "geographic|supply_quality|customer_readiness|logistics"}. '
            "One entry per input item in any order."
        )
        try:
            messages = build_analyst_prompt(json.dumps(batch_payload, indent=2), question)
            raw = query_llm(messages, model=self.model, max_tokens=4000)
            parsed_items = parse_batch_llm_response(raw, items)
            results: dict[tuple[str, str], ConnectorDecision] = {}
            for item, parsed in zip(items, parsed_items):
                state = item["state"]
                category = item["category"]
                composite_score = item["composite_score"]
                risk_flags = item["risk_flags"]

                if composite_score <= 0:
                    decision = {
                        "state": state,
                        "category": category,
                        "month": month,
                        "composite_score": composite_score,
                        "decision": "no_action",
                        "confidence": "LOW",
                        "urgency": "LOW",
                        "reasoning": "Composite score is non-positive; no growth opportunity.",
                        "challenge": "Could miss early inflection if data lags.",
                        "most_predictive_agent": "geographic",
                        "risk_flags": self._dedupe_flags(risk_flags),
                    }
                elif parsed is not None and isinstance(parsed, dict):
                    try:
                        decision = self._parse_connector_response(
                            parsed=parsed,
                            month=month,
                            state=state,
                            category=category,
                            composite_score=composite_score,
                            risk_flags=risk_flags,
                        )
                    except Exception:
                        decision = self._fallback_decision(
                            month=month,
                            state=state,
                            category=category,
                            composite_score=composite_score,
                            risk_flags=risk_flags,
                        )
                else:
                    decision = self._fallback_decision(
                        month=month,
                        state=state,
                        category=category,
                        composite_score=composite_score,
                        risk_flags=risk_flags,
                    )
                results[(state, category)] = decision
            return results
        except Exception:
            return {
                (item["state"], item["category"]): self._fallback_decision(
                    month=month,
                    state=item["state"],
                    category=item["category"],
                    composite_score=item["composite_score"],
                    risk_flags=item["risk_flags"],
                )
                for item in items
            }

    def run(
        self,
        month: str,
        geographic_outputs: list[Prediction],
        supply_outputs: list[SupplyQualityOutput],
        customer_outputs: list[CustomerReadinessOutput],
        logistics_outputs: list[LogisticsOutput],
        prev_month: str | None = None,
        states: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> ConnectorOutput:
        geo_by_pair = {(x["state"], x["category"]): x for x in geographic_outputs}
        supply_by_pair = {(x["state"], x["category"]): x for x in supply_outputs}
        customer_by_pair = {(x["state"], x["category"]): x for x in customer_outputs}
        logistics_by_pair = {(x["state"], x["category"]): x for x in logistics_outputs}

        timestamp = datetime.now(timezone.utc).isoformat()
        decisions: list[ConnectorDecision] = []
        batch_items: list[dict[str, Any]] = []

        selected_states = states if states is not None else FOCUS_STATES
        selected_categories = categories if categories is not None else FOCUS_CATEGORIES

        for state in selected_states:
            for category in selected_categories:
                pair = (state, category)
                geo = geo_by_pair.get(
                    pair,
                    {
                        "state": state,
                        "category": category,
                        "predicted_growth_pct": 0.0,
                        "confidence": "LOW",
                        "confidence_score": 0.0,
                        "reasoning": "Missing geographic signal.",
                    },
                )
                supply = supply_by_pair.get(
                    pair,
                    {
                        "agent": "supply_quality",
                        "timestamp": timestamp,
                        "state": state,
                        "category": category,
                        "month": month,
                        "seller_count": 0,
                        "avg_review_score": 0.0,
                        "avg_delivery_days": 0.0,
                        "churn_risk": "HIGH",
                        "churn_rate": 0.0,
                        "top_seller_id": "",
                        "seller_concentration": 0.0,
                        "supply_confidence": "WEAK",
                        "reasoning": "",
                        "risk_flags": ["agent_failed"],
                    },
                )
                customer = customer_by_pair.get(
                    pair,
                    {
                        "agent": "customer_readiness",
                        "timestamp": timestamp,
                        "state": state,
                        "category": category,
                        "month": month,
                        "avg_spend": 0.0,
                        "order_volume_trend": 0.0,
                        "top_payment_type": "",
                        "high_value_customer_count": 0,
                        "repeat_rate": 0.0,
                        "installment_pct": 0.0,
                        "readiness": "LOW",
                        "reasoning": "",
                        "risk_flags": ["agent_failed"],
                    },
                )
                logistics = logistics_by_pair.get(
                    pair,
                    {
                        "agent": "logistics",
                        "timestamp": timestamp,
                        "state": state,
                        "category": category,
                        "month": month,
                        "avg_delivery_days": 0.0,
                        "pct_on_time": 0.0,
                        "freight_ratio": 0.0,
                        "fastest_seller_state": "",
                        "delivery_variance": 0.0,
                        "cross_state_dependency": 0.0,
                        "feasibility": "WEAK",
                        "reasoning": "",
                        "risk_flags": ["agent_failed"],
                    },
                )

                composite_score = self._composite_score(geo, supply, customer, logistics)

                prev_outcome_text = "First iteration - no prior outcome."
                if prev_month:
                    prev_row = self.memory.read_row(state, category, prev_month)
                    if prev_row and prev_row.get("conn_actual_outcome"):
                        prev_outcome_text = f"Last month outcome: {prev_row['conn_actual_outcome']}"

                risk_flags: list[str] = []
                for agent_payload in (supply, customer, logistics):
                    flags = agent_payload.get("risk_flags", [])
                    if isinstance(flags, list) and "agent_failed" in flags:
                        risk_flags.append("agent_failed")

                batch_items.append(
                    {
                        "state": state,
                        "category": category,
                        "geo": geo,
                        "supply": supply,
                        "customer": customer,
                        "logistics": logistics,
                        "composite_score": composite_score,
                        "prev_outcome_text": prev_outcome_text,
                        "risk_flags": risk_flags,
                    }
                )

        batch_decisions = self._batch_connector_decisions(month, batch_items)

        for item in batch_items:
            state = item["state"]
            category = item["category"]
            decision = batch_decisions.get(
                (state, category),
                self._fallback_decision(
                    month=month,
                    state=state,
                    category=category,
                    composite_score=item["composite_score"],
                    risk_flags=item["risk_flags"],
                ),
            )

            self.memory.write_row(
                state,
                category,
                month,
                conn_decision=decision["decision"],
                conn_confidence=decision["confidence"],
                conn_reasoning=decision["reasoning"],
                conn_most_predictive_agent=decision["most_predictive_agent"],
            )
            decisions.append(decision)

        decisions_sorted = sorted(
            decisions, key=lambda x: x["composite_score"], reverse=True
        )

        follow_up_used = False
        follow_up_agent: str | None = None
        follow_up_question: str | None = None
        follow_up_response: str | None = None
        if len(decisions_sorted) >= 2:
            top = decisions_sorted[0]
            runner_up = decisions_sorted[1]
            should_follow_up = top["composite_score"] > 2 * max(
                runner_up["composite_score"], 0.001
            )
            if should_follow_up:
                follow_up_used = True
                follow_up_agent = top["most_predictive_agent"]
                follow_up_question = (
                    f"Why is {top['state']}x{top['category']} the top opportunity? "
                    "What would change this assessment?"
                )
                summary = (
                    f"Initial decision: {top['decision']}\n"
                    f"Initial challenge: {top['challenge']}"
                )
                messages = build_analyst_prompt(summary, follow_up_question)
                try:
                    follow_up_response = query_llm(
                        messages, model=self.model, max_tokens=200
                    )
                except Exception:
                    follow_up_response = "Follow-up unavailable due to LLM failure."

        top_three = decisions_sorted[:3]
        briefing_lines = [
            f"{idx + 1}. {x['state']} x {x['category']} -> {x['decision']} ({x['composite_score']:.2f})"
            for idx, x in enumerate(top_three)
        ]
        briefing = "\n".join(briefing_lines) if briefing_lines else "No decisions generated."

        return {
            "agent": self.AGENT_NAME,
            "timestamp": timestamp,
            "month": month,
            "decisions": decisions_sorted,
            "briefing": briefing,
            "follow_up_used": follow_up_used,
            "follow_up_agent": follow_up_agent,
            "follow_up_question": follow_up_question,
            "follow_up_response": follow_up_response,
        }
