"""Connector agent: synthesises cross-domain insights from domain agent outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel

from utils.openrouter_client import RND_MODEL, build_analyst_prompt, query_llm
from utils.schema import ConnectorOutput, DomainAgentOutput, PriorityAction


class ConnectorAgent:
    """Does not subclass BaseAgent — consumes structured domain outputs only."""

    AGENT_NAME = "connector"

    def __init__(
        self,
        outputs: list[DomainAgentOutput],
        model: str = RND_MODEL,
        teach: bool = False,
    ) -> None:
        if not outputs:
            raise ValueError("No domain agent outputs to synthesise")
        self.outputs = outputs
        self.model = model
        self.teach = teach

    def _build_question(self) -> str:
        return """You are the Connector: synthesise cross-domain insights from the JSON domain outputs above.

Return a JSON object with exactly these keys:
{
  "cross_domain_insights": ["<3-5 concise cross-domain findings as plain English strings>"],
  "strategic_recommendation": "<one top-priority line for leadership>",
  "priority_actions": [
    {"action": "<concrete next step>", "agent": "customer|product|seller|cross-domain", "urgency": "HIGH|MEDIUM|LOW"}
  ],
  "briefing": "<multi-sentence narrative suitable for a CEO terminal briefing; tie delivery, demand, catalog, and supply together>"
}

Use only facts supported by the domain metrics and insights. Return ONLY the JSON object. No prose, no markdown."""

    def _parse_llm_response(self, raw: str) -> dict[str, Any]:
        s = raw.strip()
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Connector: LLM response contained no JSON object.")
        json_str = s[start : end + 1]
        try:
            out = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Connector: non-JSON response: {e}") from e
        if not isinstance(out, dict):
            raise ValueError("Connector: JSON root must be an object.")
        return out

    def _normalize_priority_actions(self, raw: Any) -> list[PriorityAction]:
        if not isinstance(raw, list):
            return []
        out: list[PriorityAction] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip()
            agent = str(item.get("agent", "cross-domain")).strip()
            urgency = str(item.get("urgency", "MEDIUM")).strip().upper()
            if urgency not in ("HIGH", "MEDIUM", "LOW"):
                urgency = "MEDIUM"
            if action:
                out.append({"action": action, "agent": agent, "urgency": urgency})
        return out

    def _assemble(self, parsed: dict[str, Any]) -> ConnectorOutput:
        raw_insights = parsed.get("cross_domain_insights", [])
        if not isinstance(raw_insights, list):
            raw_insights = []
        insights = [str(x) for x in raw_insights if str(x).strip()]

        rec = parsed.get("strategic_recommendation", "")
        if not isinstance(rec, str):
            rec = str(rec)

        briefing = parsed.get("briefing", "")
        if not isinstance(briefing, str):
            briefing = str(briefing)

        actions = self._normalize_priority_actions(parsed.get("priority_actions"))

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cross_domain_insights": insights,
            "strategic_recommendation": rec.strip(),
            "priority_actions": actions,
            "briefing": briefing.strip(),
        }

    def _fallback_output(self) -> ConnectorOutput:
        cross: list[str] = []
        for o in self.outputs:
            agent = o.get("agent", "?")
            if "agent_failed" in o.get("risk_flags", []):
                cross.append(f"{agent}: agent output unavailable; skipped in synthesis.")
            else:
                top = o.get("top_opportunity") or ""
                if top:
                    cross.append(f"{agent}: {top}")
        if not cross:
            cross = ["Synthesis unavailable; review raw domain metrics and insights."]

        strategic = cross[0] if cross else "Review domain agent outputs."
        briefing_parts: list[str] = []
        for o in self.outputs:
            name = str(o.get("agent", "?")).upper()
            briefing_parts.append(f"{name}: {o.get('top_opportunity', 'n/a')}")
        briefing = " | ".join(briefing_parts)

        actions: list[PriorityAction] = []
        for o in self.outputs:
            top = o.get("top_opportunity")
            if top and "agent_failed" not in o.get("risk_flags", []):
                actions.append(
                    {
                        "action": str(top),
                        "agent": str(o.get("agent", "cross-domain")),
                        "urgency": "MEDIUM",
                    }
                )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cross_domain_insights": cross[:5],
            "strategic_recommendation": strategic,
            "priority_actions": actions[:5],
            "briefing": briefing,
        }

    def _print_teach_input(self) -> None:
        console = Console()
        lines = []
        for o in self.outputs:
            n_in = len(o.get("insights") or [])
            n_m = len((o.get("metrics") or {}).keys())
            n_r = len(o.get("risk_flags") or [])
            lines.append(
                f"  {o.get('agent', '?')}: insights[{n_in}], metrics{{{n_m}}}, risk_flags[{n_r}]"
            )
        console.print(
            Panel(
                "\n".join(lines) if lines else "(empty)",
                title=f"[INPUT: {len(self.outputs)} DOMAIN OUTPUTS]",
            )
        )

    def _print_teach_prompt_raw(self, messages: list[dict[str, str]], raw: str) -> None:
        console = Console()
        prompt_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        console.print(Panel(prompt_text, title="[PROMPT SENT]"))
        console.print(Panel(raw, title="[RAW RESPONSE]"))

    def _print_teach_parsed(self, out: ConnectorOutput) -> None:
        console = Console()
        rec = out["strategic_recommendation"]
        if len(rec) > 120:
            rec_disp = rec[:120] + "..."
        else:
            rec_disp = rec
        summary = (
            f"cross_domain_insights: {len(out['cross_domain_insights'])} | "
            f"strategic_recommendation: {rec_disp}"
        )
        console.print(Panel(summary, title="[PARSED OUTPUT]"))

    def run(self) -> ConnectorOutput:
        if self.teach:
            console = Console()
            console.rule("[bold]ConnectorAgent[/bold]")
            self._print_teach_input()

        try:
            payload = json.dumps(self.outputs, indent=2, default=str)
            question = self._build_question()
            messages = build_analyst_prompt(payload, question)
            raw = query_llm(messages, model=self.model, max_tokens=2000)
            if self.teach:
                self._print_teach_prompt_raw(messages, raw)
            parsed = self._parse_llm_response(raw)
            result = self._assemble(parsed)
            if self.teach:
                self._print_teach_parsed(result)
            return result
        except Exception:
            fb = self._fallback_output()
            if self.teach:
                c = Console()
                c.print(Panel(str(fb), title="[FALLBACK — LLM or parse failed]"))
            return fb
