"""Abstract base class for domain agents with shared run() orchestration."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from utils import data_loader
from utils.openrouter_client import RND_MODEL, build_analyst_prompt, query_llm
from utils.schema import DomainAgentOutput
from utils.validators import validate_output


class BaseAgent(ABC):
    """Orchestrates prepare → metrics → prompt → LLM → parse → validate."""

    AGENT_NAME: str

    def __init__(self, data: dict, model: str = RND_MODEL, teach: bool = False) -> None:
        self.data = data
        self.model = model
        self.teach = teach

    @abstractmethod
    def _prepare_data(self) -> pd.DataFrame:
        """Join and return the full working DataFrame for this agent."""

    @abstractmethod
    def _compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Deterministic metrics on the full DataFrame (no sampling)."""

    @abstractmethod
    def _build_question(self, metrics: dict[str, Any], sample_str: str) -> str:
        """User/LLM question using pre-computed metrics and a row sample string."""

    def run(self) -> DomainAgentOutput:
        try:
            df = self._prepare_data()
            metrics = self._compute_metrics(df)
            sampled = data_loader.sample(df, n=50)
            sample_str = data_loader.to_llm_string(sampled, max_rows=50)
            question = self._build_question(metrics, sample_str)
            messages = build_analyst_prompt(sample_str, question)
            raw = query_llm(messages, model=self.model)
            if self.teach:
                self._print_teach_panel(messages, raw)
            parsed = self._parse_llm_response(raw)
        except Exception as e:
            return self._handle_error(e)

        output = self._assemble_output(metrics, parsed)
        try:
            validate_output(output, self.__class__.__name__)
        except ValueError:
            raise
        return output

    def _assemble_output(self, metrics: dict[str, Any], parsed: dict[str, Any]) -> DomainAgentOutput:
        insights = parsed.get("insights", [])
        if not isinstance(insights, list):
            insights = []
        risk_flags = parsed.get("risk_flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = []
        top_op = parsed.get("top_opportunity", "")
        if not isinstance(top_op, str):
            top_op = str(top_op)
        return {
            "agent": self.AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "insights": insights,
            "metrics": metrics,
            "top_opportunity": top_op,
            "risk_flags": risk_flags,
        }

    # Brace-window extraction handles LLM prose prefix and trailing text more robustly than strip-then-loads
    def _parse_llm_response(self, raw: str) -> dict[str, Any]:
        """Extract first ``{`` through last ``}``, then ``json.loads``."""
        s = raw.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(
                f"{self.AGENT_NAME}: LLM response contained no JSON object.\nRaw: {raw[:200]}"
            )
        json_str = s[start : end + 1]
        try:
            out = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{self.AGENT_NAME}: LLM returned non-JSON response: {e}\nRaw: {raw[:200]}"
            ) from e
        if not isinstance(out, dict):
            raise ValueError(f"{self.AGENT_NAME}: LLM JSON root must be an object, got {type(out)}")
        return out

    def _handle_error(self, e: Exception) -> DomainAgentOutput:
        """Partial output when prepare/LLM/parse fails (validation errors are re-raised)."""
        return {
            "agent": self.AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "insights": [],
            "metrics": {},
            "top_opportunity": "",
            "risk_flags": ["agent_failed"],
        }

    def _print_teach_panel(self, messages: list, raw: str) -> None:
        console = Console()
        prompt_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        console.print(Panel(prompt_text, title="Prompt"))
        console.print(Panel(raw, title="Raw LLM response"))
