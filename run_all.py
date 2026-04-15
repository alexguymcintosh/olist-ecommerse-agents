#!/usr/bin/env python3
"""Orchestrator: load Olist data, run domain agents + Connector, Rich briefing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from agents.connector.connector_agent import ConnectorAgent
from agents.customer.customer_agent import CustomerAgent
from agents.product.product_agent import ProductAgent
from agents.seller.seller_agent import SellerAgent
from utils.data_loader import load_all
from utils.openrouter_client import PROD_MODEL, RND_MODEL
from utils.schema import ConnectorOutput, DomainAgentOutput


def _resolve_model(name: str) -> str:
    return PROD_MODEL if name == "prod" else RND_MODEL


def _dataset_stats(data: dict[str, Any]) -> dict[str, int]:
    n_orders = len(data["orders"])
    n_customers = len(data["customers"])
    n_sellers = int(data["sellers"]["seller_id"].nunique())
    n_cats = len(data["categories"].index)
    return {
        "n_orders": n_orders,
        "n_customers": n_customers,
        "n_sellers": n_sellers,
        "n_categories": n_cats,
    }


def _display_period(data: dict[str, Any]) -> str:
    ts = pd.to_datetime(data["orders"]["order_purchase_timestamp"], errors="coerce")
    mx = ts.max()
    if pd.isna(mx):
        return "Olist sample"
    return mx.strftime("%B %Y")


def _safe_run_domain(
    agent_class: type,
    data: dict[str, Any],
    model: str,
    teach: bool,
) -> DomainAgentOutput:
    try:
        return agent_class(data, model=model, teach=teach).run()
    except Exception as e:
        return {
            "agent": agent_class.AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "insights": [f"{agent_class.__name__} raised: {e}"],
            "metrics": {},
            "top_opportunity": "unavailable",
            "risk_flags": ["agent_failed"],
        }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _line_customer(out: DomainAgentOutput, stats: dict[str, int]) -> str:
    head = f"CUSTOMER ({stats['n_customers']:,} customers, ~{stats['n_orders']:,} orders)"
    m = out.get("metrics") or {}
    if not m:
        return f"{head}: output incomplete (agent_failed). Check risk_flags or run with --teach."
    rr = m.get("repeat_rate", 0.0)
    ar = m.get("avg_review_score", 0.0)
    late = m.get("pct_late_delivery", 0.0)
    risk_note = ""
    if late and late >= 0.2:
        risk_note = f" Top risk: late delivery {_fmt_pct(late)}."
    elif out.get("risk_flags"):
        risk_note = f" Flags: {', '.join(out['risk_flags'][:3])}."
    return (
        f"{head}: {_fmt_pct(rr)} repeat rate; avg review {ar:.2f}; "
        f"avg delivery {m.get('avg_delivery_days', 0):.1f} d.{risk_note}"
    ).strip()


def _line_product(out: DomainAgentOutput, stats: dict[str, int]) -> str:
    head = f"PRODUCT ({stats['n_categories']} cats)"
    m = out.get("metrics") or {}
    if not m:
        return f"{head}: output incomplete (agent_failed)."
    top = m.get("top_category", "")
    rev = m.get("top_category_revenue", 0.0)
    aov = m.get("avg_order_value", 0.0)
    opp = out.get("top_opportunity") or ""
    tail = f" Top opportunity: {opp}" if opp else ""
    return f"{head}: {top} leads at ${rev:,.0f}. AOV ${aov:,.0f}.{tail}"


def _line_seller(out: DomainAgentOutput, stats: dict[str, int]) -> str:
    head = f"SELLER ({stats['n_sellers']:,} sellers)"
    m = out.get("metrics") or {}
    if not m:
        return f"{head}: output incomplete (agent_failed)."
    sp = m.get("pct_sao_paulo", 0.0)
    conc = m.get("top_seller_revenue_concentration", 0.0)
    opp = out.get("top_opportunity") or ""
    tail = f" {opp}" if opp else ""
    return (
        f"{head}: ~{_fmt_pct(sp)} SP-heavy seller base; "
        f"~{_fmt_pct(conc)} revenue in top-20% sellers.{tail}"
    )


def _line_connector(conn: ConnectorOutput) -> str:
    rec = (conn.get("strategic_recommendation") or "").strip()
    if rec:
        return f"CONNECTOR: {rec}"
    cross = conn.get("cross_domain_insights") or []
    if cross:
        return f"CONNECTOR: {cross[0]}"
    br = (conn.get("briefing") or "").strip()
    return f"CONNECTOR: {br[:400]}{'…' if len(br) > 400 else ''}"


def _print_briefing_panels(
    console: Console,
    period: str,
    customer_out: DomainAgentOutput,
    product_out: DomainAgentOutput,
    seller_out: DomainAgentOutput,
    connector_out: ConnectorOutput,
    timings: dict[str, float],
    stats: dict[str, int],
) -> None:
    console.print(
        Rule(f"[bold cyan]OLIST BUSINESS BRIEFING — {period}[/bold cyan]", style="cyan")
    )
    console.print()

    panels: list[tuple[str, str]] = [
        ("Customer", _line_customer(customer_out, stats)),
        ("Product", _line_product(product_out, stats)),
        ("Seller", _line_seller(seller_out, stats)),
        ("Connector summary", _line_connector(connector_out)),
    ]
    key_for = {
        "Customer": "customer",
        "Product": "product",
        "Seller": "seller",
        "Connector summary": "connector",
    }
    for title, body in panels:
        t = timings.get(key_for.get(title, ""), 0.0)
        subtitle = f"{t:.1f}s" if t else None
        console.print(
            Panel(
                body,
                title=f"[bold]{title}[/bold]",
                subtitle=subtitle,
                border_style="dim",
            )
        )
        console.print()

    briefing = connector_out.get("briefing", "")
    if briefing:
        console.print(Panel(briefing, title="[bold]Connector briefing[/bold]", border_style="cyan"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Customer, Product, Seller, and Connector agents on Olist data.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write full JSON output to outputs/YYYY-MM-DD-HH-MM.json",
    )
    parser.add_argument(
        "--teach",
        action="store_true",
        help="Print prompts and raw LLM responses for each agent section",
    )
    parser.add_argument(
        "--model",
        choices=["rnd", "prod"],
        default="rnd",
        help='LLM profile: "rnd" (default) or "prod"',
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "Error: OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = _resolve_model(args.model)

    try:
        data = load_all()
    except Exception as e:
        print(f"Fatal error: could not load data: {e}", file=sys.stderr)
        sys.exit(1)

    stats = _dataset_stats(data)
    period = _display_period(data)
    console = Console()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    customer_out = _safe_run_domain(CustomerAgent, data, model, args.teach)
    timings["customer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    product_out = _safe_run_domain(ProductAgent, data, model, args.teach)
    timings["product"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seller_out = _safe_run_domain(SellerAgent, data, model, args.teach)
    timings["seller"] = time.perf_counter() - t0

    domain_outputs = [customer_out, product_out, seller_out]

    t0 = time.perf_counter()
    connector_out = ConnectorAgent(domain_outputs, model=model, teach=args.teach).run()
    timings["connector"] = time.perf_counter() - t0

    if not args.teach:
        _print_briefing_panels(
            console,
            period,
            customer_out,
            product_out,
            seller_out,
            connector_out,
            timings,
            stats,
        )
    else:
        if connector_out.get("briefing"):
            console.print(
                Panel(
                    connector_out["briefing"],
                    title="[bold]Connector briefing[/bold]",
                    border_style="cyan",
                )
            )

    if args.save:
        os.makedirs("outputs", exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
        path = os.path.join("outputs", f"{stamp}.json")
        payload = {
            "model": args.model,
            "teach": args.teach,
            "period_label": period,
            "timings_seconds": timings,
            "customer": customer_out,
            "product": product_out,
            "seller": seller_out,
            "connector": connector_out,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        console.print(f"[dim]Saved:[/dim] {path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
