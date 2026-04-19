"""Performance visualisation report for Olist walk-forward runs.

Reads all outputs/iterations/*.json files and writes a static HTML report.

Usage:
    python perf_viz.py
    python perf_viz.py --dir outputs/iterations --out custom_perf.html
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_AGENTS_ORDER = [
    "geographic",
    "supply_quality",
    "customer_readiness",
    "logistics",
    "connector",
]

DEFAULT_DIR = Path("outputs") / "iterations"
DEFAULT_OUT = Path("outputs") / "perf_report.html"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_iterations(src_dir: Path) -> list[dict[str, Any]]:
    """Load all *.json files from src_dir, sorted by prediction_month ascending."""
    files = sorted(src_dir.glob("*.json"), key=lambda p: p.stem)
    iterations: list[dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            iterations.append(data)
        except Exception:
            pass
    return iterations


# ── helpers ───────────────────────────────────────────────────────────────────

def _dash(val: Any) -> str:
    """Return '—' for None/empty, else str(val)."""
    if val is None or val == "":
        return "—"
    return str(val)


def _status_colour(status: str) -> str:
    return "green" if str(status).lower() == "ok" else "red"


def _conf_colour(conf: str | None) -> str:
    if not conf:
        return "grey"
    v = str(conf).upper()
    if v == "HIGH":
        return "green"
    if v == "MEDIUM":
        return "orange"
    return "red"


def _acc_colour(pct: float | None) -> str:
    if pct is None:
        return "grey"
    if pct >= 70:
        return "green"
    if pct >= 50:
        return "orange"
    return "red"


def _accuracy(validation: dict[str, Any] | None) -> float | None:
    if not validation:
        return None
    correct = validation.get("correct", 0) or 0
    incorrect = validation.get("incorrect", 0) or 0
    total = correct + incorrect
    if total == 0:
        return None
    return 100.0 * correct / total


def _total_wall(timings: dict[str, Any] | None) -> float | None:
    if not timings:
        return None
    total = 0.0
    has_any = False
    for v in timings.values():
        ws = v.get("wall_seconds")
        if isinstance(ws, (int, float)):
            total += ws
            has_any = True
    return round(total, 2) if has_any else None


def _total_llm(timings: dict[str, Any] | None) -> int | None:
    if not timings:
        return None
    return sum(
        (v.get("llm_calls") or 0) for v in timings.values()
        if isinstance(v.get("llm_calls"), int)
    )


# ── CSS / chrome ──────────────────────────────────────────────────────────────

_CSS = """
body { font-family: Arial, sans-serif; font-size: 14px; background: #fafafa;
       color: #222; margin: 0; padding: 20px 32px; }
h1 { font-size: 1.4em; margin-bottom: 4px; }
h2 { font-size: 1.15em; border-bottom: 2px solid #aaa; padding-bottom: 4px;
     margin-top: 32px; }
table { border-collapse: collapse; margin: 10px 0; }
th { background: #37474f; color: #fff; padding: 5px 10px;
     font-size: 0.85em; text-align: center; border: 1px solid #555; }
th.left { text-align: left; }
td { padding: 4px 10px; border: 1px solid #ddd; font-size: 0.85em;
     text-align: center; }
td.left { text-align: left; }
.meta { background: #eceff1; border: 1px solid #b0bec5; border-radius: 4px;
        padding: 10px 16px; margin-bottom: 24px; font-size: 0.88em; }
.best-worst { margin: 10px 0; font-size: 0.88em; color: #444; }
.bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 0.85em; }
.bar-label { width: 160px; flex-shrink: 0; color: #333; }
.bar-track { background: #e0e0e0; height: 14px; flex: 1; border-radius: 2px;
             overflow: hidden; margin: 0 8px; }
.bar-fill { height: 100%; background: #546e7a; border-radius: 2px; }
.bar-val { width: 60px; text-align: right; color: #555; font-size: 0.82em; }
"""


def _page_wrap(body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n<title>Perf Report</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


# ── report header ─────────────────────────────────────────────────────────────

def _header(src_dir: Path, iterations: list[dict[str, Any]]) -> str:
    count = len(iterations)
    months = [it.get("prediction_month", "") for it in iterations if it.get("prediction_month")]
    date_range = f"{min(months)} → {max(months)}" if months else "—"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return (
        "<h1>Olist Performance Visualisation Report</h1>\n"
        '<div class="meta">\n'
        f"<strong>Source:</strong> {src_dir.resolve()}<br>\n"
        f"<strong>Files read:</strong> {count}<br>\n"
        f"<strong>Date range:</strong> {date_range}<br>\n"
        f"<strong>Generated:</strong> {generated}\n"
        "</div>\n"
    )


# ── section 1: iteration summary table ───────────────────────────────────────

def _section1(iterations: list[dict[str, Any]]) -> str:
    html = "<h2>Section 1 — Iteration Summary</h2>\n"
    html += (
        "<table>\n<thead><tr>"
        "<th class='left'>Month</th>"
        "<th>Status</th>"
        "<th>Wall Time (s)</th>"
        "<th>LLM Calls</th>"
        "<th>Validated</th>"
        "<th>Correct</th>"
        "<th>Incorrect</th>"
        "<th>Accuracy</th>"
        "</tr></thead>\n<tbody>\n"
    )

    for it in iterations:
        month = it.get("prediction_month", "—")
        status = str(it.get("status", "?")).lower()
        status_col = _status_colour(status)

        timings = it.get("agent_timings") or None
        wall = _total_wall(timings)
        llm = _total_llm(timings)

        val = it.get("validation") or {}
        validated = val.get("validated") if val else None
        correct = val.get("correct") if val else None
        incorrect = val.get("incorrect") if val else None
        acc = _accuracy(val if val else None)
        acc_col = _acc_colour(acc)
        acc_str = f"{acc:.1f}%" if acc is not None else "—"

        html += (
            "<tr>"
            f"<td class='left'>{month}</td>"
            f'<td><span style="color:{status_col};font-weight:bold">{status.upper()}</span></td>'
            f"<td>{_dash(wall)}</td>"
            f"<td>{_dash(llm)}</td>"
            f"<td>{_dash(validated)}</td>"
            f"<td>{_dash(correct)}</td>"
            f"<td>{_dash(incorrect)}</td>"
            f'<td><span style="color:{acc_col};font-weight:bold">{acc_str}</span></td>'
            "</tr>\n"
        )

    html += "</tbody>\n</table>\n"
    return html


# ── section 2: per-agent timing table ────────────────────────────────────────

def _section2(iterations: list[dict[str, Any]]) -> str:
    # Only render if at least 1 iteration has agent_timings
    timed = [it for it in iterations if it.get("agent_timings")]
    if not timed:
        return ""  # Section entirely absent — AC 5

    # Collect wall_seconds per agent (exclude null — from error iterations)
    agent_times: dict[str, list[float]] = {a: [] for a in _AGENTS_ORDER}
    agent_llm: dict[str, int] = {a: 0 for a in _AGENTS_ORDER}

    for it in timed:
        timings = it["agent_timings"]
        for agent in _AGENTS_ORDER:
            t = timings.get(agent) or {}
            ws = t.get("wall_seconds")
            lc = t.get("llm_calls") or 0
            if isinstance(ws, (int, float)):
                agent_times[agent].append(float(ws))
            agent_llm[agent] += lc

    html = "<h2>Section 2 — Per-Agent Timing</h2>\n"
    html += (
        "<table>\n<thead><tr>"
        "<th class='left'>Agent</th>"
        "<th>Avg (s)</th>"
        "<th>Min (s)</th>"
        "<th>Max (s)</th>"
        "<th>Total LLM Calls</th>"
        "</tr></thead>\n<tbody>\n"
    )

    avgs: dict[str, float] = {}
    for agent in _AGENTS_ORDER:
        times = agent_times[agent]
        if times:
            avg = round(sum(times) / len(times), 2)
            mn = round(min(times), 2)
            mx = round(max(times), 2)
            avgs[agent] = avg
        else:
            avg = mn = mx = None  # type: ignore[assignment]
            avgs[agent] = 0.0

        lc_total = agent_llm[agent]
        html += (
            "<tr>"
            f"<td class='left'>{agent}</td>"
            f"<td>{_dash(avg)}</td>"
            f"<td>{_dash(mn)}</td>"
            f"<td>{_dash(mx)}</td>"
            f"<td>{_dash(lc_total)}</td>"
            "</tr>\n"
        )

    html += "</tbody>\n</table>\n"

    # div bar chart — proportional to avg_seconds
    max_avg = max(avgs.values()) if avgs else 1.0
    if max_avg == 0:
        max_avg = 1.0
    html += "<h3>Average wall time per agent</h3>\n"
    for agent in _AGENTS_ORDER:
        avg = avgs.get(agent, 0.0)
        bar_pct = int(100 * avg / max_avg)
        html += (
            f'<div class="bar-row">'
            f'<span class="bar-label">{agent}</span>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{bar_pct}%"></div>'
            f"</div>"
            f'<span class="bar-val">{avg}s</span>'
            f"</div>\n"
        )

    return html


# ── section 3: correct/incorrect trend ───────────────────────────────────────

def _section3(iterations: list[dict[str, Any]]) -> str:
    html = "<h2>Section 3 — Correct / Incorrect Trend</h2>\n"
    html += (
        "<table>\n<thead><tr>"
        "<th class='left'>Month</th>"
        "<th>Correct</th>"
        "<th>Incorrect</th>"
        "<th>Accuracy</th>"
        "<th>Trend</th>"
        "</tr></thead>\n<tbody>\n"
    )

    prev_acc: float | None = None
    best: tuple[str, float] | None = None     # (month, pct)
    worst: tuple[str, float] | None = None    # (month, pct)

    for it in iterations:
        month = it.get("prediction_month", "—")
        val = it.get("validation") or {}
        correct = val.get("correct") if val else None
        incorrect = val.get("incorrect") if val else None
        acc = _accuracy(val if val else None)
        acc_col = _acc_colour(acc)
        acc_str = f"{acc:.1f}%" if acc is not None else "—"

        # trend arrow
        if prev_acc is None or acc is None:
            trend_html = '<span style="color:grey">→</span>'
        elif acc > prev_acc:
            trend_html = '<span style="color:green">↑</span>'
        elif acc < prev_acc:
            trend_html = '<span style="color:red">↓</span>'
        else:
            trend_html = '<span style="color:grey">→</span>'

        html += (
            "<tr>"
            f"<td class='left'>{month}</td>"
            f"<td>{_dash(correct)}</td>"
            f"<td>{_dash(incorrect)}</td>"
            f'<td><span style="color:{acc_col};font-weight:bold">{acc_str}</span></td>'
            f"<td style='font-size:1.2em;text-align:center'>{trend_html}</td>"
            "</tr>\n"
        )

        if acc is not None:
            prev_acc = acc
            if best is None or acc > best[1]:
                best = (month, acc)
            if worst is None or acc < worst[1]:
                worst = (month, acc)

    html += "</tbody>\n</table>\n"

    if best:
        html += (
            f'<p class="best-worst">'
            f'<strong>Best iteration:</strong> {best[0]} ({best[1]:.1f}%)&nbsp;&nbsp;'
            f'<strong>Worst validated iteration:</strong> {worst[0]} ({worst[1]:.1f}%)'  # type: ignore[index]
            f"</p>\n"
        )

    return html


# ── section 4: top 10 connector decisions ────────────────────────────────────

def _section4(iterations: list[dict[str, Any]]) -> str:
    # Collect all decisions across all iterations
    all_decisions: list[dict[str, Any]] = []
    for it in iterations:
        month = it.get("prediction_month", "")
        for dec in it.get("top_connector_decisions") or []:
            entry = dict(dec)
            entry.setdefault("month", month)
            all_decisions.append(entry)

    # Sort by composite_score descending, take top 10
    all_decisions.sort(
        key=lambda d: float(d.get("composite_score") or 0),
        reverse=True,
    )
    top10 = all_decisions[:10]

    html = "<h2>Section 4 — Top Connector Decisions (All Iterations)</h2>\n"
    if not top10:
        html += "<p style='color:grey'>No connector decisions recorded yet.</p>\n"
        return html

    html += (
        "<table>\n<thead><tr>"
        "<th>Rank</th>"
        "<th class='left'>Month</th>"
        "<th>State</th>"
        "<th>Category</th>"
        "<th class='left'>Decision</th>"
        "<th>Confidence</th>"
        "<th>Score</th>"
        "<th>Most Predictive</th>"
        "</tr></thead>\n<tbody>\n"
    )

    for rank, dec in enumerate(top10, start=1):
        month = dec.get("month", "—")
        state = dec.get("state", "—")
        category = dec.get("category", "—")
        decision = dec.get("decision", "—")
        conf = dec.get("confidence") or "—"
        score = dec.get("composite_score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
        most_pred = dec.get("most_predictive_agent") or "—"
        conf_col = _conf_colour(conf if conf != "—" else None)

        html += (
            "<tr>"
            f"<td>{rank}</td>"
            f"<td class='left'>{month}</td>"
            f"<td>{state}</td>"
            f"<td>{category}</td>"
            f"<td class='left'>{decision}</td>"
            f'<td><span style="color:{conf_col};font-weight:bold">{conf}</span></td>'
            f"<td>{score_str}</td>"
            f"<td>{most_pred}</td>"
            "</tr>\n"
        )

    html += "</tbody>\n</table>\n"
    return html


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML performance report from iteration JSON files."
    )
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_DIR),
        help=f"Directory containing iteration JSON files (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output HTML path (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    src_dir = Path(args.dir)
    out_path = Path(args.out)

    if not src_dir.exists():
        print(f"Error: directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    iterations = _load_iterations(src_dir)
    if not iterations:
        print(f"Error: no JSON files found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(iterations)} iteration file(s) from {src_dir}")

    body = (
        _header(src_dir, iterations)
        + _section1(iterations)
        + _section2(iterations)
        + _section3(iterations)
        + _section4(iterations)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_page_wrap(body), encoding="utf-8")
    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()
