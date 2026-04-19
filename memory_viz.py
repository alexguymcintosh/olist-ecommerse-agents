"""Memory visualisation report for Olist walk-forward runs.

Reads a memory_*.db SQLite file and writes a static HTML report.

Usage:
    python memory_viz.py                                 # auto-detects most recent DB
    python memory_viz.py --db memory_2026-04-19.db
    python memory_viz.py --db memory_2026-04-19.db --out custom_report.html
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── DB discovery ──────────────────────────────────────────────────────────────

def _find_latest_db(root: Path) -> Path | None:
    dbs = sorted(root.glob("memory_*.db"), key=lambda p: p.stat().st_mtime)
    return dbs[-1] if dbs else None


def _out_path(db_path: Path, out_arg: str | None) -> Path:
    if out_arg:
        return Path(out_arg)
    stem = db_path.stem.replace("memory_", "memory_report_")
    if stem == db_path.stem:
        stem = "memory_report_" + db_path.stem
    return db_path.parent / f"{stem}.html"


# ── colour helpers ────────────────────────────────────────────────────────────

def _str_or_none(val: Any) -> str | None:
    """Coerce pandas NaN / None / empty string to None; else str."""
    if val is None:
        return None
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass
    s = str(val).strip()
    return s if s else None


def _cell(text: str, bg: str, fg: str = "#111") -> str:
    return (
        f'<td style="background:{bg};color:{fg};text-align:center;'
        f'padding:4px 8px;border:1px solid #ccc">{text}</td>'
    )


def _outcome_cell(outcome: Any) -> str:
    outcome = _str_or_none(outcome)
    if outcome == "correct":
        return _cell("✓", "#c8e6c9")
    if outcome == "incorrect":
        return _cell("✗", "#ffcdd2")
    if outcome == "unknown":
        return _cell("?", "#e0e0e0")
    return _cell("—", "#ffffff")


def _conf_colour(conf: str | None) -> str:
    conf = _str_or_none(conf)
    if not conf:
        return "grey"
    v = conf.upper()
    if v == "HIGH":
        return "green"
    if v == "MEDIUM":
        return "orange"
    return "red"


def _outcome_colour(outcome: str | None) -> str:
    outcome = _str_or_none(outcome)
    if outcome == "correct":
        return "green"
    if outcome == "incorrect":
        return "red"
    if outcome == "unknown":
        return "grey"
    return "grey"


_AGENT_COLOUR: dict[str, str] = {
    "geographic": "#1565c0",        # blue
    "supply_quality": "#e65100",    # orange
    "customer_readiness": "#2e7d32",# green
    "logistics": "#6a1b9a",         # purple
}


def _agent_colour(agent: str | None) -> str:
    return _AGENT_COLOUR.get(agent or "", "grey")


# ── HTML chrome ───────────────────────────────────────────────────────────────

_CSS = """
body { font-family: Arial, sans-serif; font-size: 14px; background: #fafafa;
       color: #222; margin: 0; padding: 20px 32px; }
h1 { font-size: 1.4em; margin-bottom: 4px; }
h2 { font-size: 1.15em; border-bottom: 2px solid #aaa; padding-bottom: 4px;
     margin-top: 32px; }
h3 { font-size: 1em; margin-top: 20px; color: #333; }
table { border-collapse: collapse; margin: 8px 0; }
th { background: #37474f; color: #fff; padding: 5px 10px;
     font-size: 0.85em; text-align: center; border: 1px solid #555; }
th.left { text-align: left; }
td { padding: 4px 10px; border: 1px solid #ccc; font-size: 0.85em; }
.meta { background: #eceff1; border: 1px solid #b0bec5; border-radius: 4px;
        padding: 10px 16px; margin-bottom: 24px; font-size: 0.88em; }
.summary { margin: 10px 0; font-size: 0.9em; color: #333; }
.bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 0.85em; }
.bar-label { width: 180px; flex-shrink: 0; }
.bar-track { background: #e0e0e0; height: 14px; flex: 1; border-radius: 2px;
             overflow: hidden; margin: 0 8px; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-count { width: 30px; text-align: right; color: #555; }
"""


def _page_wrap(body: str, title: str = "Memory Visualisation Report") -> str:
    return (
        f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        f"<meta charset='utf-8'>\n<title>{title}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


# ── section 1: accuracy trend table ──────────────────────────────────────────

def _section1(df: Any) -> str:
    """Accuracy Trend Table — one row per pair, columns = months."""
    import pandas as pd  # already a dependency via utils/memory.py

    pairs = sorted(df[["state", "category"]].drop_duplicates().itertuples(index=False))
    months = sorted(df["month"].unique())

    html = "<h2>Section 1 — Accuracy Trend</h2>\n"
    html += "<table>\n<thead><tr><th class='left'>State × Category</th>"
    for m in months:
        html += f"<th>{m}</th>"
    html += "</tr></thead>\n<tbody>\n"

    total_validated = 0
    total_correct = 0

    for pair in pairs:
        state, category = pair.state, pair.category
        html += f"<tr><td><strong>{state} × {category}</strong></td>"
        pair_df = df[(df["state"] == state) & (df["category"] == category)]
        month_map = dict(zip(pair_df["month"], pair_df["conn_actual_outcome"]))
        for m in months:
            raw_outcome = month_map.get(m)
            outcome = _str_or_none(raw_outcome)
            html += _outcome_cell(outcome)
            if outcome == "correct":
                total_correct += 1
                total_validated += 1
            elif outcome == "incorrect":
                total_validated += 1
        html += "</tr>\n"

    html += "</tbody>\n</table>\n"

    total_pairs = len(pairs)
    if total_validated > 0:
        acc_pct = 100.0 * total_correct / total_validated
        summary = (
            f"{total_pairs} pairs.  "
            f"{total_validated} cells validated.  "
            f"Overall accuracy: <strong>{acc_pct:.1f}%</strong> "
            f"({total_correct} correct / {total_validated - total_correct} incorrect)"
        )
    else:
        summary = f"{total_pairs} pairs.  No validated outcomes yet (all NULL or unknown)."

    html += f'<p class="summary">{summary}</p>\n'
    return html


# ── section 2: most predictive agent per pair ─────────────────────────────────

def _mode(values: list[str]) -> str | None:
    """Return most common value; ties broken alphabetically."""
    if not values:
        return None
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    winners = sorted(k for k, c in counts.items() if c == max_count)
    return winners[0]


def _section2(df: Any) -> str:
    pairs = sorted(df[["state", "category"]].drop_duplicates().itertuples(index=False))

    html = "<h2>Section 2 — Most Predictive Agent Per Pair</h2>\n"
    html += (
        "<table>\n<thead><tr>"
        "<th class='left'>State × Category</th>"
        "<th class='left'>Dominant Agent</th>"
        "<th>Data Points</th>"
        "</tr></thead>\n<tbody>\n"
    )

    agent_pair_counts: dict[str, int] = {}

    for pair in pairs:
        state, category = pair.state, pair.category
        pair_df = df[(df["state"] == state) & (df["category"] == category)]
        values = [
            _str_or_none(v)
            for v in pair_df["conn_most_predictive_agent"].tolist()
            if _str_or_none(v)
        ]
        modal_agent = _mode(values)
        colour = _agent_colour(modal_agent)
        label = modal_agent if modal_agent else "—"
        agent_pair_counts[modal_agent or "—"] = (
            agent_pair_counts.get(modal_agent or "—", 0) + 1
        )
        html += (
            f"<tr>"
            f"<td>{state} × {category}</td>"
            f'<td><span style="color:{colour};font-weight:bold">{label}</span></td>'
            f"<td style='text-align:center'>{len(values)}</td>"
            f"</tr>\n"
        )

    html += "</tbody>\n</table>\n"

    # div bar summary
    if agent_pair_counts:
        max_count = max(agent_pair_counts.values()) or 1
        html += "<h3>Agent dominance summary</h3>\n"
        for agent, count in sorted(agent_pair_counts.items()):
            colour = _agent_colour(agent if agent != "—" else None)
            bar_pct = int(100 * count / max_count)
            html += (
                f'<div class="bar-row">'
                f'<span class="bar-label" style="color:{colour};font-weight:bold">{agent}</span>'
                f'<div class="bar-track">'
                f'<div class="bar-fill" style="width:{bar_pct}%;background:{colour}"></div>'
                f"</div>"
                f'<span class="bar-count">{count}</span>'
                f"</div>\n"
            )

    return html


# ── section 3: decision history per pair ─────────────────────────────────────

def _section3(df: Any) -> str:
    pairs = sorted(df[["state", "category"]].drop_duplicates().itertuples(index=False))

    html = "<h2>Section 3 — Decision History Per Pair</h2>\n"

    for pair in pairs:
        state, category = pair.state, pair.category
        pair_df = (
            df[(df["state"] == state) & (df["category"] == category)]
            .sort_values("month")
        )
        html += f"<h3>{state} × {category}</h3>\n"
        html += (
            "<table>\n<thead><tr>"
            "<th class='left'>Month</th>"
            "<th class='left'>Decision</th>"
            "<th>Confidence</th>"
            "<th>Outcome</th>"
            "</tr></thead>\n<tbody>\n"
        )
        for _, row in pair_df.iterrows():
            month = row["month"]
            decision = _str_or_none(row.get("conn_decision")) or "—"
            conf = _str_or_none(row.get("conn_confidence"))
            outcome = _str_or_none(row.get("conn_actual_outcome"))

            conf_display = (
                f'<span style="color:{_conf_colour(conf)};font-weight:bold">'
                f"{conf or '—'}</span>"
            )
            outcome_display = (
                f'<span style="color:{_outcome_colour(outcome)}">'
                f"{outcome or '—'}</span>"
            )
            html += (
                f"<tr>"
                f"<td>{month}</td>"
                f"<td>{decision}</td>"
                f"<td style='text-align:center'>{conf_display}</td>"
                f"<td style='text-align:center'>{outcome_display}</td>"
                f"</tr>\n"
            )
        html += "</tbody>\n</table>\n"

    return html


# ── report header ─────────────────────────────────────────────────────────────

def _header(db_path: Path, df: Any) -> str:
    total_rows = len(df)
    months = sorted(df["month"].unique())
    states = sorted(df["state"].unique())
    categories = sorted(df["category"].unique())
    month_range = f"{months[0]} → {months[-1]}" if months else "—"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return (
        "<h1>Olist Memory Visualisation Report</h1>\n"
        '<div class="meta">\n'
        f"<strong>DB:</strong> {db_path.resolve()}<br>\n"
        f"<strong>Total rows:</strong> {total_rows}<br>\n"
        f"<strong>Months:</strong> {month_range} ({len(months)} months)<br>\n"
        f"<strong>States:</strong> {', '.join(states)}<br>\n"
        f"<strong>Categories:</strong> {', '.join(categories)}<br>\n"
        f"<strong>Generated:</strong> {generated}\n"
        "</div>\n"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML memory report from a memory_*.db file."
    )
    parser.add_argument("--db", help="Path to memory_*.db (default: most recently modified)")
    parser.add_argument("--out", help="Output HTML path (default: memory_report_<db-stem>.html)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(".")

    if args.db:
        db_path = Path(args.db)
        if not db_path.exists():
            print(f"Error: DB not found: {db_path}", file=sys.stderr)
            sys.exit(1)
    else:
        db_path = _find_latest_db(root)
        if db_path is None:
            print("Error: No memory_*.db files found in current directory.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-selected DB: {db_path}")

    out_path = _out_path(db_path, args.out)

    # Read all rows via Memory class (ensures table exists and uses project schema)
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.memory import Memory  # noqa: PLC0415

    mem = Memory(db_path=db_path)
    df = mem.read_all()

    if df.empty:
        body = (
            _header(db_path, df)
            + '<p style="color:grey">No data in this database yet.</p>\n'
        )
        out_path.write_text(_page_wrap(body), encoding="utf-8")
        print(f"Report written (empty): {out_path}")
        return

    body = (
        _header(db_path, df)
        + _section1(df)
        + _section2(df)
        + _section3(df)
    )
    out_path.write_text(_page_wrap(body), encoding="utf-8")
    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()
