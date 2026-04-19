"""Real-time ops dashboard for Olist walk-forward runs.

Usage:
    python dashboard.py            # port 5001 (auto-increments if busy)
    python dashboard.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ITERATIONS_DIR = Path("outputs") / "iterations"
DEFAULT_PORT = 5001

# ── colour helpers ──────────────────────────────────────────────────────────

def _c(text: str, colour: str) -> str:
    return f'<span style="color:{colour}">{text}</span>'


def _status_colour(status: str) -> str:
    return "green" if status.lower() == "ok" else "red"


def _confidence_colour(val: str) -> str:
    v = val.upper()
    if v in ("HIGH", "STRONG"):
        return "green"
    if v in ("MEDIUM", "ADEQUATE"):
        return "orange"
    return "red"


def _accuracy_colour(pct: float | None) -> str:
    if pct is None:
        return "grey"
    if pct >= 70:
        return "green"
    if pct >= 50:
        return "orange"
    return "red"


# ── data loading ─────────────────────────────────────────────────────────────

def _load_iterations() -> list[dict[str, Any]]:
    """Read all iteration JSON files, sorted by prediction_month descending."""
    if not ITERATIONS_DIR.exists():
        return []
    files = sorted(ITERATIONS_DIR.glob("*.json"), key=lambda p: p.stem, reverse=True)
    iterations: list[dict[str, Any]] = []
    for f in files:
        try:
            with f.open(encoding="utf-8") as fh:
                iterations.append(json.load(fh))
        except Exception:
            pass
    return iterations


# ── signal distribution ───────────────────────────────────────────────────────

def _count_signals(
    summary: list[dict[str, Any]], key: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in summary:
        val = row.get(key)
        if val:
            counts[val.upper()] = counts.get(val.upper(), 0) + 1
    return counts


def _render_geo_signal(summary: list[dict[str, Any]]) -> str:
    counts = _count_signals(summary, "geo_confidence")
    parts = []
    for label, colour in [("HIGH", "green"), ("MEDIUM", "orange"), ("LOW", "red")]:
        n = counts.get(label, 0)
        parts.append(f'{_c(label, colour)}: {n}')
    return "  ".join(parts) if parts else "—"


def _render_supply_signal(summary: list[dict[str, Any]]) -> str:
    counts = _count_signals(summary, "supply_confidence")
    parts = []
    for label, colour in [("STRONG", "green"), ("ADEQUATE", "orange"), ("WEAK", "red")]:
        n = counts.get(label, 0)
        parts.append(f'{_c(label, colour)}: {n}')
    return "  ".join(parts) if parts else "—"


def _render_customer_signal(summary: list[dict[str, Any]]) -> str:
    counts = _count_signals(summary, "customer_confidence")
    parts = []
    for label, colour in [("HIGH", "green"), ("MEDIUM", "orange"), ("LOW", "red")]:
        n = counts.get(label, 0)
        parts.append(f'{_c(label, colour)}: {n}')
    return "  ".join(parts) if parts else "—"


def _render_logistics_signal(summary: list[dict[str, Any]]) -> str:
    counts = _count_signals(summary, "logistics_confidence")
    parts = []
    for label, colour in [("STRONG", "green"), ("ADEQUATE", "orange"), ("WEAK", "red")]:
        n = counts.get(label, 0)
        parts.append(f'{_c(label, colour)}: {n}')
    return "  ".join(parts) if parts else "—"


# ── iteration HTML ────────────────────────────────────────────────────────────

def _render_iter_html(it: dict[str, Any]) -> str:
    iteration = it.get("iteration", "?")
    month = it.get("prediction_month", "?")
    status = it.get("status", "?")
    timings = it.get("agent_timings") or {}
    summary = it.get("agent_signal_summary") or []
    decisions = it.get("top_connector_decisions") or []
    validation = it.get("validation")

    status_tag = _c(f"[{status.upper()}]", _status_colour(status))

    # ── agent table ──────────────────────────────────────────────────────────
    agent_rows_html = []
    agents_cfg = [
        ("geographic",        _render_geo_signal(summary)),
        ("supply_quality",    _render_supply_signal(summary)),
        ("customer_readiness",_render_customer_signal(summary)),
        ("logistics",         _render_logistics_signal(summary)),
        ("connector",         "—"),
    ]
    for agent_name, signal_html in agents_cfg:
        t = timings.get(agent_name) or {}
        ws = t.get("wall_seconds")
        lc = t.get("llm_calls", 0)
        time_str = f"{ws}s" if ws is not None else "—"
        calls_str = str(lc) if lc is not None else "—"
        agent_rows_html.append(
            f"  <tr>"
            f'<td style="padding:0 12px 0 4px;color:#aaa">{agent_name}</td>'
            f'<td style="padding:0 12px;text-align:center">{calls_str}</td>'
            f'<td style="padding:0 12px;text-align:right">{time_str}</td>'
            f'<td style="padding:0 4px">{signal_html}</td>'
            f"</tr>"
        )
    agent_table = (
        '<table style="border-collapse:collapse;font-family:monospace;font-size:0.9em;margin:8px 0">'
        '<thead><tr>'
        '<th style="text-align:left;padding:0 12px 4px 4px;color:#888">AGENT</th>'
        '<th style="text-align:center;padding:0 12px 4px;color:#888">CALLS</th>'
        '<th style="text-align:right;padding:0 12px 4px;color:#888">TIME</th>'
        '<th style="text-align:left;padding:0 4px 4px;color:#888">SIGNAL</th>'
        "</tr></thead>"
        "<tbody>" + "".join(agent_rows_html) + "</tbody>"
        "</table>"
    )

    # ── top decision ─────────────────────────────────────────────────────────
    if decisions:
        top = decisions[0]
        state = top.get("state", "?")
        cat = top.get("category", "?")
        dec = top.get("decision", "?")
        conf = top.get("confidence", "")
        score = top.get("composite_score")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
        conf_str = _c(f"[{conf}]", _confidence_colour(conf)) if conf else ""
        top_dec_html = (
            f'<div style="margin:6px 0;font-size:0.9em">'
            f'<span style="color:#888">TOP DECISION:</span>  '
            f'<strong>{state} × {cat}</strong>  →  {dec}  {conf_str}'
            f'  <span style="color:#888">score: {score_str}</span>'
            f"</div>"
        )
    else:
        top_dec_html = '<div style="margin:6px 0;color:#888;font-size:0.9em">TOP DECISION: —</div>'

    # ── validation row ────────────────────────────────────────────────────────
    if validation is None:
        val_html = _c("No validation (first iteration)", "grey")
    else:
        validated = validation.get("validated", 0)
        correct = validation.get("correct", 0)
        incorrect = validation.get("incorrect", 0)
        if validated > 0:
            pct = 100.0 * correct / validated
            pct_str = f"{pct:.1f}%"
            acc_colour = _accuracy_colour(pct)
            flag = "GREEN" if pct >= 70 else ("ORANGE" if pct >= 50 else "RED")
            val_html = (
                f"validated: {validated}   "
                f"correct: {correct}   "
                f"incorrect: {incorrect}   "
                f"accuracy: {_c(pct_str, acc_colour)}  {_c(f'[{flag}]', acc_colour)}"
            )
        else:
            val_html = _c("No validation yet", "grey")
    val_section_html = (
        f'<div style="margin:6px 0;font-size:0.9em">'
        f'<span style="color:#888">VALIDATION (prev iteration):</span><br>'
        f'&nbsp;&nbsp;&nbsp;&nbsp;{val_html}'
        f"</div>"
    )

    divider = '<hr style="border:none;border-top:1px solid #444;margin:12px 0">'
    return (
        f'<div style="margin-bottom:8px">'
        f'<div style="font-weight:bold;font-size:1em;letter-spacing:1px">'
        f'ITER {iteration}&nbsp;&nbsp;{month}&nbsp;&nbsp;{status_tag}'
        f"</div>"
        f"{agent_table}"
        f"{top_dec_html}"
        f"{val_section_html}"
        f"{divider}"
        f"</div>"
    )


# ── full page ─────────────────────────────────────────────────────────────────

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Olist Walk-Forward Dashboard</title>
<style>
  body {{
    background: #1a1a1a;
    color: #e0e0e0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 14px;
    margin: 0;
    padding: 16px 24px;
  }}
  #header {{
    border-bottom: 2px solid #555;
    padding-bottom: 8px;
    margin-bottom: 16px;
    color: #fff;
    font-size: 1.1em;
    letter-spacing: 1px;
  }}
  #content {{ margin-top: 8px; }}
</style>
</head>
<body>
<div id="header">
  OLIST WALK-FORWARD&nbsp;&nbsp;|&nbsp;&nbsp;last updated:
  <span id="updated">{updated}</span>
</div>
<div id="content">
{content}
</div>
<script>
var _pollErrorShown = false;
function _colourConf(val) {{
  if (!val) return val;
  var v = val.toUpperCase();
  var c = (v === 'HIGH' || v === 'STRONG') ? 'green'
        : (v === 'MEDIUM' || v === 'ADEQUATE') ? 'orange' : 'red';
  return '<span style="color:' + c + '">[' + val + ']</span>';
}}
function _colourAcc(pct) {{
  return pct >= 70 ? 'green' : pct >= 50 ? 'orange' : 'red';
}}
function _countSignals(summary, key) {{
  var counts = {{}};
  (summary || []).forEach(function(row) {{
    var v = (row[key] || '').toUpperCase();
    if (v) counts[v] = (counts[v] || 0) + 1;
  }});
  return counts;
}}
function _sigRow(counts, labels) {{
  return labels.map(function(cfg) {{
    var lbl = cfg[0], col = cfg[1];
    var n = counts[lbl] || 0;
    return '<span style="color:' + col + '">' + lbl + '</span>: ' + n;
  }}).join('&nbsp;&nbsp;');
}}
function renderData(iterations) {{
  var updated = new Date().toUTCString();
  document.getElementById('updated').textContent = updated;
  if (!iterations || iterations.length === 0) {{
    document.getElementById('content').innerHTML =
      '<p style="color:grey">No iterations yet.</p>';
    return;
  }}
  var html = '';
  iterations.forEach(function(it) {{
    var iter = it.iteration || '?';
    var month = it.prediction_month || '?';
    var status = (it.status || '?').toUpperCase();
    var stCol = status === 'OK' ? 'green' : 'red';
    var timings = it.agent_timings || {{}};
    var summary = it.agent_signal_summary || [];
    var decisions = it.top_connector_decisions || [];
    var val = it.validation;

    // agent table
    var agentCfg = [
      ['geographic',         _sigRow(_countSignals(summary,'geo_confidence'),
                              [['HIGH','green'],['MEDIUM','orange'],['LOW','red']])],
      ['supply_quality',     _sigRow(_countSignals(summary,'supply_confidence'),
                              [['STRONG','green'],['ADEQUATE','orange'],['WEAK','red']])],
      ['customer_readiness', _sigRow(_countSignals(summary,'customer_confidence'),
                              [['HIGH','green'],['MEDIUM','orange'],['LOW','red']])],
      ['logistics',          _sigRow(_countSignals(summary,'logistics_confidence'),
                              [['STRONG','green'],['ADEQUATE','orange'],['WEAK','red']])],
      ['connector',          '&mdash;']
    ];
    var tbody = agentCfg.map(function(ac) {{
      var t = timings[ac[0]] || {{}};
      var ws = (t.wall_seconds !== null && t.wall_seconds !== undefined) ? t.wall_seconds + 's' : '&mdash;';
      var lc = (t.llm_calls !== null && t.llm_calls !== undefined) ? t.llm_calls : '&mdash;';
      return '<tr>'
        + '<td style="padding:0 12px 0 4px;color:#aaa">' + ac[0] + '</td>'
        + '<td style="padding:0 12px;text-align:center">' + lc + '</td>'
        + '<td style="padding:0 12px;text-align:right">' + ws + '</td>'
        + '<td style="padding:0 4px">' + ac[1] + '</td>'
        + '</tr>';
    }}).join('');
    var agentTable = '<table style="border-collapse:collapse;font-family:monospace;font-size:0.9em;margin:8px 0">'
      + '<thead><tr>'
      + '<th style="text-align:left;padding:0 12px 4px 4px;color:#888">AGENT</th>'
      + '<th style="text-align:center;padding:0 12px 4px;color:#888">CALLS</th>'
      + '<th style="text-align:right;padding:0 12px 4px;color:#888">TIME</th>'
      + '<th style="text-align:left;padding:0 4px 4px;color:#888">SIGNAL</th>'
      + '</tr></thead><tbody>' + tbody + '</tbody></table>';

    // top decision
    var topDecHtml;
    if (decisions.length > 0) {{
      var top = decisions[0];
      var score = typeof top.composite_score === 'number' ? top.composite_score.toFixed(2) : '?';
      topDecHtml = '<div style="margin:6px 0;font-size:0.9em">'
        + '<span style="color:#888">TOP DECISION:</span>  '
        + '<strong>' + (top.state||'?') + ' &times; ' + (top.category||'?') + '</strong>'
        + '  &rarr;  ' + (top.decision||'?')
        + '  ' + _colourConf(top.confidence||'')
        + '  <span style="color:#888">score: ' + score + '</span>'
        + '</div>';
    }} else {{
      topDecHtml = '<div style="margin:6px 0;color:#888;font-size:0.9em">TOP DECISION: &mdash;</div>';
    }}

    // validation
    var valHtml;
    if (val === null || val === undefined) {{
      valHtml = '<span style="color:grey">No validation (first iteration)</span>';
    }} else {{
      var validated = val.validated || 0;
      var correct   = val.correct   || 0;
      var incorrect = val.incorrect || 0;
      if (validated > 0) {{
        var pct = 100.0 * correct / validated;
        var pctStr = pct.toFixed(1) + '%';
        var ac = _colourAcc(pct);
        var flag = pct >= 70 ? 'GREEN' : (pct >= 50 ? 'ORANGE' : 'RED');
        valHtml = 'validated: ' + validated
          + '   correct: ' + correct
          + '   incorrect: ' + incorrect
          + '   accuracy: <span style="color:' + ac + '">' + pctStr + '</span>'
          + '  <span style="color:' + ac + '">[' + flag + ']</span>';
      }} else {{
        valHtml = '<span style="color:grey">No validation yet</span>';
      }}
    }}
    var valSection = '<div style="margin:6px 0;font-size:0.9em">'
      + '<span style="color:#888">VALIDATION (prev iteration):</span><br>'
      + '&nbsp;&nbsp;&nbsp;&nbsp;' + valHtml
      + '</div>';

    html += '<div style="margin-bottom:8px">'
      + '<div style="font-weight:bold;font-size:1em;letter-spacing:1px">'
      + 'ITER ' + iter + '&nbsp;&nbsp;' + month + '&nbsp;&nbsp;'
      + '<span style="color:' + stCol + '">[' + status + ']</span>'
      + '</div>'
      + agentTable + topDecHtml + valSection
      + '<hr style="border:none;border-top:1px solid #444;margin:12px 0">'
      + '</div>';
  }});
  document.getElementById('content').innerHTML = html;
}}
setInterval(function() {{
  fetch('/data')
    .then(function(r) {{ return r.json(); }})
    .then(function(data) {{
      _pollErrorShown = false;
      renderData(data);
    }})
    .catch(function() {{
      if (!_pollErrorShown) {{
        document.getElementById('header').insertAdjacentHTML(
          'beforeend',
          ' <span style="color:red">[POLL ERROR]</span>'
        );
        _pollErrorShown = true;
      }}
    }});
}}, 5000);
</script>
</body>
</html>
"""


def _build_page(iterations: list[dict[str, Any]]) -> str:
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if iterations:
        content = "\n".join(_render_iter_html(it) for it in iterations)
    else:
        content = '<p style="color:grey">No iterations yet.</p>'
    return _PAGE_TEMPLATE.format(updated=updated, content=content)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/data":
            self._serve_data()
        else:
            self._serve_page()

    def _serve_data(self) -> None:
        iterations = _load_iterations()
        body = json.dumps(iterations, ensure_ascii=False).encode("utf-8")
        self._write_response(200, "application/json", body)

    def _serve_page(self) -> None:
        iterations = _load_iterations()
        body = _build_page(iterations).encode("utf-8")
        self._write_response(200, "text/html; charset=utf-8", body)

    def _write_response(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # type: ignore[override]
        pass  # suppress per-request log noise


# ── server bootstrap ──────────────────────────────────────────────────────────

def _find_free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + 99}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Olist walk-forward dashboard")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Starting port (auto-increments if busy, default {DEFAULT_PORT})"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    port = _find_free_port(args.port)
    server = ThreadingHTTPServer(("", port), _DashboardHandler)
    print(f"Dashboard: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
