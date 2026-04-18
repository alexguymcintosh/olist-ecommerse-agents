#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from Olist CSVs (pandas + Chart.js)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils.data_loader import load_all

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "outputs" / "dashboard.html"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Olist dashboard HTML generator")
    p.add_argument(
        "--states",
        type=int,
        default=10,
        metavar="N",
        help="Top N states for heatmap and seller/customer chart (default 10, max 27)",
    )
    p.add_argument(
        "--categories",
        type=int,
        default=10,
        metavar="M",
        help="Top M categories for heatmap and revenue bar (default 10, max 73)",
    )
    p.add_argument(
        "--months",
        type=int,
        default=25,
        metavar="K",
        help="Number of most recent months to show for orders-over-time (default 25, min 1)",
    )
    return p.parse_args()


def validate_args(ns: argparse.Namespace) -> None:
    if not 1 <= ns.states <= 27:
        sys.exit("--states must be between 1 and 27")
    if not 1 <= ns.categories <= 73:
        sys.exit("--categories must be between 1 and 73")
    if ns.months < 1:
        sys.exit("--months must be at least 1")


def month_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=False, errors="coerce").dt.to_period("M").dt.to_timestamp()


def build_aggregations(
    dfs: dict[str, pd.DataFrame],
    n_states: int,
    n_categories: int,
    n_months: int,
) -> dict:
    orders = dfs["orders"].copy()
    customers = dfs["customers"].copy()
    order_items = dfs["order_items"].copy()
    products = dfs["products"].copy()
    sellers = dfs["sellers"].copy()
    categories = dfs["categories"].copy()

    cat_col_en = "product_category_name_english"
    cat_col_pt = "product_category_name"
    if cat_col_en not in categories.columns:
        raise ValueError("categories table must include product_category_name_english")

    products = products.merge(
        categories[[cat_col_pt, cat_col_en]],
        on=cat_col_pt,
        how="left",
    )
    products[cat_col_en] = products[cat_col_en].fillna(products[cat_col_pt])

    oc = orders.merge(customers, on="customer_id", how="inner")
    oc["order_month"] = month_key(oc["order_purchase_timestamp"])

    # --- Orders per state per month (orders + customers joined)
    orders_state_month = (
        oc.groupby(["customer_state", "order_month"], observed=True)
        .size()
        .reset_index(name="order_count")
    )

    # --- Item-level frame with state + English category
    items = order_items.merge(orders[["order_id", "customer_id"]], on="order_id", how="inner")
    items = items.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="inner")
    items = items.merge(products[["product_id", cat_col_en]], on="product_id", how="inner")
    items.rename(columns={cat_col_en: "category_en"}, inplace=True)

    # Top states by item volume
    state_vol = items.groupby("customer_state", observed=True).size().sort_values(ascending=False)
    top_states = list(state_vol.head(n_states).index)

    # Top categories by item volume (global)
    cat_vol = items.groupby("category_en", observed=True).size().sort_values(ascending=False)
    top_cats = list(cat_vol.head(n_categories).index)

    heatmap_items = items[
        items["customer_state"].isin(top_states) & items["category_en"].isin(top_cats)
    ]
    heatmap_counts = (
        heatmap_items.groupby(["customer_state", "category_en"], observed=True)
        .size()
        .reset_index(name="volume")
    )
    pivot = heatmap_counts.pivot(
        index="customer_state",
        columns="category_en",
        values="volume",
    ).reindex(index=top_states, columns=top_cats, fill_value=0)

    matrix_data = []
    for yi, st in enumerate(pivot.index):
        for xi, cat in enumerate(pivot.columns):
            v = int(pivot.loc[st, cat])
            matrix_data.append({"x": xi, "y": yi, "v": v})

    # --- Total orders per month (all states), last n_months
    monthly_orders = (
        oc.drop_duplicates(subset=["order_id"])
        .groupby("order_month", observed=True)
        .size()
        .sort_index()
    )
    if len(monthly_orders) > n_months:
        monthly_orders = monthly_orders.iloc[-n_months:]
    monthly_labels = [d.strftime("%Y-%m") for d in monthly_orders.index]
    monthly_values = [int(x) for x in monthly_orders.values]

    # --- Revenue per category per state (for top-category bar we use global category revenue)
    items_rev = order_items.merge(orders[["order_id", "customer_id"]], on="order_id", how="inner")
    items_rev = items_rev.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="inner")
    items_rev = items_rev.merge(products[["product_id", cat_col_en]], on="product_id", how="inner")
    items_rev["revenue"] = items_rev["price"].astype(float) + items_rev["freight_value"].astype(float)

    revenue_category_state = (
        items_rev.groupby(["customer_state", cat_col_en], observed=True)["revenue"]
        .sum()
        .reset_index()
    )

    rev_by_cat = (
        items_rev.groupby(cat_col_en, observed=True)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(n_categories)
    )
    rev_cat_labels = [str(x) for x in rev_by_cat.index]
    rev_cat_values = [float(x) for x in rev_by_cat.values]

    # --- Seller count vs customer count per state (top N states by order count)
    orders_per_state = oc.drop_duplicates("order_id").groupby("customer_state", observed=True).size()
    top_states_sc = list(orders_per_state.sort_values(ascending=False).head(n_states).index)

    seller_counts = sellers.groupby("seller_state", observed=True).size()
    customer_counts = customers.groupby("customer_state", observed=True).size()

    seller_vals = [int(seller_counts.get(s, 0)) for s in top_states_sc]
    customer_vals = [int(customer_counts.get(s, 0)) for s in top_states_sc]

    return {
        "heatmap": {
            "matrix": matrix_data,
            "state_labels": [str(s) for s in pivot.index.tolist()],
            "category_labels": [str(c) for c in pivot.columns.tolist()],
            "max_v": max((d["v"] for d in matrix_data), default=1),
        },
        "orders_per_month": {"labels": monthly_labels, "values": monthly_values},
        "seller_customer": {
            "labels": top_states_sc,
            "sellers": seller_vals,
            "customers": customer_vals,
        },
        "revenue_categories": {"labels": rev_cat_labels, "values": rev_cat_values},
    }


def render_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Olist dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@3.0.0/dist/chartjs-chart-matrix.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1rem 1.25rem; background: #0f1419; color: #e7e9ea; }}
    h1 {{ font-size: 1.25rem; font-weight: 600; margin: 0 0 1rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.25rem; align-items: start; }}
    .card {{ background: #1a2332; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,.35); }}
    .card h2 {{ font-size: 0.95rem; margin: 0 0 0.75rem; color: #8b98a5; font-weight: 600; }}
    .chart-wrap {{ position: relative; height: 360px; width: 100%; }}
    .chart-wrap.tall {{ height: 420px; }}
    footer {{ margin-top: 1.25rem; font-size: 0.75rem; color: #6e767d; }}
  </style>
</head>
<body>
  <h1>Olist — dashboard</h1>
  <div class="grid">
    <div class="card">
      <h2>Heatmap — top states × top categories (order items)</h2>
      <div class="chart-wrap tall"><canvas id="chartHeat"></canvas></div>
    </div>
    <div class="card">
      <h2>Orders per month (all states)</h2>
      <div class="chart-wrap"><canvas id="chartMonth"></canvas></div>
    </div>
    <div class="card">
      <h2>Seller vs customer count by state</h2>
      <div class="chart-wrap"><canvas id="chartSellCust"></canvas></div>
    </div>
    <div class="card">
      <h2>Top categories by revenue (price + freight)</h2>
      <div class="chart-wrap"><canvas id="chartRev"></canvas></div>
    </div>
  </div>
  <footer>Generated by visualise.py — Chart.js + chartjs-chart-matrix</footer>
  <script type="application/json" id="payload">{data_json}</script>
  <script>
    const P = JSON.parse(document.getElementById('payload').textContent);

    function heatColor(t) {{
      const a = [15, 98, 254];
      const b = [255, 99, 132];
      const r = Math.round(a[0] + (b[0] - a[0]) * t);
      const g = Math.round(a[1] + (b[1] - a[1]) * t);
      const bl = Math.round(a[2] + (b[2] - a[2]) * t);
      return 'rgba(' + r + ',' + g + ',' + bl + ',0.92)';
    }}

    const hm = P.heatmap;
    const maxV = Math.max(1, hm.max_v);
    new Chart(document.getElementById('chartHeat'), {{
      type: 'matrix',
      data: {{
        datasets: [{{
          label: 'Order items',
          data: hm.matrix,
          backgroundColor: (ctx) => {{
            const v = ctx.raw && ctx.raw.v !== undefined ? ctx.raw.v : 0;
            return heatColor(v / maxV);
          }},
          borderColor: 'rgba(0,0,0,0.15)',
          borderWidth: 1,
          width: (ctx) => {{
            const a = ctx.chart.chartArea || {{}};
            if (!a.width) return 0;
            return (a.right - a.left) / Math.max(1, hm.category_labels.length);
          }},
          height: (ctx) => {{
            const a = ctx.chart.chartArea || {{}};
            if (!a.height) return 0;
            return (a.bottom - a.top) / Math.max(1, hm.state_labels.length);
          }},
        }}]
      }},
      options: {{
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              title: () => '',
              label: (item) => {{
                const r = item.raw;
                const st = hm.state_labels[r.y];
                const cat = hm.category_labels[r.x];
                return st + ' / ' + cat + ': ' + r.v;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            type: 'category',
            labels: hm.category_labels,
            offset: true,
            ticks: {{ maxRotation: 90, minRotation: 45, color: '#8b98a5', font: {{ size: 10 }} }},
            grid: {{ display: false }}
          }},
          y: {{
            type: 'category',
            labels: hm.state_labels,
            offset: true,
            reverse: true,
            ticks: {{ color: '#8b98a5', font: {{ size: 11 }} }},
            grid: {{ display: false }}
          }}
        }}
      }}
    }});

    const om = P.orders_per_month;
    new Chart(document.getElementById('chartMonth'), {{
      type: 'bar',
      data: {{
        labels: om.labels,
        datasets: [{{
          label: 'Orders',
          data: om.values,
          backgroundColor: 'rgba(54, 162, 235, 0.75)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        }}]
      }},
      options: {{
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#8b98a5', maxRotation: 45 }}, grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
          y: {{ beginAtZero: true, ticks: {{ color: '#8b98a5' }}, grid: {{ color: 'rgba(255,255,255,0.06)' }} }}
        }}
      }}
    }});

    const sc = P.seller_customer;
    new Chart(document.getElementById('chartSellCust'), {{
      type: 'bar',
      data: {{
        labels: sc.labels,
        datasets: [
          {{
            label: 'Sellers',
            data: sc.sellers,
            backgroundColor: 'rgba(75, 192, 192, 0.75)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
          }},
          {{
            label: 'Customers',
            data: sc.customers,
            backgroundColor: 'rgba(153, 102, 255, 0.75)',
            borderColor: 'rgba(153, 102, 255, 1)',
            borderWidth: 1
          }}
        ]
      }},
      options: {{
        maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#e7e9ea' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#8b98a5' }}, grid: {{ display: false }} }},
          y: {{ beginAtZero: true, ticks: {{ color: '#8b98a5' }}, grid: {{ color: 'rgba(255,255,255,0.06)' }} }}
        }}
      }}
    }});

    const rc = P.revenue_categories;
    new Chart(document.getElementById('chartRev'), {{
      type: 'bar',
      data: {{
        labels: rc.labels,
        datasets: [{{
          label: 'Revenue (BRL)',
          data: rc.values,
          backgroundColor: 'rgba(255, 159, 64, 0.75)',
          borderColor: 'rgba(255, 159, 64, 1)',
          borderWidth: 1
        }}]
      }},
      options: {{
        indexAxis: 'y',
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ beginAtZero: true, ticks: {{ color: '#8b98a5' }}, grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
          y: {{ ticks: {{ color: '#8b98a5', font: {{ size: 10 }} }}, grid: {{ display: false }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    validate_args(args)
    try:
        dfs = load_all()
    except FileNotFoundError as e:
        sys.exit(f"Missing data file: {e}. Place all 8 Olist CSVs under data/.")

    payload = build_aggregations(
        dfs,
        n_states=args.states,
        n_categories=args.categories,
        n_months=args.months,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
