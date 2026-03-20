"""Generate an interactive HTML benchmark dashboard.

Reads all 3 benchmark CSVs, computes validation metrics, and outputs
a single self-contained HTML file with Chart.js visualizations,
sortable tables, and outlier detection.

Usage:
    python -m tests.generate_dashboard

Output:
    benchmark_dashboard.html  (open in any browser)
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIER_MAP = {"Reliable": 1, "Above Avg": 2, "Average": 3, "Below Avg": 4, "Unreliable": 5}
TIER_LABELS = {1: "Reliable", 2: "Above Avg", 3: "Average", 4: "Below Avg", 5: "Unreliable"}
TIER_COLORS = {1: "#22c55e", 2: "#84cc16", 3: "#eab308", 4: "#f97316", 5: "#ef4444"}

BENCHMARKS = [
    ("benchmark_mileage_matrix.csv", "Original 100"),
    ("validation_mileage_matrix.csv", "Validation 100"),
    ("luxury_benchmark_mileage.csv", "Luxury 100"),
]

GRADE_THRESHOLDS = [(15, "A"), (30, "B"), (50, "C"), (70, "D"), (100, "F")]


def _score_to_grade(s: float) -> str:
    for th, g in GRADE_THRESHOLDS:
        if s <= th:
            return g
    return "F"


def _grade_to_tier(g: str) -> int:
    return {"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}.get(g, 3)


def _load_csv(filename: str) -> list[dict]:
    path = PROJECT_ROOT / filename
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    parsed = []
    for r in rows:
        has_label = "TierLabel" in r
        if has_label:
            tier_num = int(r["Tier"])
            tier_lbl = r["TierLabel"]
        else:
            tier_lbl = r["Tier"]
            tier_num = TIER_MAP.get(tier_lbl, 3)

        mileage_re = re.compile(r"^(\d+)k_(V[12])$")
        mileages_v1 = {}
        mileages_v2 = {}
        for key, val in r.items():
            m = mileage_re.match(key)
            if m:
                mi = int(m.group(1))
                ver = m.group(2)
                try:
                    score = float(val)
                except (ValueError, TypeError):
                    score = -1.0
                if ver == "V1":
                    mileages_v1[mi] = score
                else:
                    mileages_v2[mi] = score

        v1_75 = mileages_v1.get(75, -1.0)
        v2_75 = mileages_v2.get(75, -1.0)

        complaints = 0
        try:
            complaints = int(r.get("Complaints", 0))
        except (ValueError, TypeError):
            pass

        sales = r.get("Sales", "")

        parsed.append({
            "make": r["Make"],
            "model": r["Model"],
            "year": int(r["Year"]),
            "tier": tier_num,
            "tier_label": tier_lbl,
            "complaints": complaints,
            "sales": sales,
            "v1_75k": round(v1_75, 1),
            "v2_75k": round(v2_75, 1),
            "mileages_v1": {k: round(v, 1) for k, v in sorted(mileages_v1.items())},
            "mileages_v2": {k: round(v, 1) for k, v in sorted(mileages_v2.items())},
        })
    return parsed


def _spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0

    def _ranks(vals):
        si = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[si[j]] == vals[si[j + 1]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[si[k]] = avg
            i = j + 1
        return r

    rx, ry = _ranks(x), _ranks(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return round(1 - (6 * d_sq) / (n * (n ** 2 - 1)), 3)


def _compute_stats(vehicles: list[dict]) -> dict:
    ok = [v for v in vehicles if v["v1_75k"] >= 0 and v["v2_75k"] >= 0]
    if not ok:
        return {}

    tiers = [float(v["tier"]) for v in ok]
    v1s = [v["v1_75k"] for v in ok]
    v2s = [v["v2_75k"] for v in ok]

    spearman_v1 = _spearman(tiers, v1s)
    spearman_v2 = _spearman(tiers, v2s)

    tier_avgs_v1 = {}
    tier_avgs_v2 = {}
    tier_scores_v2 = defaultdict(list)
    for v in ok:
        t = v["tier"]
        tier_avgs_v1.setdefault(t, []).append(v["v1_75k"])
        tier_avgs_v2.setdefault(t, []).append(v["v2_75k"])
        tier_scores_v2[t].append(v["v2_75k"])

    for t in tier_avgs_v1:
        tier_avgs_v1[t] = round(sum(tier_avgs_v1[t]) / len(tier_avgs_v1[t]), 1)
        tier_avgs_v2[t] = round(sum(tier_avgs_v2[t]) / len(tier_avgs_v2[t]), 1)

    predicted_tiers_v2 = [_grade_to_tier(_score_to_grade(v["v2_75k"])) for v in ok]
    actual_tiers = [v["tier"] for v in ok]
    within_1 = sum(1 for a, p in zip(actual_tiers, predicted_tiers_v2) if abs(a - p) <= 1)
    tier_accuracy = round(100.0 * within_1 / len(ok), 1)

    confusion = [[0] * 5 for _ in range(5)]
    for a, p in zip(actual_tiers, predicted_tiers_v2):
        confusion[a - 1][p - 1] += 1

    mono_pass = 0
    mono_total = 0
    for v in ok:
        mi_v2 = v["mileages_v2"]
        keys = sorted(mi_v2.keys())
        if len(keys) < 2:
            continue
        mono_total += 1
        is_mono = all(mi_v2[keys[i]] <= mi_v2[keys[i + 1]] + 0.05 for i in range(len(keys) - 1))
        if is_mono:
            mono_pass += 1
    mono_rate = round(100.0 * mono_pass / mono_total, 1) if mono_total > 0 else 100.0

    worst = sorted(ok, key=lambda v: abs(v["tier"] - _grade_to_tier(_score_to_grade(v["v2_75k"]))), reverse=True)[:10]
    worst_misses = []
    for v in worst:
        pred = _grade_to_tier(_score_to_grade(v["v2_75k"]))
        gap = abs(v["tier"] - pred)
        if gap == 0:
            continue
        worst_misses.append({
            "vehicle": f"{v['make']} {v['model']} {v['year']}",
            "expected": TIER_LABELS[v["tier"]],
            "predicted": TIER_LABELS[pred],
            "v2_score": v["v2_75k"],
            "gap": gap,
        })

    box_data = {}
    for t in range(1, 6):
        scores = sorted(tier_scores_v2.get(t, []))
        if not scores:
            continue
        n = len(scores)
        box_data[t] = {
            "min": scores[0],
            "q1": scores[max(0, n // 4)],
            "median": scores[n // 2],
            "q3": scores[min(n - 1, 3 * n // 4)],
            "max": scores[-1],
            "values": scores,
        }

    return {
        "spearman_v1": spearman_v1,
        "spearman_v2": spearman_v2,
        "tier_avgs_v1": tier_avgs_v1,
        "tier_avgs_v2": tier_avgs_v2,
        "tier_accuracy": tier_accuracy,
        "confusion": confusion,
        "mono_rate": mono_rate,
        "worst_misses": worst_misses,
        "box_data": {str(k): v for k, v in box_data.items()},
        "count": len(ok),
    }


def _build_html(all_data: dict) -> str:
    data_json = json.dumps(all_data, indent=None)

    return f"""<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Risk Score Benchmark Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; }}
  .metric-card {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); }}
  .tier-1 {{ color: #22c55e; }} .tier-2 {{ color: #84cc16; }} .tier-3 {{ color: #eab308; }}
  .tier-4 {{ color: #f97316; }} .tier-5 {{ color: #ef4444; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ cursor: pointer; user-select: none; position: sticky; top: 0; background: #1e293b; z-index: 2; }}
  th:hover {{ background: #334155; }}
  td, th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }}
  tr:hover td {{ background: #334155; }}
  .outlier {{ background: #7f1d1d !important; }}
  .sort-asc::after {{ content: ' \\25B2'; font-size: 10px; }}
  .sort-desc::after {{ content: ' \\25BC'; font-size: 10px; }}
  select, input {{ background: #1e293b; border: 1px solid #475569; color: #e2e8f0; padding: 6px 12px;
                   border-radius: 6px; font-size: 13px; }}
  .nav-link {{ padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 500;
               transition: background 0.2s; }}
  .nav-link:hover {{ background: #334155; }}
  .nav-link.active {{ background: #3b82f6; color: white; }}
  .confusion td {{ text-align: center; font-weight: 600; min-width: 48px; }}
  .confusion .diag {{ background: #166534; }}
  .confusion .off1 {{ background: #854d0e; }}
  .confusion .off2 {{ background: #7f1d1d; }}
</style>
</head>
<body class="min-h-screen">

<nav class="sticky top-0 z-50 bg-[#0f172a]/95 backdrop-blur border-b border-slate-700 px-6 py-3">
  <div class="max-w-[1400px] mx-auto flex items-center justify-between">
    <h1 class="text-xl font-bold text-white">Risk Score Benchmark Dashboard</h1>
    <div class="flex gap-2">
      <a href="#summary" class="nav-link">Summary</a>
      <a href="#charts" class="nav-link">Charts</a>
      <a href="#all-mileage" class="nav-link">All Mileage</a>
      <a href="#mileage" class="nav-link">Mileage</a>
      <a href="#vehicles" class="nav-link">Vehicles</a>
      <a href="#validation" class="nav-link">Validation</a>
    </div>
  </div>
</nav>

<main class="max-w-[1400px] mx-auto px-6 py-8 space-y-10">

<!-- SUMMARY -->
<section id="summary">
  <h2 class="text-2xl font-bold mb-6">Summary</h2>
  <div id="summary-cards" class="grid grid-cols-1 md:grid-cols-3 gap-6"></div>
  <div class="card mt-6">
    <h3 class="text-lg font-semibold mb-4">Tier Averages at 75k Miles</h3>
    <div id="tier-avg-table"></div>
  </div>
</section>

<!-- CHARTS -->
<section id="charts">
  <h2 class="text-2xl font-bold mb-6">Score Distributions</h2>
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div class="card">
      <h3 class="text-lg font-semibold mb-2">V1 vs V2 Scatter (75k)</h3>
      <div class="mb-2">
        <select id="scatter-benchmark" onchange="updateScatter()" class="text-sm"></select>
      </div>
      <canvas id="scatterChart" height="300"></canvas>
    </div>
    <div class="card">
      <h3 class="text-lg font-semibold mb-2">V2 Score Distribution by Tier</h3>
      <div class="mb-2">
        <select id="box-benchmark" onchange="updateBoxPlot()" class="text-sm"></select>
      </div>
      <canvas id="boxChart" height="300"></canvas>
    </div>
    <div class="card lg:col-span-2">
      <h3 class="text-lg font-semibold mb-2">Score Histograms (75k)</h3>
      <div class="mb-2">
        <select id="hist-benchmark" onchange="updateHistogram()" class="text-sm"></select>
      </div>
      <canvas id="histChart" height="200"></canvas>
    </div>
  </div>
</section>

<!-- ALL VEHICLES MILEAGE -->
<section id="all-mileage">
  <h2 class="text-2xl font-bold mb-6">All Vehicles: Mileage vs Risk Score</h2>
  <div class="card">
    <div class="flex gap-4 mb-4 flex-wrap items-center">
      <select id="allmi-tier" onchange="updateAllMileage()" class="text-sm">
        <option value="0">All Tiers</option>
        <option value="1">Tier 1: Reliable</option>
        <option value="2">Tier 2: Above Avg</option>
        <option value="3">Tier 3: Average</option>
        <option value="4">Tier 4: Below Avg</option>
        <option value="5">Tier 5: Unreliable</option>
      </select>
      <span class="text-xs text-slate-400">
        <span class="inline-block w-8 border-t-2 border-solid border-blue-500 mr-1 align-middle"></span> V1
        <span class="inline-block w-8 border-t-2 border-dashed border-purple-500 ml-3 mr-1 align-middle"></span> V2
        · thick lines = tier average
      </span>
    </div>
    <canvas id="allMileageChart" height="400"></canvas>
  </div>
</section>

<!-- MILEAGE -->
<section id="mileage">
  <h2 class="text-2xl font-bold mb-6">Mileage Curves</h2>
  <div class="card">
    <div class="flex gap-4 mb-4 flex-wrap">
      <select id="mileage-benchmark" onchange="populateMileageVehicles()" class="text-sm"></select>
      <select id="mileage-vehicle" onchange="updateMileageChart()" class="text-sm"></select>
    </div>
    <canvas id="mileageChart" height="250"></canvas>
  </div>
</section>

<!-- VEHICLES TABLE -->
<section id="vehicles">
  <h2 class="text-2xl font-bold mb-6">All Vehicles</h2>
  <div class="card">
    <div class="flex gap-4 mb-4 flex-wrap items-center">
      <select id="table-benchmark" onchange="updateTable()" class="text-sm"></select>
      <input id="table-search" type="text" placeholder="Search make/model..." oninput="updateTable()" class="text-sm w-48">
      <select id="table-tier" onchange="updateTable()" class="text-sm">
        <option value="">All Tiers</option>
        <option value="1">Tier 1: Reliable</option>
        <option value="2">Tier 2: Above Avg</option>
        <option value="3">Tier 3: Average</option>
        <option value="4">Tier 4: Below Avg</option>
        <option value="5">Tier 5: Unreliable</option>
      </select>
      <label class="text-sm flex items-center gap-2">
        <input type="checkbox" id="table-outliers" onchange="updateTable()"> Show outliers only
      </label>
    </div>
    <div class="overflow-x-auto max-h-[600px] overflow-y-auto">
      <table id="vehicle-table">
        <thead><tr id="table-header"></tr></thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>
</section>

<!-- VALIDATION -->
<section id="validation">
  <h2 class="text-2xl font-bold mb-6">Validation Metrics</h2>
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div class="card">
      <h3 class="text-lg font-semibold mb-4">Tier Confusion Matrix (V2)</h3>
      <div class="mb-2">
        <select id="conf-benchmark" onchange="updateConfusion()" class="text-sm"></select>
      </div>
      <div id="confusion-container"></div>
    </div>
    <div class="card">
      <h3 class="text-lg font-semibold mb-4">Worst Misses (V2)</h3>
      <div class="mb-2">
        <select id="miss-benchmark" onchange="updateWorstMisses()" class="text-sm"></select>
      </div>
      <div id="misses-container"></div>
    </div>
  </div>
</section>

</main>

<script>
const DATA = {data_json};

const TIER_LABELS = {{1:"Reliable", 2:"Above Avg", 3:"Average", 4:"Below Avg", 5:"Unreliable"}};
const TIER_COLORS = {{1:"#22c55e", 2:"#84cc16", 3:"#eab308", 4:"#f97316", 5:"#ef4444"}};
const GRADE_TH = [[15,"A"],[30,"B"],[50,"C"],[70,"D"],[100,"F"]];

function scoreToGrade(s) {{ for (const [th,g] of GRADE_TH) if (s<=th) return g; return "F"; }}
function gradeToTier(g) {{ return {{"A":1,"B":2,"C":3,"D":4,"F":5}}[g]||3; }}

let charts = {{}};

function initSelectors() {{
  const names = Object.keys(DATA.benchmarks);
  for (const sel of document.querySelectorAll('select[id$="-benchmark"]')) {{
    sel.innerHTML = names.map(n => `<option value="${{n}}">${{n}}</option>`).join('');
  }}
}}

// ── SUMMARY ──
function renderSummary() {{
  const container = document.getElementById('summary-cards');
  let html = '';
  for (const [name, b] of Object.entries(DATA.benchmarks)) {{
    const s = b.stats;
    if (!s.spearman_v1 && s.spearman_v1 !== 0) continue;
    const delta = (s.spearman_v2 - s.spearman_v1).toFixed(3);
    const deltaColor = delta > 0 ? 'text-green-400' : 'text-red-400';
    html += `
      <div class="card metric-card border-l-4" style="border-left-color:${{s.spearman_v2 > 0.5 ? '#22c55e' : s.spearman_v2 > 0.3 ? '#eab308' : '#ef4444'}}">
        <div class="text-sm text-slate-400 mb-1">${{name}}</div>
        <div class="text-3xl font-bold">${{s.spearman_v2.toFixed(3)}}</div>
        <div class="text-sm text-slate-400 mt-1">V2 Spearman</div>
        <div class="mt-2 text-sm">V1: ${{s.spearman_v1.toFixed(3)}} <span class="${{deltaColor}}">${{delta > 0 ? '+' : ''}}${{delta}}</span></div>
        <div class="mt-1 text-xs text-slate-500">${{s.count}} vehicles | Accuracy ${{s.tier_accuracy}}% | Mono ${{s.mono_rate}}%</div>
      </div>`;
  }}
  container.innerHTML = html;

  let tbl = '<table><thead><tr><th>Tier</th>';
  for (const name of Object.keys(DATA.benchmarks)) tbl += `<th colspan="2">${{name}}</th>`;
  tbl += '</tr><tr><th></th>';
  for (const _ of Object.keys(DATA.benchmarks)) tbl += '<th>V1</th><th>V2</th>';
  tbl += '</tr></thead><tbody>';
  for (let t = 1; t <= 5; t++) {{
    tbl += `<tr><td class="tier-${{t}} font-semibold">${{TIER_LABELS[t]}}</td>`;
    for (const [name, b] of Object.entries(DATA.benchmarks)) {{
      const v1 = b.stats.tier_avgs_v1?.[t] ?? '-';
      const v2 = b.stats.tier_avgs_v2?.[t] ?? '-';
      tbl += `<td>${{v1}}</td><td class="font-semibold">${{v2}}</td>`;
    }}
    tbl += '</tr>';
  }}
  tbl += '</tbody></table>';
  document.getElementById('tier-avg-table').innerHTML = tbl;
}}

// ── SCATTER ──
function updateScatter() {{
  const name = document.getElementById('scatter-benchmark').value;
  const vehicles = DATA.benchmarks[name]?.vehicles || [];
  if (charts.scatter) charts.scatter.destroy();

  const datasets = [];
  for (let t = 1; t <= 5; t++) {{
    const pts = vehicles.filter(v => v.tier === t && v.v1_75k >= 0).map(v => ({{
      x: v.v1_75k, y: v.v2_75k,
      label: `${{v.make}} ${{v.model}} ${{v.year}}`
    }}));
    if (pts.length) datasets.push({{
      label: TIER_LABELS[t], data: pts,
      backgroundColor: TIER_COLORS[t] + '99',
      borderColor: TIER_COLORS[t], pointRadius: 5, pointHoverRadius: 8,
    }});
  }}

  const maxVal = Math.max(
    ...vehicles.map(v => Math.max(v.v1_75k, v.v2_75k)).filter(x => x >= 0), 80
  );

  datasets.push({{
    label: 'V1 = V2', data: [{{x:0,y:0}},{{x:maxVal,y:maxVal}}],
    type: 'line', borderColor: '#475569', borderDash: [5,5],
    pointRadius: 0, borderWidth: 1, order: 10,
  }});

  charts.scatter = new Chart(document.getElementById('scatterChart'), {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true,
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const p = ctx.raw;
              return `${{p.label || ''}}: V1=${{p.x}}, V2=${{p.y}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'V1 Score', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
        y: {{ title: {{ display: true, text: 'V2 Score', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
      }}
    }}
  }});
}}

// ── BOX PLOT (simulated with bar + error bars) ──
function updateBoxPlot() {{
  const name = document.getElementById('box-benchmark').value;
  const stats = DATA.benchmarks[name]?.stats;
  if (!stats?.box_data) return;
  if (charts.box) charts.box.destroy();

  const labels = []; const mins = []; const q1s = []; const medians = []; const q3s = []; const maxs = [];
  const bgColors = []; const scatterData = [];

  for (let t = 1; t <= 5; t++) {{
    const bd = stats.box_data[String(t)];
    if (!bd) continue;
    labels.push(TIER_LABELS[t]);
    mins.push(bd.min); q1s.push(bd.q1); medians.push(bd.median); q3s.push(bd.q3); maxs.push(bd.max);
    bgColors.push(TIER_COLORS[t] + '40');
    bd.values.forEach(v => scatterData.push({{ x: labels.length - 1, y: v, tier: t }}));
  }}

  charts.box = new Chart(document.getElementById('boxChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'Min', data: mins, backgroundColor: 'transparent', borderColor: '#64748b', borderWidth: 1, barPercentage: 0.5 }},
        {{ label: 'Q1-Q3 Range', data: q3s.map((q, i) => q - q1s[i]), backgroundColor: bgColors,
           borderColor: Object.values(TIER_COLORS).slice(0, labels.length), borderWidth: 2, barPercentage: 0.5 }},
        {{ label: 'Median', data: medians, type: 'line', borderColor: '#f8fafc', backgroundColor: '#f8fafc',
           pointRadius: 8, pointStyle: 'line', borderWidth: 2, showLine: false }},
      ]
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ stacked: false, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
        y: {{ title: {{ display: true, text: 'V2 Score', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }},
              min: 0, max: 100 }},
      }},
      plugins: {{
        tooltip: {{ callbacks: {{ label: ctx => {{
          const i = ctx.dataIndex;
          return `Min:${{mins[i]}} Q1:${{q1s[i]}} Med:${{medians[i]}} Q3:${{q3s[i]}} Max:${{maxs[i]}}`;
        }}}} }}
      }}
    }}
  }});
}}

// ── HISTOGRAM ──
function updateHistogram() {{
  const name = document.getElementById('hist-benchmark').value;
  const vehicles = DATA.benchmarks[name]?.vehicles || [];
  if (charts.hist) charts.hist.destroy();

  const bins = Array.from({{length: 10}}, (_, i) => i * 10);
  const v1Counts = new Array(10).fill(0);
  const v2Counts = new Array(10).fill(0);

  for (const v of vehicles) {{
    if (v.v1_75k >= 0) v1Counts[Math.min(9, Math.floor(v.v1_75k / 10))]++;
    if (v.v2_75k >= 0) v2Counts[Math.min(9, Math.floor(v.v2_75k / 10))]++;
  }}

  charts.hist = new Chart(document.getElementById('histChart'), {{
    type: 'bar',
    data: {{
      labels: bins.map(b => `${{b}}-${{b+10}}`),
      datasets: [
        {{ label: 'V1', data: v1Counts, backgroundColor: '#3b82f680', borderColor: '#3b82f6', borderWidth: 1 }},
        {{ label: 'V2', data: v2Counts, backgroundColor: '#8b5cf680', borderColor: '#8b5cf6', borderWidth: 1 }},
      ]
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ title: {{ display: true, text: 'Score Range', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
        y: {{ title: {{ display: true, text: 'Count', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
      }}
    }}
  }});
}}

// ── ALL VEHICLES MILEAGE ──
function updateAllMileage() {{
  const tierFilter = parseInt(document.getElementById('allmi-tier').value) || 0;
  if (charts.allMileage) charts.allMileage.destroy();

  const allVehicles = [];
  for (const [bName, b] of Object.entries(DATA.benchmarks)) {{
    for (const v of (b.vehicles || [])) {{
      allVehicles.push({{ ...v, benchmark: bName }});
    }}
  }}

  const filtered = tierFilter ? allVehicles.filter(v => v.tier === tierFilter) : allVehicles;

  const datasets = [];
  const tiersToShow = tierFilter ? [tierFilter] : [1, 2, 3, 4, 5];

  for (const t of tiersToShow) {{
    const tierVehicles = filtered.filter(v => v.tier === t);
    if (!tierVehicles.length) continue;

    for (const v of tierVehicles) {{
      const label = `${{v.make}} ${{v.model}} ${{v.year}}`;

      const v1pts = Object.entries(v.mileages_v1)
        .map(([k, val]) => ({{ x: parseInt(k), y: val }}))
        .filter(p => p.y >= 0)
        .sort((a, b) => a.x - b.x);
      const v2pts = Object.entries(v.mileages_v2)
        .map(([k, val]) => ({{ x: parseInt(k), y: val }}))
        .filter(p => p.y >= 0)
        .sort((a, b) => a.x - b.x);

      if (v1pts.length > 1) {{
        datasets.push({{
          label: label + ' V1',
          data: v1pts,
          borderColor: TIER_COLORS[t] + '30',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.3,
          borderDash: [],
        }});
      }}
      if (v2pts.length > 1) {{
        datasets.push({{
          label: label + ' V2',
          data: v2pts,
          borderColor: TIER_COLORS[t] + '50',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          borderDash: [4, 2],
        }});
      }}
    }}

    const allMileages = new Set();
    tierVehicles.forEach(v => {{
      Object.keys(v.mileages_v1).forEach(k => allMileages.add(parseInt(k)));
      Object.keys(v.mileages_v2).forEach(k => allMileages.add(parseInt(k)));
    }});
    const sortedMi = [...allMileages].sort((a, b) => a - b);

    const avgV1 = sortedMi.map(mi => {{
      const vals = tierVehicles.map(v => v.mileages_v1[mi]).filter(x => x !== undefined && x >= 0);
      return vals.length ? {{ x: mi, y: vals.reduce((a, b) => a + b, 0) / vals.length }} : null;
    }}).filter(Boolean);

    const avgV2 = sortedMi.map(mi => {{
      const vals = tierVehicles.map(v => v.mileages_v2[mi]).filter(x => x !== undefined && x >= 0);
      return vals.length ? {{ x: mi, y: vals.reduce((a, b) => a + b, 0) / vals.length }} : null;
    }}).filter(Boolean);

    if (avgV1.length > 1) {{
      datasets.push({{
        label: TIER_LABELS[t] + ' Avg V1',
        data: avgV1,
        borderColor: TIER_COLORS[t],
        backgroundColor: 'transparent',
        borderWidth: 3,
        pointRadius: 4,
        pointBackgroundColor: TIER_COLORS[t],
        tension: 0.3,
        borderDash: [],
        order: -1,
      }});
    }}
    if (avgV2.length > 1) {{
      datasets.push({{
        label: TIER_LABELS[t] + ' Avg V2',
        data: avgV2,
        borderColor: TIER_COLORS[t],
        backgroundColor: 'transparent',
        borderWidth: 3,
        pointRadius: 4,
        pointStyle: 'rectRot',
        pointBackgroundColor: TIER_COLORS[t],
        tension: 0.3,
        borderDash: [6, 3],
        order: -1,
      }});
    }}
  }}

  charts.allMileage = new Chart(document.getElementById('allMileageChart'), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true,
      animation: false,
      plugins: {{
        legend: {{
          display: true,
          labels: {{
            filter: item => item.text.includes('Avg'),
            color: '#94a3b8',
            font: {{ size: 11 }},
          }}
        }},
        tooltip: {{
          mode: 'nearest',
          intersect: true,
          callbacks: {{
            label: ctx => {{
              const ds = ctx.dataset;
              return `${{ds.label}}: ${{ctx.parsed.y.toFixed(1)}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ type: 'linear',
              title: {{ display: true, text: 'Mileage (k)', color: '#94a3b8' }},
              grid: {{ color: '#1e293b' }},
              ticks: {{ color: '#94a3b8', callback: v => v + 'k' }} }},
        y: {{ title: {{ display: true, text: 'Risk Score', color: '#94a3b8' }},
              grid: {{ color: '#1e293b' }},
              ticks: {{ color: '#94a3b8' }},
              min: 0, max: 100 }},
      }}
    }}
  }});
}}

// ── MILEAGE ──
function populateMileageVehicles() {{
  const name = document.getElementById('mileage-benchmark').value;
  const vehicles = DATA.benchmarks[name]?.vehicles || [];
  const sel = document.getElementById('mileage-vehicle');
  sel.innerHTML = vehicles.map((v, i) =>
    `<option value="${{i}}">${{v.make}} ${{v.model}} ${{v.year}} (Tier ${{v.tier}})</option>`
  ).join('');
  updateMileageChart();
}}

function updateMileageChart() {{
  const name = document.getElementById('mileage-benchmark').value;
  const idx = parseInt(document.getElementById('mileage-vehicle').value) || 0;
  const vehicles = DATA.benchmarks[name]?.vehicles || [];
  const v = vehicles[idx];
  if (!v) return;
  if (charts.mileage) charts.mileage.destroy();

  const milesV1 = Object.entries(v.mileages_v1).map(([k, val]) => ({{ x: parseInt(k), y: val }}));
  const milesV2 = Object.entries(v.mileages_v2).map(([k, val]) => ({{ x: parseInt(k), y: val }}));

  charts.mileage = new Chart(document.getElementById('mileageChart'), {{
    type: 'line',
    data: {{
      datasets: [
        {{ label: 'V1', data: milesV1, borderColor: '#3b82f6', backgroundColor: '#3b82f620',
           fill: true, tension: 0.3, pointRadius: 4 }},
        {{ label: 'V2', data: milesV2, borderColor: '#8b5cf6', backgroundColor: '#8b5cf620',
           fill: true, tension: 0.3, pointRadius: 4 }},
      ]
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Mileage (k)', color: '#94a3b8' }},
              grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', callback: v => v + 'k' }} }},
        y: {{ title: {{ display: true, text: 'Risk Score', color: '#94a3b8' }},
              grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }}, min: 0, max: 100 }},
      }},
      plugins: {{
        title: {{ display: true, text: `${{v.make}} ${{v.model}} ${{v.year}} - ${{TIER_LABELS[v.tier]}}`,
                  color: '#f8fafc', font: {{ size: 14 }} }}
      }}
    }}
  }});
}}

// ── TABLE ──
let sortCol = 'v2_75k';
let sortDir = 'desc';

function updateTable() {{
  const name = document.getElementById('table-benchmark').value;
  const search = document.getElementById('table-search').value.toLowerCase();
  const tierFilter = document.getElementById('table-tier').value;
  const outlierOnly = document.getElementById('table-outliers').checked;
  let vehicles = DATA.benchmarks[name]?.vehicles || [];

  if (search) vehicles = vehicles.filter(v =>
    (v.make + ' ' + v.model).toLowerCase().includes(search));
  if (tierFilter) vehicles = vehicles.filter(v => v.tier === parseInt(tierFilter));

  vehicles = vehicles.map(v => {{
    const predTier = gradeToTier(scoreToGrade(v.v2_75k));
    return {{ ...v, predTier, gap: Math.abs(v.tier - predTier), isOutlier: Math.abs(v.tier - predTier) >= 2 }};
  }});

  if (outlierOnly) vehicles = vehicles.filter(v => v.isOutlier);

  vehicles.sort((a, b) => {{
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'string') {{ va = va.toLowerCase(); vb = (vb||'').toLowerCase(); }}
    if (va < vb) return sortDir === 'asc' ? -1 : 1;
    if (va > vb) return sortDir === 'asc' ? 1 : -1;
    return 0;
  }});

  const cols = [
    ['make','Make'], ['model','Model'], ['year','Year'], ['tier_label','Tier'],
    ['complaints','Complaints'], ['sales','Sales'],
    ['v1_75k','V1 (75k)'], ['v2_75k','V2 (75k)'], ['gap','Gap'],
  ];

  const hdr = document.getElementById('table-header');
  hdr.innerHTML = cols.map(([key, label]) => {{
    let cls = '';
    if (key === sortCol) cls = sortDir === 'asc' ? 'sort-asc' : 'sort-desc';
    return `<th class="${{cls}}" onclick="sortTable('${{key}}')">${{label}}</th>`;
  }}).join('');

  const body = document.getElementById('table-body');
  body.innerHTML = vehicles.map(v => {{
    const rowCls = v.isOutlier ? 'outlier' : '';
    return `<tr class="${{rowCls}}">
      <td>${{v.make}}</td><td>${{v.model}}</td><td>${{v.year}}</td>
      <td class="tier-${{v.tier}}">${{v.tier_label}}</td>
      <td>${{v.complaints}}</td><td>${{v.sales || 'N/A'}}</td>
      <td>${{v.v1_75k}}</td><td class="font-semibold">${{v.v2_75k}}</td>
      <td class="${{v.gap >= 2 ? 'text-red-400 font-bold' : v.gap >= 1 ? 'text-yellow-400' : 'text-green-400'}}">${{v.gap}}</td>
    </tr>`;
  }}).join('');
}}

function sortTable(col) {{
  if (sortCol === col) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
  else {{ sortCol = col; sortDir = col === 'make' || col === 'model' ? 'asc' : 'desc'; }}
  updateTable();
}}

// ── CONFUSION ──
function updateConfusion() {{
  const name = document.getElementById('conf-benchmark').value;
  const stats = DATA.benchmarks[name]?.stats;
  if (!stats?.confusion) return;

  let html = '<table class="confusion"><thead><tr><th></th>';
  for (let p = 1; p <= 5; p++) html += `<th>Pred ${{TIER_LABELS[p].split(' ')[0]}}</th>`;
  html += '</tr></thead><tbody>';
  for (let a = 0; a < 5; a++) {{
    html += `<tr><td class="tier-${{a+1}} font-semibold">${{TIER_LABELS[a+1]}}</td>`;
    for (let p = 0; p < 5; p++) {{
      const v = stats.confusion[a][p];
      const diff = Math.abs(a - p);
      const cls = diff === 0 ? 'diag' : diff === 1 ? 'off1' : diff >= 2 ? 'off2' : '';
      html += `<td class="${{cls}}">${{v || ''}}</td>`;
    }}
    html += '</tr>';
  }}
  html += '</tbody></table>';
  html += `<p class="text-sm text-slate-400 mt-3">
    <span class="inline-block w-3 h-3 bg-[#166534] mr-1"></span> Correct
    <span class="inline-block w-3 h-3 bg-[#854d0e] ml-3 mr-1"></span> Off by 1
    <span class="inline-block w-3 h-3 bg-[#7f1d1d] ml-3 mr-1"></span> Off by 2+
  </p>`;
  document.getElementById('confusion-container').innerHTML = html;
}}

// ── WORST MISSES ──
function updateWorstMisses() {{
  const name = document.getElementById('miss-benchmark').value;
  const stats = DATA.benchmarks[name]?.stats;
  if (!stats?.worst_misses) return;

  let html = '<table><thead><tr><th>Vehicle</th><th>Expected</th><th>Predicted</th><th>V2</th><th>Gap</th></tr></thead><tbody>';
  for (const m of stats.worst_misses) {{
    html += `<tr>
      <td>${{m.vehicle}}</td>
      <td>${{m.expected}}</td><td>${{m.predicted}}</td>
      <td>${{m.v2_score}}</td>
      <td class="text-red-400 font-bold">${{m.gap}}</td>
    </tr>`;
  }}
  if (!stats.worst_misses.length) html += '<tr><td colspan="5" class="text-slate-500">No misses - all predictions within 1 tier</td></tr>';
  html += '</tbody></table>';
  document.getElementById('misses-container').innerHTML = html;
}}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {{
  initSelectors();
  renderSummary();
  updateScatter();
  updateBoxPlot();
  updateHistogram();
  updateAllMileage();
  populateMileageVehicles();
  updateTable();
  updateConfusion();
  updateWorstMisses();
}});
</script>
</body>
</html>"""


def main():
    print("Generating benchmark dashboard...")

    all_data = {"benchmarks": {}}

    for filename, label in BENCHMARKS:
        vehicles = _load_csv(filename)
        if not vehicles:
            print(f"  SKIP {filename} (not found)")
            continue
        stats = _compute_stats(vehicles)
        all_data["benchmarks"][label] = {
            "vehicles": vehicles,
            "stats": stats,
        }
        print(f"  Loaded {label}: {len(vehicles)} vehicles, "
              f"Spearman V1={stats.get('spearman_v1', '?')} V2={stats.get('spearman_v2', '?')}")

    html = _build_html(all_data)

    out_path = PROJECT_ROOT / "benchmark_dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = out_path.stat().st_size / 1024
    print(f"\nDashboard: {out_path} ({size_kb:.0f} KB)")
    print("Open in your browser to view.")


if __name__ == "__main__":
    main()
