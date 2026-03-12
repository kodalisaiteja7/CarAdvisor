"""Full end-to-end timing test with actual LLM calls."""

import time
import json
import logging
import os
import sys

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
log = logging.getLogger("timing")
log.setLevel(logging.INFO)

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Set it and re-run.")
    sys.exit(1)

from database.models import init_db
init_db()

TEST_CASES = [
    ("TOYOTA", "CAMRY", 2020, 45000),
    ("HONDA", "CIVIC", 2018, 80000),
    ("HYUNDAI", "SONATA", 2017, 75000),
    ("FORD", "F-150", 2019, 60000),
    ("BMW", "3 SERIES", 2016, 95000),
]

all_results = []

for idx, (make, model, year, mileage) in enumerate(TEST_CASES, 1):
    print(f"\n{'='*60}")
    print(f"  [{idx}/{len(TEST_CASES)}] {year} {make} {model} @ {mileage:,} mi")
    print(f"{'='*60}")
    total_start = time.time()
    timings = {"vehicle": f"{year} {make} {model}", "mileage": mileage}

    # Phase 1: Scraping
    t0 = time.time()
    from scrapers.nhtsa import NHTSAScraper
    from scrapers.carcomplaints import CarComplaintsScraper
    results = []
    for name, cls in [("NHTSA", NHTSAScraper), ("CarComplaints", CarComplaintsScraper)]:
        try:
            data = cls().fetch(make, model, year)
            results.append(data)
        except Exception:
            pass
    timings["scraping"] = round(time.time() - t0, 1)

    if not results:
        print("  SKIPPED - all scrapers failed")
        continue

    # Phase 2: Analysis
    t0 = time.time()
    from analysis.aggregator import aggregate
    from analysis.mileage_model import analyze_mileage, _failure_curve_cache
    from analysis.scorer import score_vehicle, _cached_weights, _cached_baseline
    _failure_curve_cache.clear()
    _cached_weights.clear()
    _cached_baseline.clear()
    agg = aggregate(results)
    ma = analyze_mileage(agg, mileage)
    vs = score_vehicle(ma, make=make, model=model, year=year,
                       num_recalls=len(agg.recalls))
    timings["analysis"] = round(time.time() - t0, 1)
    timings["problems"] = len(agg.problems)

    # Phase 3: Bulk stats
    t0 = time.time()
    bulk_stats = None
    try:
        from data.stats_builder import get_model_stats
        bulk_stats = get_model_stats(make, model, year)
    except Exception:
        pass
    timings["bulk_stats"] = round(time.time() - t0, 1)

    # Phase 4: Full report with LLM
    t0 = time.time()
    from reports.generator import generate_report
    report = generate_report(agg, ma, vs, mileage, bulk_stats=bulk_stats)
    timings["report_llm"] = round(time.time() - t0, 1)

    has_verdict = bool(report.get("sections", {}).get("executive_summary", {}).get("text"))
    checklist = report.get("sections", {}).get("inspection_checklist", {})
    enhanced = sum(
        1 for s in ("must_check", "recommended", "standard")
        for item in checklist.get(s, []) if item.get("llm_enhanced")
    )
    total_cl = sum(len(checklist.get(s, [])) for s in ("must_check", "recommended", "standard"))

    timings["total"] = round(time.time() - total_start, 1)
    timings["verdict"] = "YES" if has_verdict else "NO"
    timings["checklist"] = f"{enhanced}/{total_cl}"
    all_results.append(timings)

    print(f"  Scraping:       {timings['scraping']:5.1f}s")
    print(f"  Analysis:       {timings['analysis']:5.1f}s  ({timings['problems']} problems)")
    print(f"  Bulk stats:     {timings['bulk_stats']:5.1f}s")
    print(f"  Report+LLM:     {timings['report_llm']:5.1f}s")
    print(f"  -------------------------")
    print(f"  TOTAL:          {timings['total']:5.1f}s")
    print(f"  Verdict: {timings['verdict']}  |  Checklist: {timings['checklist']}")

print(f"\n\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  {'Vehicle':<25} {'Scrape':>6} {'Analyze':>7} {'Bulk':>5} {'Rpt+LLM':>7} {'TOTAL':>6}  Result")
print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*5} {'-'*7} {'-'*6}  ------")
for r in all_results:
    print(f"  {r['vehicle']:<25} {r['scraping']:5.1f}s {r['analysis']:6.1f}s {r['bulk_stats']:4.1f}s {r['report_llm']:6.1f}s {r['total']:5.1f}s  {r['verdict']} ({r['checklist']})")

totals = [r["total"] for r in all_results]
if totals:
    print(f"\n  Avg: {sum(totals)/len(totals):.1f}s  |  Min: {min(totals):.1f}s  |  Max: {max(totals):.1f}s")

with open("timing_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to timing_results.json")
