"""Benchmark for Risk Score V2.

Runs the same 100 benchmark vehicles through both scorer v1 and v2
at 75k miles and outputs a side-by-side comparison CSV.

Usage:
    python -m tests.score_benchmark_v2
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from database.models import init_db
from scrapers.nhtsa import NHTSAScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from analysis.scorer_v2 import score_vehicle_v2
from data.sales_data import get_sales_volume
from tests.score_benchmark import BENCHMARK_VEHICLES, TIER_LABELS, BenchmarkVehicle

MILEAGE = 75_000
VEHICLE_TIMEOUT = 45
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "benchmark_v2_results.csv"


def _run_one(v: BenchmarkVehicle) -> dict:
    """Run one vehicle through both scoring pipelines."""
    try:
        scraper = NHTSAScraper()
        data = scraper.fetch(v.make, v.model, v.year)
        agg = aggregate([data])
        ma = analyze_mileage(agg, MILEAGE)
        sales_vol = get_sales_volume(v.make, v.model, v.year)

        vs1 = score_vehicle(
            ma, make=v.make, model=v.model, year=v.year,
            num_recalls=len(agg.recalls), sales_volume=sales_vol,
            complaint_dates=agg.complaint_dates,
        )

        vs2 = score_vehicle_v2(
            nhtsa_risk_score=vs1.reliability_risk_score,
            make=v.make, model=v.model, year=v.year,
            mileage=MILEAGE,
        )

        return {
            "make": v.make,
            "model": v.model,
            "year": v.year,
            "tier": v.expected_tier,
            "tier_label": TIER_LABELS[v.expected_tier],
            "complaints": agg.total_complaints,
            "sales": sales_vol or "",
            "score_v1": vs1.reliability_risk_score,
            "grade_v1": vs1.letter_grade,
            "score_v2": vs2.risk_score_v2,
            "grade_v2": vs2.letter_grade,
            "nhtsa_pct": vs2.nhtsa_component,
            "tsb_score": vs2.tsb_component,
            "tsb_count": vs2.tsb_raw_count,
            "inv_score": vs2.investigation_component,
            "inv_count": vs2.inv_raw_count,
            "mfr_score": vs2.mfr_comm_component,
            "mfr_count": vs2.mfr_comm_raw_count,
            "dl_qir": vs2.dl_qir if vs2.dl_qir is not None else "",
            "dl_score": vs2.dl_qir_component,
            "error": "",
        }
    except Exception as exc:
        return {
            "make": v.make, "model": v.model, "year": v.year,
            "tier": v.expected_tier, "tier_label": TIER_LABELS[v.expected_tier],
            "complaints": 0, "sales": "",
            "score_v1": -1, "grade_v1": "?",
            "score_v2": -1, "grade_v2": "?",
            "nhtsa_pct": 0, "tsb_score": 0, "tsb_count": 0,
            "inv_score": 0, "inv_count": 0,
            "mfr_score": 0, "mfr_count": 0,
            "dl_qir": "", "dl_score": 0,
            "error": str(exc)[:80],
        }


def run_with_timeout(v: BenchmarkVehicle) -> dict:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_one, v)
        try:
            return future.result(timeout=VEHICLE_TIMEOUT)
        except (FuturesTimeout, TimeoutError):
            return {
                "make": v.make, "model": v.model, "year": v.year,
                "tier": v.expected_tier, "tier_label": TIER_LABELS[v.expected_tier],
                "complaints": 0, "sales": "",
                "score_v1": -1, "grade_v1": "?",
                "score_v2": -1, "grade_v2": "?",
                "nhtsa_pct": 0, "tsb_score": 0, "tsb_count": 0,
                "inv_score": 0, "inv_count": 0,
                "mfr_score": 0, "mfr_count": 0,
                "dl_qir": "", "dl_score": 0,
                "error": f"Timeout ({VEHICLE_TIMEOUT}s)",
            }


def main():
    init_db()
    total = len(BENCHMARK_VEHICLES)
    print(f"Risk Score V2 Benchmark: {total} vehicles at {MILEAGE:,} miles")
    print("=" * 80)

    results: list[dict] = []
    t0 = time.time()

    for i, v in enumerate(BENCHMARK_VEHICLES, 1):
        label = f"{v.make} {v.model} {v.year}"
        sys.stdout.write(f"\r  [{i:>3}/{total}] {label:<40}")
        sys.stdout.flush()
        result = run_with_timeout(v)
        results.append(result)

    elapsed = time.time() - t0
    print(f"\n\nCompleted in {elapsed:.0f}s")

    headers = [
        "Make", "Model", "Year", "Tier", "TierLabel", "Complaints", "Sales",
        "Score_V1", "Grade_V1", "Score_V2", "Grade_V2",
        "NHTSA_Pct", "TSB_Score", "TSB_Count", "Inv_Score", "Inv_Count",
        "Mfr_Score", "Mfr_Count", "DL_QIR", "DL_Score", "Error",
    ]
    keys = [
        "make", "model", "year", "tier", "tier_label", "complaints", "sales",
        "score_v1", "grade_v1", "score_v2", "grade_v2",
        "nhtsa_pct", "tsb_score", "tsb_count", "inv_score", "inv_count",
        "mfr_score", "mfr_count", "dl_qir", "dl_score", "error",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([r[k] for k in keys])

    print(f"CSV: {OUTPUT_CSV}")

    ok = [r for r in results if not r["error"]]
    err = [r for r in results if r["error"]]

    print(f"\nResults: {len(ok)} ok, {len(err)} errors")

    if ok:
        _print_summary(ok)

    if err:
        print(f"\nErrors ({len(err)}):")
        for r in err:
            print(f"  {r['make']} {r['model']} {r['year']}: {r['error']}")


def _print_summary(results: list[dict]):
    """Print a tier-by-tier comparison."""
    print("\n" + "=" * 110)
    print(f"{'VEHICLE':<35} {'TIER':>8} {'V1':>6} {'G1':>3} {'V2':>6} {'G2':>3} {'TSB':>6} {'INV':>6} {'MFR':>6} {'DL':>6}")
    print("=" * 110)

    for tier in range(1, 6):
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue

        tier_results.sort(key=lambda x: x["score_v2"])
        print(f"\n-- {TIER_LABELS[tier]} --")

        for r in tier_results:
            label = f"{r['make']} {r['model']} {r['year']}"
            print(
                f"  {label:<33} {TIER_LABELS[r['tier']]:>8} "
                f"{r['score_v1']:>6.1f} {r['grade_v1']:>3} "
                f"{r['score_v2']:>6.1f} {r['grade_v2']:>3} "
                f"{r['tsb_score']:>6.1f} {r['inv_score']:>6.1f} "
                f"{r['mfr_score']:>6.1f} {r['dl_score']:>6.1f}"
            )

        v1_avg = sum(r["score_v1"] for r in tier_results) / len(tier_results)
        v2_avg = sum(r["score_v2"] for r in tier_results) / len(tier_results)
        print(f"  {'AVG':<33} {'':>8} {v1_avg:>6.1f} {'':>3} {v2_avg:>6.1f}")

    print("\n" + "=" * 110)
    v1_scores = [r["score_v1"] for r in results]
    v2_scores = [r["score_v2"] for r in results]
    tiers = [float(r["tier"]) for r in results]

    from tests.score_benchmark import spearman_rank_correlation

    rho_v1 = spearman_rank_correlation(tiers, v1_scores)
    rho_v2 = spearman_rank_correlation(tiers, v2_scores)

    print(f"  Spearman correlation (tier vs V1): {rho_v1:.3f}")
    print(f"  Spearman correlation (tier vs V2): {rho_v2:.3f}")
    print(f"  Improvement: {rho_v2 - rho_v1:+.3f}")

    print(f"\n  V1 range: {min(v1_scores):.1f} - {max(v1_scores):.1f}")
    print(f"  V2 range: {min(v2_scores):.1f} - {max(v2_scores):.1f}")

    avg_shift = sum(v2 - v1 for v1, v2 in zip(v1_scores, v2_scores)) / len(results)
    print(f"  Average V2-V1 shift: {avg_shift:+.1f}")


if __name__ == "__main__":
    main()
