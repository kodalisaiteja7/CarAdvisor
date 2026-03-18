"""Run 100 benchmark vehicles at multiple mileage points and export to CSV.

Usage:
    python -m tests.mileage_matrix
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from database.models import init_db
from scrapers.nhtsa import NHTSAScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from analysis.scorer_v2 import score_vehicle_v2
from data.sales_data import get_sales_volume
from tests.score_benchmark import BENCHMARK_VEHICLES, TIER_LABELS

MILEAGES = [50_000, 75_000, 100_000, 125_000, 150_000, 200_000]
VEHICLE_TIMEOUT = 60


def _run_vehicle(v):
    """Fetch NHTSA data once, then score at each mileage point."""
    scraper = NHTSAScraper()
    data = scraper.fetch(v.make, v.model, v.year)
    agg = aggregate([data])
    sales_vol = get_sales_volume(v.make, v.model, v.year)
    total_complaints = agg.total_complaints if hasattr(agg, 'total_complaints') else len(agg.complaint_dates or [])

    results = {}
    prev_v1 = 0.0
    prev_v2 = 0.0
    for mi in MILEAGES:
        ma = analyze_mileage(agg, mi)
        vs = score_vehicle(
            ma, make=v.make, model=v.model, year=v.year,
            num_recalls=len(agg.recalls), sales_volume=sales_vol,
            complaint_dates=agg.complaint_dates,
            min_score=prev_v1,
        )
        prev_v1 = vs.reliability_risk_score

        v2 = score_vehicle_v2(
            nhtsa_risk_score=vs.reliability_risk_score,
            make=v.make, model=v.model, year=v.year,
            mileage=mi, min_score=prev_v2,
        )
        prev_v2 = v2.risk_score_v2

        results[mi] = (
            vs.reliability_risk_score, vs.letter_grade,
            v2.risk_score_v2, v2.letter_grade,
        )

    return sales_vol, total_complaints, results


def run_one(v):
    """Wrap with timeout."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_vehicle, v)
        try:
            return future.result(timeout=VEHICLE_TIMEOUT)
        except (FuturesTimeout, TimeoutError):
            return None, 0, {mi: (-1, "?", -1, "?") for mi in MILEAGES}
        except Exception as exc:
            logging.warning("Error on %s %s %s: %s", v.make, v.model, v.year, exc)
            return None, 0, {mi: (-1, "?", -1, "?") for mi in MILEAGES}


def main():
    init_db()
    total = len(BENCHMARK_VEHICLES)
    print(f"Running {total} vehicles x {len(MILEAGES)} mileage points...")

    headers = ["Make", "Model", "Year", "Tier", "Sales", "Complaints"]
    for mi in MILEAGES:
        k = f"{mi // 1000}k"
        headers += [f"{k}_V1", f"{k}_G1", f"{k}_V2", f"{k}_G2"]

    rows = []
    t0 = time.time()

    for i, v in enumerate(BENCHMARK_VEHICLES, 1):
        label = f"{v.make} {v.model} {v.year}"
        sys.stdout.write(f"\r  [{i:>3}/{total}] {label:<40}")
        sys.stdout.flush()

        sales_vol, total_complaints, results = run_one(v)

        row = [
            v.make, v.model, v.year,
            TIER_LABELS[v.expected_tier],
            sales_vol if sales_vol else "",
            total_complaints,
        ]
        for mi in MILEAGES:
            v1_score, v1_grade, v2_score, v2_grade = results[mi]
            row += [v1_score, v1_grade, v2_score, v2_grade]
        rows.append(row)

    sys.stdout.write("\r" + " " * 60 + "\r")

    out_path = "benchmark_mileage_matrix.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    elapsed = time.time() - t0
    print(f"Exported {len(rows)} vehicles x {len(MILEAGES)} mileages to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
