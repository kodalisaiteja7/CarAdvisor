"""Benchmark for low-sales, high-end / luxury vehicles.

100 premium vehicles scored at 9 mileage points (V1 + V2).
This stress-tests the scorer on low-volume vehicles where
sales normalization and sparse complaint data are challenges.

Usage:
    python -m tests.luxury_benchmark
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from database.models import init_db
from scrapers.nhtsa import NHTSAScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from analysis.scorer_v2 import score_vehicle_v2
from data.sales_data import get_sales_volume


@dataclass
class BenchmarkVehicle:
    make: str
    model: str
    year: int
    expected_tier: int
    note: str = ""


TIER_LABELS = {1: "Reliable", 2: "Above Avg", 3: "Average", 4: "Below Avg", 5: "Unreliable"}

LUXURY_VEHICLES: list[BenchmarkVehicle] = [
    # ── Tier 1: Reliable luxury (Porsche, Lexus flagships, Genesis) ──
    BenchmarkVehicle("PORSCHE", "911", 2018, 1, "CR/JD Power top sports car"),
    BenchmarkVehicle("PORSCHE", "MACAN", 2019, 1, "Best small luxury SUV reliability"),
    BenchmarkVehicle("PORSCHE", "CAYENNE", 2019, 1, "Strong luxury SUV"),
    BenchmarkVehicle("PORSCHE", "PANAMERA", 2018, 1, "Reliable grand tourer"),
    BenchmarkVehicle("PORSCHE", "911", 2017, 1, "991.2 proven platform"),
    BenchmarkVehicle("PORSCHE", "CAYENNE", 2018, 1, "Mature generation"),
    BenchmarkVehicle("PORSCHE", "MACAN", 2018, 1, "Consistent reliability"),
    BenchmarkVehicle("LEXUS", "LC", 2018, 1, "Halo coupe, Toyota DNA"),
    BenchmarkVehicle("LEXUS", "LS", 2018, 1, "Flagship sedan, few issues"),
    BenchmarkVehicle("LEXUS", "GS", 2018, 1, "Reliable sport sedan"),
    BenchmarkVehicle("LEXUS", "RC", 2018, 1, "Reliable coupe"),
    BenchmarkVehicle("LEXUS", "LX", 2018, 1, "Land Cruiser based, legendary"),
    BenchmarkVehicle("GENESIS", "G80", 2018, 1, "Excellent debut reliability"),
    BenchmarkVehicle("GENESIS", "G90", 2018, 1, "Low-volume flagship, few issues"),
    BenchmarkVehicle("GENESIS", "G80", 2019, 1, "Continued strong reliability"),
    BenchmarkVehicle("TOYOTA", "LAND CRUISER", 2018, 1, "Ultra durable, low volume US"),
    BenchmarkVehicle("ACURA", "RLX", 2018, 1, "Honda luxury, very few complaints"),
    BenchmarkVehicle("ACURA", "MDX", 2019, 1, "Honda-based luxury SUV"),
    BenchmarkVehicle("ACURA", "NSX", 2017, 1, "Supercar, minimal complaints"),
    BenchmarkVehicle("LEXUS", "GX", 2019, 1, "4Runner DNA, bulletproof"),

    # ── Tier 2: Above average luxury ──
    BenchmarkVehicle("VOLVO", "S60", 2019, 2, "Decent luxury sedan"),
    BenchmarkVehicle("VOLVO", "XC90", 2018, 2, "Good but complex powertrain"),
    BenchmarkVehicle("VOLVO", "XC40", 2019, 2, "Solid small luxury SUV"),
    BenchmarkVehicle("VOLVO", "S90", 2018, 2, "Low volume, few complaints"),
    BenchmarkVehicle("AUDI", "A5", 2018, 2, "Coupe, above average"),
    BenchmarkVehicle("AUDI", "Q7", 2019, 2, "Improved generation"),
    BenchmarkVehicle("AUDI", "A3", 2018, 2, "Entry luxury, fewer issues"),
    BenchmarkVehicle("BMW", "4 SERIES", 2018, 2, "Sporty, above average"),
    BenchmarkVehicle("BMW", "Z4", 2020, 2, "Simple sports car"),
    BenchmarkVehicle("BMW", "X4", 2019, 2, "Niche SUV, decent"),
    BenchmarkVehicle("MERCEDES-BENZ", "GLC", 2019, 2, "Decent compact luxury SUV"),
    BenchmarkVehicle("LINCOLN", "AVIATOR", 2020, 2, "Strong debut reliability"),
    BenchmarkVehicle("LINCOLN", "CORSAIR", 2020, 2, "Solid new model"),
    BenchmarkVehicle("LINCOLN", "NAUTILUS", 2019, 2, "Decent crossover"),
    BenchmarkVehicle("CADILLAC", "CT5", 2020, 2, "Good new platform"),
    BenchmarkVehicle("CADILLAC", "XT4", 2019, 2, "Newer, good start"),
    BenchmarkVehicle("INFINITI", "Q60", 2018, 2, "Niche coupe, decent"),
    BenchmarkVehicle("ACURA", "RDX", 2019, 2, "Refreshed, strong"),
    BenchmarkVehicle("GENESIS", "G70", 2020, 2, "Sporty, reliable luxury"),
    BenchmarkVehicle("LEXUS", "IS", 2019, 2, "Mostly reliable sport sedan"),

    # ── Tier 3: Average luxury ──
    BenchmarkVehicle("BMW", "7 SERIES", 2018, 3, "Complex flagship, average"),
    BenchmarkVehicle("BMW", "X2", 2018, 3, "Average small luxury SUV"),
    BenchmarkVehicle("BMW", "X6", 2018, 3, "Niche coupe SUV, average"),
    BenchmarkVehicle("BMW", "X7", 2019, 3, "New large SUV, average start"),
    BenchmarkVehicle("MERCEDES-BENZ", "S-CLASS", 2018, 3, "Complex flagship, average"),
    BenchmarkVehicle("MERCEDES-BENZ", "GLE", 2019, 3, "Average for class"),
    BenchmarkVehicle("MERCEDES-BENZ", "CLA", 2019, 3, "Entry compact, average"),
    BenchmarkVehicle("MERCEDES-BENZ", "GLA", 2019, 3, "Small SUV, average"),
    BenchmarkVehicle("MERCEDES-BENZ", "CLS", 2018, 3, "Four-door coupe, average"),
    BenchmarkVehicle("AUDI", "A8", 2019, 3, "Complex flagship"),
    BenchmarkVehicle("AUDI", "Q8", 2019, 3, "New coupe SUV, average"),
    BenchmarkVehicle("CADILLAC", "ESCALADE", 2018, 3, "Average large luxury SUV"),
    BenchmarkVehicle("CADILLAC", "CT6", 2018, 3, "Mixed reliability reviews"),
    BenchmarkVehicle("INFINITI", "QX80", 2018, 3, "Aging Nissan platform"),
    BenchmarkVehicle("INFINITI", "QX50", 2019, 3, "New VC-Turbo, mixed"),
    BenchmarkVehicle("LINCOLN", "NAVIGATOR", 2018, 3, "Average for class"),
    BenchmarkVehicle("BUICK", "REGAL", 2018, 3, "Low volume, average"),
    BenchmarkVehicle("VOLVO", "V60", 2019, 3, "Wagon, average"),
    BenchmarkVehicle("TESLA", "MODEL 3", 2019, 3, "Mixed build quality"),
    BenchmarkVehicle("ACURA", "TLX", 2017, 3, "Older year, average"),

    # ── Tier 4: Below average luxury ──
    BenchmarkVehicle("JAGUAR", "F-PACE", 2018, 4, "Electrical/infotainment issues"),
    BenchmarkVehicle("JAGUAR", "XE", 2018, 4, "Electrical problems"),
    BenchmarkVehicle("JAGUAR", "XF", 2017, 4, "Aging, known issues"),
    BenchmarkVehicle("LAND ROVER", "DISCOVERY", 2018, 4, "Typical LR electrical issues"),
    BenchmarkVehicle("LAND ROVER", "DISCOVERY SPORT", 2018, 4, "Below average reliability"),
    BenchmarkVehicle("LAND ROVER", "RANGE ROVER", 2018, 4, "Electrical gremlins"),
    BenchmarkVehicle("BMW", "X5", 2018, 4, "Complex, expensive repairs"),
    BenchmarkVehicle("BMW", "3 SERIES", 2018, 4, "Oil leaks, electronics"),
    BenchmarkVehicle("MERCEDES-BENZ", "C-CLASS", 2017, 4, "Complex electronics"),
    BenchmarkVehicle("MERCEDES-BENZ", "GLS", 2018, 4, "Complex, below average"),
    BenchmarkVehicle("CADILLAC", "CTS", 2017, 4, "Reliability concerns"),
    BenchmarkVehicle("CADILLAC", "SRX", 2016, 4, "Aging platform, issues"),
    BenchmarkVehicle("AUDI", "Q5", 2017, 4, "Oil consumption issues"),
    BenchmarkVehicle("TESLA", "MODEL X", 2018, 4, "Falcon doors, drivetrain"),
    BenchmarkVehicle("TESLA", "MODEL S", 2018, 4, "MCU, drivetrain concerns"),
    BenchmarkVehicle("LINCOLN", "MKZ", 2017, 4, "Below average sedan"),
    BenchmarkVehicle("VOLVO", "XC90", 2016, 4, "First-year turbo issues"),
    BenchmarkVehicle("VOLVO", "XC60", 2017, 4, "First-year new gen issues"),
    BenchmarkVehicle("INFINITI", "QX70", 2017, 4, "Aging, transmission issues"),
    BenchmarkVehicle("MASERATI", "GHIBLI", 2018, 4, "Expensive, poor reliability"),

    # ── Tier 5: Unreliable luxury ──
    BenchmarkVehicle("MASERATI", "QUATTROPORTE", 2016, 5, "Terrible JD Power scores"),
    BenchmarkVehicle("MASERATI", "LEVANTE", 2017, 5, "Worst new luxury SUV"),
    BenchmarkVehicle("MASERATI", "GHIBLI", 2015, 5, "Very poor reliability"),
    BenchmarkVehicle("MASERATI", "GHIBLI", 2017, 5, "Continued reliability problems"),
    BenchmarkVehicle("MASERATI", "LEVANTE", 2018, 5, "Still unreliable"),
    BenchmarkVehicle("ALFA ROMEO", "GIULIA", 2017, 5, "Very poor CR reliability"),
    BenchmarkVehicle("ALFA ROMEO", "STELVIO", 2018, 5, "Poor reliability ratings"),
    BenchmarkVehicle("JAGUAR", "F-TYPE", 2016, 5, "Known drivetrain problems"),
    BenchmarkVehicle("JAGUAR", "XJ", 2016, 5, "Aging, chronic electrical"),
    BenchmarkVehicle("JAGUAR", "E-PACE", 2018, 5, "First year quality problems"),
    BenchmarkVehicle("LAND ROVER", "RANGE ROVER EVOQUE", 2016, 5, "Chronic unreliability"),
    BenchmarkVehicle("LAND ROVER", "DISCOVERY SPORT", 2016, 5, "Very poor reliability"),
    BenchmarkVehicle("LAND ROVER", "RANGE ROVER SPORT", 2015, 5, "Chronic issues"),
    BenchmarkVehicle("LAND ROVER", "RANGE ROVER", 2016, 5, "Electrical nightmare"),
    BenchmarkVehicle("LAND ROVER", "DISCOVERY", 2017, 5, "First year redesign problems"),
    BenchmarkVehicle("TESLA", "MODEL X", 2017, 5, "Early production falcon doors"),
    BenchmarkVehicle("TESLA", "MODEL S", 2016, 5, "MCU failures, drivetrain"),
    BenchmarkVehicle("BMW", "X3", 2015, 5, "Timing chain era"),
    BenchmarkVehicle("INFINITI", "QX60", 2016, 5, "CVT transmission failures"),
    BenchmarkVehicle("JAGUAR", "XF", 2015, 5, "Severe electrical issues"),
]

MILEAGES = [10_000, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000]
VEHICLE_TIMEOUT = 60


def _run_vehicle(v):
    scraper = NHTSAScraper()
    data = scraper.fetch(v.make, v.model, v.year)
    agg = aggregate([data])
    sales_vol = get_sales_volume(v.make, v.model, v.year)
    total_complaints = agg.total_complaints if hasattr(agg, "total_complaints") else len(agg.complaint_dates or [])

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
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_vehicle, v)
        try:
            return future.result(timeout=VEHICLE_TIMEOUT)
        except (FuturesTimeout, TimeoutError):
            return None, 0, {mi: (-1, "?", -1, "?") for mi in MILEAGES}
        except Exception as exc:
            logging.warning("Error on %s %s %s: %s", v.make, v.model, v.year, exc)
            return None, 0, {mi: (-1, "?", -1, "?") for mi in MILEAGES}


def spearman_rank_correlation(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    def _ranks(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_idx[j]] == vals[sorted_idx[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks
    rx, ry = _ranks(x), _ranks(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def main():
    init_db()
    total = len(LUXURY_VEHICLES)
    print(f"Luxury benchmark: {total} high-end vehicles x {len(MILEAGES)} mileage points")
    print("=" * 80)

    headers = ["Make", "Model", "Year", "Tier", "TierLabel", "Sales", "Complaints"]
    for mi in MILEAGES:
        k = f"{mi // 1000}k"
        headers += [f"{k}_V1", f"{k}_G1", f"{k}_V2", f"{k}_G2"]

    rows = []
    all_results = []
    t0 = time.time()

    for i, v in enumerate(LUXURY_VEHICLES, 1):
        label = f"{v.make} {v.model} {v.year}"
        sys.stdout.write(f"\r  [{i:>3}/{total}] {label:<45}")
        sys.stdout.flush()

        sales_vol, total_complaints, results = run_one(v)

        row = [
            v.make, v.model, v.year,
            v.expected_tier,
            TIER_LABELS[v.expected_tier],
            sales_vol if sales_vol else "",
            total_complaints,
        ]
        for mi in MILEAGES:
            v1_score, v1_grade, v2_score, v2_grade = results[mi]
            row += [v1_score, v1_grade, v2_score, v2_grade]
        rows.append(row)

        v1_75k = results[75_000][0]
        v2_75k = results[75_000][2]
        all_results.append({
            "make": v.make, "model": v.model, "year": v.year,
            "tier": v.expected_tier, "score_v1": v1_75k, "score_v2": v2_75k,
            "complaints": total_complaints, "sales": sales_vol,
        })

    sys.stdout.write("\r" + " " * 60 + "\r")
    elapsed = time.time() - t0

    out_path = "luxury_benchmark_mileage.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    print(f"Exported {len(rows)} vehicles x {len(MILEAGES)} mileages to {out_path}")
    print(f"Total time: {elapsed:.1f}s")

    ok = [r for r in all_results if r["score_v1"] >= 0]
    if ok:
        _print_summary(ok)


def _print_summary(results: list[dict]):
    print("\n" + "=" * 100)
    print(f"  LUXURY BENCHMARK SUMMARY (at 75k miles)")
    print("=" * 100)
    print(f"  {'VEHICLE':<40} {'TIER':>10} {'CMPL':>6} {'SALES':>8} {'V1':>6} {'V2':>6}")
    print("-" * 100)

    for tier in range(1, 6):
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue
        tier_results.sort(key=lambda x: x["score_v2"])
        print(f"\n  -- {TIER_LABELS[tier]} --")
        for r in tier_results:
            label = f"{r['make']} {r['model']} {r['year']}"
            sales_str = f"{r['sales']:,}" if r['sales'] else "N/A"
            print(f"    {label:<38} {TIER_LABELS[r['tier']]:>10} "
                  f"{r['complaints']:>6} {sales_str:>8} "
                  f"{r['score_v1']:>6.1f} {r['score_v2']:>6.1f}")
        v1_avg = sum(r["score_v1"] for r in tier_results) / len(tier_results)
        v2_avg = sum(r["score_v2"] for r in tier_results) / len(tier_results)
        print(f"    {'AVG':<38} {'':>10} {'':>6} {'':>8} {v1_avg:>6.1f} {v2_avg:>6.1f}")

    print("\n" + "=" * 100)
    v1_scores = [r["score_v1"] for r in results]
    v2_scores = [r["score_v2"] for r in results]
    tiers = [float(r["tier"]) for r in results]

    rho_v1 = spearman_rank_correlation(tiers, v1_scores)
    rho_v2 = spearman_rank_correlation(tiers, v2_scores)

    print(f"  Spearman correlation (tier vs V1): {rho_v1:.3f}")
    print(f"  Spearman correlation (tier vs V2): {rho_v2:.3f}")
    print(f"  Improvement: {rho_v2 - rho_v1:+.3f}")

    low_sales = [r for r in results if not r["sales"] or r["sales"] == 0]
    print(f"\n  Vehicles with NO sales data: {len(low_sales)} / {len(results)}")
    if low_sales:
        ls_v1 = sum(r["score_v1"] for r in low_sales) / len(low_sales)
        ls_v2 = sum(r["score_v2"] for r in low_sales) / len(low_sales)
        print(f"  No-sales-data avg V1: {ls_v1:.1f}  avg V2: {ls_v2:.1f}")

    print(f"\n  V1 range: {min(v1_scores):.1f} - {max(v1_scores):.1f}")
    print(f"  V2 range: {min(v2_scores):.1f} - {max(v2_scores):.1f}")


if __name__ == "__main__":
    main()
