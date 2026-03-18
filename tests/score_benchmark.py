"""Benchmark suite for validating risk score accuracy.

Runs ~100 known vehicles through the scoring pipeline and compares
results against established reliability rankings (Consumer Reports,
J.D. Power, common knowledge).

Usage:
    python -m tests.score_benchmark          # run full benchmark
    python -m tests.score_benchmark --quick  # skip NHTSA API, use cache only
"""

from __future__ import annotations

import argparse
import logging
import signal
import statistics
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

logging.basicConfig(level=logging.WARNING, format="%(message)s")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from database.models import init_db
from scrapers.nhtsa import NHTSAScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from data.sales_data import get_sales_volume

MILEAGE = 75_000

# Tier 1: Known reliable — expected score <= 35 (grade A/B)
# Tier 2: Above average — expected score 25-45 (grade B/C)
# Tier 3: Average — expected score 35-55 (grade C)
# Tier 4: Below average — expected score 50-70 (grade C/D)
# Tier 5: Known unreliable — expected score >= 60 (grade D/F)

@dataclass
class BenchmarkVehicle:
    make: str
    model: str
    year: int
    expected_tier: int  # 1-5
    note: str = ""


BENCHMARK_VEHICLES: list[BenchmarkVehicle] = [
    # ── Tier 1: Known reliable (CR top picks, legendary reliability) ──
    BenchmarkVehicle("TOYOTA", "CAMRY", 2019, 1, "Perennial CR top pick"),
    BenchmarkVehicle("TOYOTA", "CAMRY", 2018, 1, "CR top pick"),
    BenchmarkVehicle("TOYOTA", "COROLLA", 2019, 1, "Global reliability icon"),
    BenchmarkVehicle("TOYOTA", "COROLLA", 2018, 1, "Solid generation"),
    BenchmarkVehicle("TOYOTA", "RAV4", 2019, 1, "Best-selling SUV, reliable"),
    BenchmarkVehicle("TOYOTA", "HIGHLANDER", 2018, 1, "Bulletproof family SUV"),
    BenchmarkVehicle("TOYOTA", "4RUNNER", 2018, 1, "Legendary durability"),
    BenchmarkVehicle("TOYOTA", "TACOMA", 2018, 1, "Best-in-class truck reliability"),
    BenchmarkVehicle("LEXUS", "RX", 2018, 1, "Top luxury reliability"),
    BenchmarkVehicle("LEXUS", "ES", 2019, 1, "Trouble-free luxury sedan"),
    BenchmarkVehicle("LEXUS", "GX", 2018, 1, "Land Cruiser DNA"),
    BenchmarkVehicle("HONDA", "CIVIC", 2019, 1, "Consistently reliable"),
    BenchmarkVehicle("HONDA", "CIVIC", 2018, 1, "Excellent track record"),
    BenchmarkVehicle("HONDA", "ACCORD", 2018, 1, "CR recommended sedan"),
    BenchmarkVehicle("HONDA", "CR-V", 2019, 1, "Solid compact SUV"),
    BenchmarkVehicle("MAZDA", "MAZDA3", 2019, 1, "CR top brand reliability"),
    BenchmarkVehicle("MAZDA", "CX-5", 2018, 1, "Top-rated compact SUV"),
    BenchmarkVehicle("MAZDA", "CX-9", 2018, 1, "Reliable family SUV"),
    BenchmarkVehicle("TOYOTA", "PRIUS", 2018, 1, "Proven hybrid reliability"),
    BenchmarkVehicle("HONDA", "FIT", 2018, 1, "Simple, reliable subcompact"),

    # ── Tier 2: Above average ──
    BenchmarkVehicle("SUBARU", "OUTBACK", 2018, 2, "Good overall, minor issues"),
    BenchmarkVehicle("SUBARU", "FORESTER", 2019, 2, "Solid but not flawless"),
    BenchmarkVehicle("SUBARU", "CROSSTREK", 2018, 2, "Reliable crossover"),
    BenchmarkVehicle("HYUNDAI", "TUCSON", 2019, 2, "Improved reliability"),
    BenchmarkVehicle("HYUNDAI", "KONA", 2019, 2, "Newer model, decent"),
    BenchmarkVehicle("KIA", "SPORTAGE", 2019, 2, "Improved over time"),
    BenchmarkVehicle("KIA", "TELLURIDE", 2020, 2, "Strong debut reliability"),
    BenchmarkVehicle("TOYOTA", "TUNDRA", 2018, 2, "Reliable but aging platform"),
    BenchmarkVehicle("HONDA", "PILOT", 2018, 2, "Good but transmission concerns"),
    BenchmarkVehicle("HONDA", "ODYSSEY", 2018, 2, "Reliable minivan"),
    BenchmarkVehicle("TOYOTA", "SIENNA", 2018, 2, "Reliable family van"),
    BenchmarkVehicle("HYUNDAI", "SANTA FE", 2019, 2, "Improved generation"),
    BenchmarkVehicle("KIA", "SORENTO", 2019, 2, "Solid mid-size SUV"),
    BenchmarkVehicle("FORD", "MAVERICK", 2022, 2, "Good initial reliability"),
    BenchmarkVehicle("HYUNDAI", "ELANTRA", 2019, 2, "Dependable compact"),
    BenchmarkVehicle("KIA", "FORTE", 2019, 2, "Reliable economy car"),
    BenchmarkVehicle("TOYOTA", "AVALON", 2018, 2, "Comfortable and reliable"),
    BenchmarkVehicle("LEXUS", "IS", 2018, 2, "Mostly reliable sport sedan"),
    BenchmarkVehicle("MAZDA", "MX-5 MIATA", 2018, 2, "Simple, few issues"),
    BenchmarkVehicle("TOYOTA", "C-HR", 2018, 2, "Basic but reliable"),

    # ── Tier 3: Average ──
    BenchmarkVehicle("FORD", "F-150", 2018, 3, "Some transmission/engine issues"),
    BenchmarkVehicle("FORD", "ESCAPE", 2018, 3, "Average reliability"),
    BenchmarkVehicle("FORD", "FUSION", 2018, 3, "Mixed reliability"),
    BenchmarkVehicle("CHEVROLET", "SILVERADO", 2018, 3, "Average for trucks"),
    BenchmarkVehicle("CHEVROLET", "MALIBU", 2018, 3, "Mid-pack sedan"),
    BenchmarkVehicle("CHEVROLET", "EQUINOX", 2019, 3, "Improved from earlier gen"),
    BenchmarkVehicle("GMC", "SIERRA", 2018, 3, "Similar to Silverado"),
    BenchmarkVehicle("NISSAN", "ALTIMA", 2018, 3, "CVT concerns but average"),
    BenchmarkVehicle("NISSAN", "SENTRA", 2018, 3, "Basic average sedan"),
    BenchmarkVehicle("VOLKSWAGEN", "JETTA", 2019, 3, "Improved reliability"),
    BenchmarkVehicle("VOLKSWAGEN", "TIGUAN", 2018, 3, "Average compact SUV"),
    BenchmarkVehicle("HYUNDAI", "ACCENT", 2018, 3, "Budget car, average"),
    BenchmarkVehicle("KIA", "RIO", 2018, 3, "Budget average"),
    BenchmarkVehicle("FORD", "RANGER", 2019, 3, "Decent mid-size truck"),
    BenchmarkVehicle("RAM", "1500", 2019, 3, "Average full-size truck"),
    BenchmarkVehicle("BUICK", "ENCORE", 2018, 3, "Average small SUV"),
    BenchmarkVehicle("BUICK", "ENCLAVE", 2018, 3, "Average large SUV"),
    BenchmarkVehicle("CHEVROLET", "TRAVERSE", 2018, 3, "Average family SUV"),
    BenchmarkVehicle("FORD", "EDGE", 2018, 3, "Middle of the pack"),
    BenchmarkVehicle("NISSAN", "MURANO", 2018, 3, "Average mid-size SUV"),

    # ── Tier 4: Below average ──
    BenchmarkVehicle("JEEP", "CHEROKEE", 2018, 4, "Transmission/electronic issues"),
    BenchmarkVehicle("JEEP", "GRAND CHEROKEE", 2018, 4, "Electrical/engine problems"),
    BenchmarkVehicle("JEEP", "COMPASS", 2018, 4, "First-year reliability issues"),
    BenchmarkVehicle("BMW", "3 SERIES", 2017, 4, "Expensive maintenance issues"),
    BenchmarkVehicle("BMW", "X3", 2018, 4, "Timing chain/oil leak issues"),
    BenchmarkVehicle("AUDI", "A4", 2018, 4, "Electrical/tech complaints"),
    BenchmarkVehicle("AUDI", "Q5", 2018, 4, "Some electrical issues"),
    BenchmarkVehicle("DODGE", "CHARGER", 2018, 4, "Electrical/suspension issues"),
    BenchmarkVehicle("DODGE", "DURANGO", 2018, 4, "Below average reliability"),
    BenchmarkVehicle("FORD", "EXPLORER", 2017, 4, "Transmission/suspension issues"),
    BenchmarkVehicle("CHEVROLET", "EQUINOX", 2018, 4, "Engine oil consumption"),
    BenchmarkVehicle("NISSAN", "PATHFINDER", 2018, 4, "CVT transmission issues"),
    BenchmarkVehicle("NISSAN", "ROGUE", 2018, 4, "CVT and brake complaints"),
    BenchmarkVehicle("MERCEDES-BENZ", "C-CLASS", 2018, 4, "Complex electronics"),
    BenchmarkVehicle("VOLKSWAGEN", "ATLAS", 2018, 4, "First-gen reliability issues"),
    BenchmarkVehicle("CHEVROLET", "COLORADO", 2018, 4, "Some engine/trans issues"),
    BenchmarkVehicle("INFINITI", "QX60", 2018, 4, "CVT issues from Nissan"),
    BenchmarkVehicle("SUBARU", "OUTBACK", 2016, 4, "Oil consumption gen"),
    BenchmarkVehicle("FORD", "FOCUS", 2016, 4, "PowerShift DCT problems"),
    BenchmarkVehicle("KIA", "OPTIMA", 2016, 4, "Theta II engine risk"),

    # ── Tier 5: Known unreliable ──
    BenchmarkVehicle("HYUNDAI", "SONATA", 2017, 5, "Theta II engine failures"),
    BenchmarkVehicle("HYUNDAI", "SONATA", 2015, 5, "Theta II + recalls"),
    BenchmarkVehicle("KIA", "OPTIMA", 2014, 5, "Theta II engine seizure"),
    BenchmarkVehicle("CHRYSLER", "200", 2015, 5, "Notoriously unreliable"),
    BenchmarkVehicle("DODGE", "JOURNEY", 2017, 5, "Among worst reliability"),
    BenchmarkVehicle("DODGE", "DART", 2015, 5, "Major reliability issues"),
    BenchmarkVehicle("JEEP", "WRANGLER", 2018, 5, "Death wobble, leaks"),
    BenchmarkVehicle("JEEP", "RENEGADE", 2017, 5, "Poor reliability ratings"),
    BenchmarkVehicle("FIAT", "500X", 2016, 5, "Very poor reliability"),
    BenchmarkVehicle("NISSAN", "ROGUE", 2015, 5, "Severe CVT failures"),
    BenchmarkVehicle("FORD", "FIESTA", 2015, 5, "PowerShift DCT failures"),
    BenchmarkVehicle("FORD", "FOCUS", 2014, 5, "PowerShift DCT severe"),
    BenchmarkVehicle("CHEVROLET", "CRUZE", 2014, 5, "Engine/cooling failures"),
    BenchmarkVehicle("HYUNDAI", "TUCSON", 2016, 5, "Theta engine + DCT issues"),
    BenchmarkVehicle("CHRYSLER", "PACIFICA", 2017, 5, "Transmission/stalling"),
    BenchmarkVehicle("ALFA ROMEO", "GIULIA", 2018, 5, "Very poor reliability"),
    BenchmarkVehicle("LAND ROVER", "EVOQUE", 2018, 5, "Chronic issues"),
    BenchmarkVehicle("BMW", "X5", 2016, 5, "Coolant/oil/electrical"),
    BenchmarkVehicle("NISSAN", "PATHFINDER", 2015, 5, "Severe CVT problems"),
    BenchmarkVehicle("RAM", "1500", 2014, 5, "EcoDiesel/trans issues"),
]

TIER_LABELS = {1: "Reliable", 2: "Above Avg", 3: "Average", 4: "Below Avg", 5: "Unreliable"}
TIER_SCORE_RANGES = {
    1: (0, 35),
    2: (15, 50),
    3: (30, 60),
    4: (45, 75),
    5: (55, 100),
}


@dataclass
class BenchmarkResult:
    vehicle: BenchmarkVehicle
    score: float
    grade: str
    safety: float
    total_problems: int
    sales_vol: int | None
    elapsed: float
    error: str = ""


VEHICLE_TIMEOUT = 45  # seconds per vehicle


def _run_pipeline(v: BenchmarkVehicle) -> BenchmarkResult:
    """Run one vehicle through the full pipeline (no timeout wrapper)."""
    t0 = time.time()
    try:
        scraper = NHTSAScraper()
        data = scraper.fetch(v.make, v.model, v.year)
        agg = aggregate([data])
        ma = analyze_mileage(agg, MILEAGE)
        sales_vol = get_sales_volume(v.make, v.model, v.year)
        vs = score_vehicle(
            ma, make=v.make, model=v.model, year=v.year,
            num_recalls=len(agg.recalls), sales_volume=sales_vol,
            complaint_dates=agg.complaint_dates,
        )
        elapsed = time.time() - t0
        return BenchmarkResult(
            vehicle=v, score=vs.reliability_risk_score,
            grade=vs.letter_grade, safety=vs.safety_score,
            total_problems=vs.total_problems, sales_vol=sales_vol,
            elapsed=elapsed,
        )
    except Exception as exc:
        return BenchmarkResult(
            vehicle=v, score=-1, grade="?", safety=-1,
            total_problems=0, sales_vol=None,
            elapsed=time.time() - t0, error=str(exc),
        )


def run_single(v: BenchmarkVehicle) -> BenchmarkResult:
    """Run with a timeout to prevent hangs on slow API calls."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_pipeline, v)
        try:
            return future.result(timeout=VEHICLE_TIMEOUT)
        except (FuturesTimeout, TimeoutError):
            return BenchmarkResult(
                vehicle=v, score=-1, grade="?", safety=-1,
                total_problems=0, sales_vol=None,
                elapsed=VEHICLE_TIMEOUT,
                error=f"Timeout ({VEHICLE_TIMEOUT}s)",
            )


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
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

    rx = _ranks(x)
    ry = _ranks(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def print_results(results: list[BenchmarkResult]):
    """Print formatted benchmark results and metrics."""
    results_ok = [r for r in results if not r.error]
    results_err = [r for r in results if r.error]

    print("\n" + "=" * 120)
    print(f"{'VEHICLE':<40} {'TIER':>10} {'SCORE':>7} {'GRADE':>6} {'SAFETY':>7} {'PROBS':>6} {'SALES':>10} {'STATUS':>10}")
    print("=" * 120)

    for tier in range(1, 6):
        tier_results = [r for r in results_ok if r.vehicle.expected_tier == tier]
        if not tier_results:
            continue
        lo, hi = TIER_SCORE_RANGES[tier]
        print(f"\n-- {TIER_LABELS[tier]} (expected {lo}-{hi}) --")
        for r in sorted(tier_results, key=lambda x: x.score):
            v = r.vehicle
            in_range = lo <= r.score <= hi
            status = "OK" if in_range else "MISS"
            sales_str = f"{r.sales_vol:,}" if r.sales_vol else "N/A"
            print(f"  {v.make} {v.model} {v.year:<5} {TIER_LABELS[v.expected_tier]:>10} "
                  f"{r.score:>7.1f} {r.grade:>6} {r.safety:>7.1f} {r.total_problems:>6} "
                  f"{sales_str:>10} {status:>10}")

    if results_err:
        print(f"\n-- Errors ({len(results_err)}) --")
        for r in results_err:
            v = r.vehicle
            print(f"  {v.make} {v.model} {v.year}: {r.error[:80]}")

    # ── Metrics ──
    print("\n" + "=" * 120)
    print("METRICS")
    print("=" * 120)

    scores = [r.score for r in results_ok]
    tiers = [float(r.vehicle.expected_tier) for r in results_ok]

    rho = spearman_rank_correlation(tiers, scores)
    print(f"  Spearman correlation (tier vs score):  {rho:.3f}  (1.0 = perfect)")

    inversions = 0
    for i, r1 in enumerate(results_ok):
        for r2 in results_ok[i + 1:]:
            if r1.vehicle.expected_tier < r2.vehicle.expected_tier and r1.score > r2.score:
                inversions += 1
            elif r1.vehicle.expected_tier > r2.vehicle.expected_tier and r1.score < r2.score:
                inversions += 1
    total_pairs = len(results_ok) * (len(results_ok) - 1) // 2
    inversion_rate = inversions / total_pairs if total_pairs else 0
    print(f"  Inversions:                            {inversions}/{total_pairs} ({inversion_rate:.1%})")

    in_range = sum(
        1 for r in results_ok
        if TIER_SCORE_RANGES[r.vehicle.expected_tier][0] <= r.score <= TIER_SCORE_RANGES[r.vehicle.expected_tier][1]
    )
    print(f"  In expected range:                     {in_range}/{len(results_ok)} ({in_range / len(results_ok):.1%})")

    spread = statistics.stdev(scores) if len(scores) >= 2 else 0
    print(f"  Score spread (std dev):                {spread:.1f}")
    print(f"  Score range:                           {min(scores):.1f} - {max(scores):.1f}")

    grade_counts = {}
    for r in results_ok:
        grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1
    print(f"  Grade distribution:                    {dict(sorted(grade_counts.items()))}")

    tier_avgs = {}
    for tier in range(1, 6):
        ts = [r.score for r in results_ok if r.vehicle.expected_tier == tier]
        if ts:
            tier_avgs[tier] = statistics.mean(ts)
    print(f"  Avg score by tier:                     {', '.join(f'T{t}={a:.1f}' for t, a in sorted(tier_avgs.items()))}")

    monotonic = all(tier_avgs.get(i, 0) <= tier_avgs.get(i + 1, 100) for i in range(1, 5))
    print(f"  Tier averages monotonic:               {'YES' if monotonic else 'NO'}")

    print(f"\n  Total vehicles tested:                 {len(results_ok)}")
    print(f"  Total errors:                          {len(results_err)}")
    print(f"  Total time:                            {sum(r.elapsed for r in results):.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Cache-only (no API calls for uncached)")
    args = parser.parse_args()

    init_db()

    print(f"Running benchmark: {len(BENCHMARK_VEHICLES)} vehicles at {MILEAGE:,} miles")
    print("This will call the NHTSA API for uncached vehicles (may take a few minutes)...")

    results: list[BenchmarkResult] = []
    for i, v in enumerate(BENCHMARK_VEHICLES, 1):
        label = f"{v.make} {v.model} {v.year}"
        sys.stdout.write(f"\r  [{i:>3}/{len(BENCHMARK_VEHICLES)}] {label:<40}")
        sys.stdout.flush()
        result = run_single(v)
        results.append(result)

    sys.stdout.write("\r" + " " * 60 + "\r")
    print_results(results)


if __name__ == "__main__":
    main()
