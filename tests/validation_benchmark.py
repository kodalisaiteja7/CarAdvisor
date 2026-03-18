"""Validation benchmark: 100 NEW vehicles (no overlap with score_benchmark.py).

Runs both V1 and V2 scoring at 75k miles and outputs two separate CSVs:
  - validation_v1_results.csv  (V1 only)
  - validation_v2_results.csv  (V2 only + component breakdown)

Usage:
    python -m tests.validation_benchmark
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

VALIDATION_VEHICLES: list[BenchmarkVehicle] = [
    # ── Tier 1: Known reliable ──
    BenchmarkVehicle("TOYOTA", "CAMRY", 2017, 1, "Strong generation"),
    BenchmarkVehicle("TOYOTA", "COROLLA", 2017, 1, "Perennial reliability"),
    BenchmarkVehicle("TOYOTA", "RAV4", 2018, 1, "Pre-redesign reliable"),
    BenchmarkVehicle("TOYOTA", "HIGHLANDER", 2019, 1, "Dependable SUV"),
    BenchmarkVehicle("TOYOTA", "TACOMA", 2019, 1, "Proven truck"),
    BenchmarkVehicle("LEXUS", "RX", 2019, 1, "Luxury reliability leader"),
    BenchmarkVehicle("LEXUS", "NX", 2018, 1, "Compact luxury reliable"),
    BenchmarkVehicle("HONDA", "CIVIC", 2017, 1, "10th gen strong start"),
    BenchmarkVehicle("HONDA", "ACCORD", 2019, 1, "CR best sedan"),
    BenchmarkVehicle("HONDA", "CR-V", 2018, 1, "Solid compact SUV"),
    BenchmarkVehicle("MAZDA", "CX-5", 2019, 1, "Top reliability marks"),
    BenchmarkVehicle("MAZDA", "MAZDA3", 2018, 1, "Reliable compact"),
    BenchmarkVehicle("TOYOTA", "CAMRY", 2016, 1, "Proven platform"),
    BenchmarkVehicle("TOYOTA", "COROLLA", 2016, 1, "Mature generation"),
    BenchmarkVehicle("LEXUS", "ES", 2018, 1, "Trouble-free sedan"),
    BenchmarkVehicle("HONDA", "HR-V", 2019, 1, "Simple subcompact SUV"),
    BenchmarkVehicle("TOYOTA", "PRIUS", 2017, 1, "Hybrid proven tech"),
    BenchmarkVehicle("ACURA", "RDX", 2018, 1, "Honda-based reliability"),
    BenchmarkVehicle("ACURA", "TLX", 2018, 1, "Accord DNA"),
    BenchmarkVehicle("MAZDA", "CX-3", 2018, 1, "Small, few problems"),

    # ── Tier 2: Above average ──
    BenchmarkVehicle("SUBARU", "IMPREZA", 2018, 2, "Solid small car"),
    BenchmarkVehicle("HYUNDAI", "SONATA", 2019, 2, "Post-Theta improvement"),
    BenchmarkVehicle("KIA", "SOUL", 2019, 2, "Simple, reliable box"),
    BenchmarkVehicle("TOYOTA", "COROLLA", 2020, 2, "New platform, minor issues"),
    BenchmarkVehicle("HONDA", "RIDGELINE", 2018, 2, "Pilot-based truck"),
    BenchmarkVehicle("HYUNDAI", "PALISADE", 2020, 2, "Strong debut"),
    BenchmarkVehicle("KIA", "SELTOS", 2021, 2, "Decent subcompact SUV"),
    BenchmarkVehicle("TOYOTA", "RAV4", 2020, 2, "New gen, some growing pains"),
    BenchmarkVehicle("SUBARU", "LEGACY", 2018, 2, "Dependable sedan"),
    BenchmarkVehicle("HONDA", "PASSPORT", 2019, 2, "Pilot twin reliable"),
    BenchmarkVehicle("KIA", "CARNIVAL", 2022, 2, "Modern minivan, decent"),
    BenchmarkVehicle("HYUNDAI", "VENUE", 2020, 2, "Basic, fewer issues"),
    BenchmarkVehicle("MAZDA", "CX-30", 2020, 2, "Newer, good reliability"),
    BenchmarkVehicle("TOYOTA", "VENZA", 2021, 2, "RAV4 based hybrid"),
    BenchmarkVehicle("HONDA", "INSIGHT", 2019, 2, "Civic hybrid, reliable"),
    BenchmarkVehicle("SUBARU", "ASCENT", 2019, 2, "First gen, decent"),
    BenchmarkVehicle("KIA", "STINGER", 2018, 2, "Sporty but reliable"),
    BenchmarkVehicle("HYUNDAI", "IONIQ", 2019, 2, "Hybrid, few complaints"),
    BenchmarkVehicle("GENESIS", "G70", 2019, 2, "Luxury Hyundai, solid"),
    BenchmarkVehicle("TOYOTA", "86", 2018, 2, "Simple sports car"),

    # ── Tier 3: Average ──
    BenchmarkVehicle("FORD", "F-150", 2019, 3, "Popular truck, some issues"),
    BenchmarkVehicle("CHEVROLET", "SILVERADO", 2019, 3, "Average truck reliability"),
    BenchmarkVehicle("FORD", "ESCAPE", 2019, 3, "Average compact SUV"),
    BenchmarkVehicle("CHEVROLET", "IMPALA", 2018, 3, "Average full-size"),
    BenchmarkVehicle("NISSAN", "ALTIMA", 2019, 3, "CVT concerns persist"),
    BenchmarkVehicle("NISSAN", "KICKS", 2019, 3, "Basic, average"),
    BenchmarkVehicle("GMC", "TERRAIN", 2018, 3, "Average small SUV"),
    BenchmarkVehicle("CHEVROLET", "BLAZER", 2019, 3, "New model, average"),
    BenchmarkVehicle("FORD", "MUSTANG", 2018, 3, "Some issues, mostly OK"),
    BenchmarkVehicle("CHEVROLET", "CAMARO", 2018, 3, "Average sports car"),
    BenchmarkVehicle("GMC", "ACADIA", 2018, 3, "Average mid-size"),
    BenchmarkVehicle("NISSAN", "MAXIMA", 2018, 3, "Average sedan"),
    BenchmarkVehicle("FORD", "FLEX", 2018, 3, "Aging but average"),
    BenchmarkVehicle("CHEVROLET", "TAHOE", 2018, 3, "Average full-size SUV"),
    BenchmarkVehicle("NISSAN", "FRONTIER", 2018, 3, "Old platform, average"),
    BenchmarkVehicle("RAM", "1500", 2018, 3, "Average full-size truck"),
    BenchmarkVehicle("BUICK", "ENVISION", 2018, 3, "Average luxury SUV"),
    BenchmarkVehicle("CHEVROLET", "SPARK", 2018, 3, "Basic budget car"),
    BenchmarkVehicle("FORD", "TRANSIT CONNECT", 2018, 3, "Commercial average"),
    BenchmarkVehicle("NISSAN", "VERSA", 2018, 3, "Budget average sedan"),

    # ── Tier 4: Below average ──
    BenchmarkVehicle("JEEP", "WRANGLER", 2017, 4, "Known issues pre-JL"),
    BenchmarkVehicle("BMW", "5 SERIES", 2017, 4, "Complex, expensive repairs"),
    BenchmarkVehicle("MERCEDES-BENZ", "E-CLASS", 2018, 4, "Electronics problems"),
    BenchmarkVehicle("AUDI", "A6", 2018, 4, "Complex luxury issues"),
    BenchmarkVehicle("DODGE", "CHALLENGER", 2018, 4, "Below average muscle"),
    BenchmarkVehicle("CADILLAC", "XT5", 2018, 4, "Below average luxury"),
    BenchmarkVehicle("NISSAN", "ROGUE", 2017, 4, "CVT problems worsen"),
    BenchmarkVehicle("VOLKSWAGEN", "PASSAT", 2018, 4, "Electrical issues"),
    BenchmarkVehicle("FORD", "EXPLORER", 2018, 4, "Transmission complaints"),
    BenchmarkVehicle("CHEVROLET", "EQUINOX", 2017, 4, "Oil consumption era"),
    BenchmarkVehicle("LINCOLN", "MKC", 2018, 4, "Ford-based issues"),
    BenchmarkVehicle("INFINITI", "Q50", 2018, 4, "Electronic gremlins"),
    BenchmarkVehicle("NISSAN", "MURANO", 2017, 4, "CVT aging"),
    BenchmarkVehicle("CADILLAC", "ATS", 2016, 4, "Reliability issues"),
    BenchmarkVehicle("VOLVO", "XC60", 2018, 4, "Complex electronics"),
    BenchmarkVehicle("CHRYSLER", "300", 2018, 4, "Aging platform issues"),
    BenchmarkVehicle("DODGE", "GRAND CARAVAN", 2018, 4, "Old design, issues"),
    BenchmarkVehicle("FORD", "FUSION", 2017, 4, "Increasing problems"),
    BenchmarkVehicle("MINI", "COUNTRYMAN", 2018, 4, "BMW-based issues"),
    BenchmarkVehicle("JEEP", "CHEROKEE", 2017, 4, "9-speed trans problems"),

    # ── Tier 5: Known unreliable ──
    BenchmarkVehicle("HYUNDAI", "SONATA", 2013, 5, "Theta II failures peak"),
    BenchmarkVehicle("KIA", "OPTIMA", 2013, 5, "Theta II engine seizures"),
    BenchmarkVehicle("CHRYSLER", "200", 2014, 5, "Notoriously bad"),
    BenchmarkVehicle("DODGE", "JOURNEY", 2016, 5, "Consistently unreliable"),
    BenchmarkVehicle("JEEP", "PATRIOT", 2016, 5, "Poor reliability, CVT"),
    BenchmarkVehicle("FORD", "FOCUS", 2015, 5, "PowerShift DCT failures"),
    BenchmarkVehicle("FORD", "FIESTA", 2014, 5, "PowerShift DCT severe"),
    BenchmarkVehicle("NISSAN", "ROGUE", 2014, 5, "CVT failures common"),
    BenchmarkVehicle("NISSAN", "PATHFINDER", 2014, 5, "CVT transmission failure"),
    BenchmarkVehicle("CHEVROLET", "CRUZE", 2015, 5, "Cooling/engine problems"),
    BenchmarkVehicle("FIAT", "500L", 2016, 5, "Very unreliable"),
    BenchmarkVehicle("JEEP", "RENEGADE", 2016, 5, "First year problems"),
    BenchmarkVehicle("DODGE", "DART", 2014, 5, "Poor build quality"),
    BenchmarkVehicle("BMW", "X1", 2016, 5, "Oil leaks, electrical"),
    BenchmarkVehicle("CHRYSLER", "TOWN AND COUNTRY", 2016, 5, "Aging minivan"),
    BenchmarkVehicle("LAND ROVER", "RANGE ROVER SPORT", 2016, 5, "Chronic issues"),
    BenchmarkVehicle("HYUNDAI", "SANTA FE", 2014, 5, "Engine/trans issues"),
    BenchmarkVehicle("KIA", "SORENTO", 2014, 5, "Theta engine risk"),
    BenchmarkVehicle("NISSAN", "ALTIMA", 2015, 5, "CVT severe failures"),
    BenchmarkVehicle("RAM", "1500", 2015, 5, "EcoDiesel/trans problems"),
]

MILEAGE = 75_000
VEHICLE_TIMEOUT = 45


def _run_one(v: BenchmarkVehicle) -> dict:
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
            "make": v.make, "model": v.model, "year": v.year,
            "tier": v.expected_tier, "tier_label": TIER_LABELS[v.expected_tier],
            "complaints": agg.total_complaints,
            "sales": sales_vol or "",
            "score_v1": vs1.reliability_risk_score, "grade_v1": vs1.letter_grade,
            "score_v2": vs2.risk_score_v2, "grade_v2": vs2.letter_grade,
            "nhtsa_pct": vs2.nhtsa_component,
            "tsb_score": vs2.tsb_component, "tsb_count": vs2.tsb_raw_count,
            "inv_score": vs2.investigation_component, "inv_count": vs2.inv_raw_count,
            "mfr_score": vs2.mfr_comm_component, "mfr_count": vs2.mfr_comm_raw_count,
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
    total = len(VALIDATION_VEHICLES)
    print(f"Validation benchmark: {total} NEW vehicles at {MILEAGE:,} miles")
    print("=" * 80)

    results: list[dict] = []
    t0 = time.time()

    for i, v in enumerate(VALIDATION_VEHICLES, 1):
        label = f"{v.make} {v.model} {v.year}"
        sys.stdout.write(f"\r  [{i:>3}/{total}] {label:<40}")
        sys.stdout.flush()
        results.append(run_with_timeout(v))

    elapsed = time.time() - t0
    print(f"\n\nCompleted in {elapsed:.0f}s")

    ok = [r for r in results if not r["error"]]
    err = [r for r in results if r["error"]]
    print(f"Results: {len(ok)} ok, {len(err)} errors")

    # ── Write V1-only CSV ──
    v1_headers = ["Make", "Model", "Year", "Tier", "TierLabel", "Complaints",
                  "Sales", "Score_V1", "Grade_V1"]
    v1_keys = ["make", "model", "year", "tier", "tier_label", "complaints",
               "sales", "score_v1", "grade_v1"]

    with open("validation_v1_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(v1_headers)
        for r in results:
            w.writerow([r[k] for k in v1_keys])

    # ── Write V2-only CSV ──
    v2_headers = ["Make", "Model", "Year", "Tier", "TierLabel", "Complaints",
                  "Sales", "Score_V1", "Grade_V1", "Score_V2", "Grade_V2",
                  "NHTSA_Pct", "TSB_Score", "TSB_Count", "Inv_Score", "Inv_Count",
                  "Mfr_Score", "Mfr_Count", "DL_QIR", "DL_Score"]
    v2_keys = ["make", "model", "year", "tier", "tier_label", "complaints",
               "sales", "score_v1", "grade_v1", "score_v2", "grade_v2",
               "nhtsa_pct", "tsb_score", "tsb_count", "inv_score", "inv_count",
               "mfr_score", "mfr_count", "dl_qir", "dl_score"]

    with open("validation_v2_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(v2_headers)
        for r in results:
            w.writerow([r[k] for k in v2_keys])

    print(f"CSV: validation_v1_results.csv")
    print(f"CSV: validation_v2_results.csv")

    if err:
        print(f"\nErrors ({len(err)}):")
        for r in err:
            print(f"  {r['make']} {r['model']} {r['year']}: {r['error']}")

    if ok:
        _print_summary(ok)


def _print_summary(results: list[dict]):
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
            print(f"  {label:<33} {TIER_LABELS[r['tier']]:>8} "
                  f"{r['score_v1']:>6.1f} {r['grade_v1']:>3} "
                  f"{r['score_v2']:>6.1f} {r['grade_v2']:>3} "
                  f"{r['tsb_score']:>6.1f} {r['inv_score']:>6.1f} "
                  f"{r['mfr_score']:>6.1f} {r['dl_score']:>6.1f}")
        v1_avg = sum(r["score_v1"] for r in tier_results) / len(tier_results)
        v2_avg = sum(r["score_v2"] for r in tier_results) / len(tier_results)
        print(f"  {'AVG':<33} {'':>8} {v1_avg:>6.1f} {'':>3} {v2_avg:>6.1f}")

    print("\n" + "=" * 110)
    v1_scores = [r["score_v1"] for r in results]
    v2_scores = [r["score_v2"] for r in results]
    tiers = [float(r["tier"]) for r in results]

    rho_v1 = spearman_rank_correlation(tiers, v1_scores)
    rho_v2 = spearman_rank_correlation(tiers, v2_scores)

    print(f"  Spearman correlation (tier vs V1): {rho_v1:.3f}")
    print(f"  Spearman correlation (tier vs V2): {rho_v2:.3f}")
    print(f"  Improvement: {rho_v2 - rho_v1:+.3f}")
    print(f"\n  V1 range: {min(v1_scores):.1f} - {max(v1_scores):.1f}")
    print(f"  V2 range: {min(v2_scores):.1f} - {max(v2_scores):.1f}")


if __name__ == "__main__":
    main()
