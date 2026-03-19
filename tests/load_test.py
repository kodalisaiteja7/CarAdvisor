"""Load test for CarAdvisr scaling changes.

Verifies concurrent report generation, caching, request deduplication,
and health endpoint responsiveness under load.

Usage:
    python tests/load_test.py --url https://caradvisr.com --concurrency 10
    python tests/load_test.py --url http://localhost:5000 --concurrency 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

VEHICLES = [
    {"make": "TOYOTA", "model": "CAMRY", "year": 2018, "mileage": 75000},
    {"make": "HONDA", "model": "CIVIC", "year": 2019, "mileage": 60000},
    {"make": "FORD", "model": "F-150", "year": 2020, "mileage": 45000},
    {"make": "CHEVROLET", "model": "SILVERADO 1500", "year": 2019, "mileage": 80000},
    {"make": "TOYOTA", "model": "RAV4", "year": 2021, "mileage": 30000},
    {"make": "HONDA", "model": "ACCORD", "year": 2018, "mileage": 90000},
    {"make": "NISSAN", "model": "ALTIMA", "year": 2020, "mileage": 55000},
    {"make": "HYUNDAI", "model": "ELANTRA", "year": 2019, "mileage": 70000},
    {"make": "KIA", "model": "OPTIMA", "year": 2018, "mileage": 85000},
    {"make": "SUBARU", "model": "OUTBACK", "year": 2020, "mileage": 40000},
    {"make": "MAZDA", "model": "CX-5", "year": 2021, "mileage": 25000},
    {"make": "VOLKSWAGEN", "model": "JETTA", "year": 2019, "mileage": 65000},
    {"make": "BMW", "model": "3 SERIES", "year": 2018, "mileage": 70000},
    {"make": "MERCEDES-BENZ", "model": "C-CLASS", "year": 2019, "mileage": 55000},
    {"make": "AUDI", "model": "A4", "year": 2020, "mileage": 40000},
    {"make": "JEEP", "model": "WRANGLER", "year": 2019, "mileage": 60000},
    {"make": "DODGE", "model": "CHARGER", "year": 2018, "mileage": 80000},
    {"make": "FORD", "model": "ESCAPE", "year": 2020, "mileage": 50000},
    {"make": "TOYOTA", "model": "COROLLA", "year": 2019, "mileage": 55000},
    {"make": "CHEVROLET", "model": "EQUINOX", "year": 2020, "mileage": 45000},
    {"make": "HONDA", "model": "CR-V", "year": 2021, "mileage": 20000},
    {"make": "HYUNDAI", "model": "TUCSON", "year": 2020, "mileage": 35000},
    {"make": "KIA", "model": "SPORTAGE", "year": 2019, "mileage": 50000},
    {"make": "NISSAN", "model": "ROGUE", "year": 2020, "mileage": 40000},
    {"make": "SUBARU", "model": "FORESTER", "year": 2021, "mileage": 25000},
    {"make": "FORD", "model": "EXPLORER", "year": 2019, "mileage": 65000},
    {"make": "TOYOTA", "model": "HIGHLANDER", "year": 2020, "mileage": 40000},
    {"make": "CHEVROLET", "model": "MALIBU", "year": 2018, "mileage": 85000},
    {"make": "HONDA", "model": "PILOT", "year": 2019, "mileage": 60000},
    {"make": "DODGE", "model": "DURANGO", "year": 2018, "mileage": 75000},
]

DEDUP_VEHICLE = {"make": "TOYOTA", "model": "CAMRY", "year": 2022, "mileage": 30000}


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _print_result(r: TestResult):
    status = "PASS" if r.passed else "FAIL"
    print(f"\n  [{status}] {r.name} ({r.duration:.1f}s)")
    for k, v in r.metrics.items():
        print(f"    {k}: {v}")
    for d in r.details:
        print(f"    - {d}")


def _request_report(base_url: str, vehicle: dict, timeout: int = 10) -> dict:
    """POST to /api/analyze and return the JSON response with timing."""
    t0 = time.time()
    resp = requests.post(
        f"{base_url}/api/analyze",
        json=vehicle,
        timeout=timeout,
    )
    elapsed = time.time() - t0
    data = resp.json()
    data["_status_code"] = resp.status_code
    data["_request_time"] = elapsed
    return data


def _poll_until_done(base_url: str, report_id: str, timeout: int = 180) -> dict:
    """Poll /api/progress/<id> via SSE until 'done' or 'error'."""
    t0 = time.time()
    url = f"{base_url}/api/progress/{report_id}"
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                evt = json.loads(line[6:])
                status = evt.get("status", "")
                if status == "done":
                    return {"status": "done", "elapsed": time.time() - t0, "report_id": evt.get("message", report_id)}
                if status == "error":
                    return {"status": "error", "elapsed": time.time() - t0, "message": evt.get("message", "")}
    except requests.exceptions.ReadTimeout:
        pass
    return {"status": "timeout", "elapsed": time.time() - t0}


def test_concurrent_cold(base_url: str, concurrency: int) -> TestResult:
    """Test 1: Fire concurrent requests for different vehicles."""
    _print_header(f"Test 1: Concurrent Cold Reports (n={concurrency})")
    vehicles = VEHICLES[:concurrency]

    t_start = time.time()
    request_times = []
    report_ids = []

    print(f"  Sending {len(vehicles)} concurrent /api/analyze requests...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_request_report, base_url, v): v for v in vehicles}
        for f in as_completed(futures):
            try:
                result = f.result()
                request_times.append(result["_request_time"])
                if result.get("report_id"):
                    report_ids.append(result["report_id"])
                    cached = result.get("cached", False)
                    tag = " (cached)" if cached else ""
                    print(f"    Got report_id={result['report_id']}{tag} in {result['_request_time']:.2f}s")
                else:
                    print(f"    ERROR: {result}")
            except Exception as e:
                print(f"    EXCEPTION: {e}")

    print(f"\n  Polling {len(report_ids)} reports until completion...")
    completion_times = []
    successes = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_poll_until_done, base_url, rid): rid for rid in report_ids}
        for f in as_completed(futures):
            rid = futures[f]
            try:
                result = f.result()
                completion_times.append(result["elapsed"])
                if result["status"] == "done":
                    successes += 1
                    print(f"    {rid}: done in {result['elapsed']:.1f}s")
                else:
                    print(f"    {rid}: {result['status']} after {result['elapsed']:.1f}s - {result.get('message', '')}")
            except Exception as e:
                print(f"    {rid}: EXCEPTION {e}")

    total_time = time.time() - t_start
    avg_req = sum(request_times) / len(request_times) if request_times else 0
    avg_comp = sum(completion_times) / len(completion_times) if completion_times else 0
    max_comp = max(completion_times) if completion_times else 0

    all_fast_requests = all(t < 5.0 for t in request_times)
    enough_success = successes >= len(report_ids) * 0.8

    return TestResult(
        name=f"Concurrent Cold Reports (n={concurrency})",
        passed=all_fast_requests and enough_success,
        duration=total_time,
        metrics={
            "Avg request time": f"{avg_req:.2f}s",
            "Avg completion time": f"{avg_comp:.1f}s",
            "Max completion time": f"{max_comp:.1f}s",
            "Success rate": f"{successes}/{len(report_ids)}",
            "Total wall clock": f"{total_time:.1f}s",
        },
    )


def test_cache_hits(base_url: str, concurrency: int) -> TestResult:
    """Test 2: Re-request same vehicles, expect cache hits."""
    _print_header(f"Test 2: Cache Hit Performance (n={concurrency})")
    vehicles = VEHICLES[:concurrency]

    t_start = time.time()
    cache_hits = 0
    request_times = []

    print(f"  Re-requesting {len(vehicles)} vehicles (expecting cache hits)...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_request_report, base_url, v): v for v in vehicles}
        for f in as_completed(futures):
            v = futures[f]
            try:
                result = f.result()
                request_times.append(result["_request_time"])
                cached = result.get("cached", False)
                if cached:
                    cache_hits += 1
                label = f"{v['year']} {v['make']} {v['model']}"
                print(f"    {label}: {result['_request_time']:.2f}s {'(CACHED)' if cached else '(MISS)'}")
            except Exception as e:
                print(f"    EXCEPTION: {e}")

    total_time = time.time() - t_start
    avg_time = sum(request_times) / len(request_times) if request_times else 0

    return TestResult(
        name=f"Cache Hit Performance (n={concurrency})",
        passed=cache_hits >= len(vehicles) * 0.7,
        duration=total_time,
        metrics={
            "Cache hits": f"{cache_hits}/{len(vehicles)}",
            "Avg response time": f"{avg_time:.3f}s",
            "Max response time": f"{max(request_times):.3f}s" if request_times else "N/A",
        },
    )


def test_deduplication(base_url: str) -> TestResult:
    """Test 3: Fire 10 identical requests, verify same report_id."""
    _print_header("Test 3: Request Deduplication (n=10)")
    n = 10

    t_start = time.time()
    report_ids = []

    print(f"  Sending {n} identical requests for {DEDUP_VEHICLE['year']} {DEDUP_VEHICLE['make']} {DEDUP_VEHICLE['model']}...")
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_request_report, base_url, DEDUP_VEHICLE) for _ in range(n)]
        for f in as_completed(futures):
            try:
                result = f.result()
                rid = result.get("report_id", "?")
                report_ids.append(rid)
                print(f"    report_id={rid} (cached={result.get('cached', False)}) in {result['_request_time']:.2f}s")
            except Exception as e:
                print(f"    EXCEPTION: {e}")

    unique_ids = set(report_ids)
    total_time = time.time() - t_start

    details = []
    if len(unique_ids) == 1:
        details.append(f"All {n} requests returned the same report_id: {unique_ids.pop()}")
    else:
        details.append(f"Got {len(unique_ids)} different report_ids: {unique_ids}")

    return TestResult(
        name="Request Deduplication",
        passed=len(unique_ids) <= 2,
        duration=total_time,
        metrics={
            "Unique report_ids": f"{len(unique_ids)}/{n}",
        },
        details=details,
    )


def test_health_under_load(base_url: str) -> TestResult:
    """Test 4: Hammer /health during load."""
    _print_header("Test 4: Health Endpoint Under Load (100 requests)")
    n = 100

    t_start = time.time()
    response_times = []
    errors = 0

    def _hit_health():
        t0 = time.time()
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            return {"status": resp.status_code, "time": time.time() - t0}
        except Exception as e:
            return {"status": 0, "time": time.time() - t0, "error": str(e)}

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(_hit_health) for _ in range(n)]
        for f in as_completed(futures):
            result = f.result()
            response_times.append(result["time"])
            if result["status"] != 200:
                errors += 1

    total_time = time.time() - t_start
    avg = sum(response_times) / len(response_times)
    p95 = sorted(response_times)[int(len(response_times) * 0.95)]
    max_t = max(response_times)

    print(f"  {n} requests completed in {total_time:.1f}s")
    print(f"  Avg: {avg*1000:.0f}ms  |  P95: {p95*1000:.0f}ms  |  Max: {max_t*1000:.0f}ms  |  Errors: {errors}")

    return TestResult(
        name="Health Under Load",
        passed=errors == 0 and p95 < 1.0,
        duration=total_time,
        metrics={
            "Avg response": f"{avg*1000:.0f}ms",
            "P95 response": f"{p95*1000:.0f}ms",
            "Max response": f"{max_t*1000:.0f}ms",
            "Errors": str(errors),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="CarAdvisr load test")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL to test")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent reports")
    parser.add_argument("--skip-cold", action="store_true", help="Skip cold cache test (slow)")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"\nCarAdvisr Load Test")
    print(f"Target: {base}")
    print(f"Concurrency: {args.concurrency}")

    try:
        resp = requests.get(f"{base}/health", timeout=5)
        print(f"Health check: {resp.status_code}")
        if resp.status_code != 200:
            print("Server is not healthy, aborting.")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot reach server: {e}")
        sys.exit(1)

    results: list[TestResult] = []

    if not args.skip_cold:
        results.append(test_concurrent_cold(base, args.concurrency))
        _print_result(results[-1])

        results.append(test_cache_hits(base, args.concurrency))
        _print_result(results[-1])

    results.append(test_deduplication(base))
    _print_result(results[-1])

    results.append(test_health_under_load(base))
    _print_result(results[-1])

    _print_header("SUMMARY")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.duration:.1f}s)")

    print(f"\n  {passed}/{total} tests passed\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
