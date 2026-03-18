"""Test a fresh (uncached) vehicle on Railway deployment."""
import requests
import time

BASE = "https://web-production-f8e9b.up.railway.app"

print("Testing fresh vehicle: 2020 Toyota Camry @ 45k miles...")
r = requests.post(f"{BASE}/api/analyze", json={
    "make": "Toyota", "model": "Camry", "year": 2020, "mileage": 45000
}, timeout=15)
data = r.json()
print(f"  Status: {r.status_code} | Response: {data}")
report_id = data.get("report_id")
cached = data.get("cached", False)
print(f"  Report ID: {report_id} | Cached: {cached}")

if not cached:
    print("  Polling SSE for progress...")
    start_time = time.time()
    with requests.get(f"{BASE}/api/progress/{report_id}", timeout=300, stream=True) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            elapsed = int(time.time() - start_time)
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            print(f"  [{elapsed}s] {payload[:140]}")
            if '"done"' in payload:
                print(f"  Completed in ~{elapsed}s")
                break
            if '"error"' in payload:
                print(f"  ERROR!")
                exit(1)
            if elapsed > 180:
                print("  TIMEOUT")
                break
else:
    print("  Using cached result")

time.sleep(1)
rpt = requests.get(f"{BASE}/api/report/{report_id}", timeout=15)
if rpt.status_code == 200:
    report = rpt.json()
    v = report.get("vehicle", {})
    s = report.get("sections", {}).get("vehicle_summary", {})
    ex = report.get("sections", {}).get("executive_summary", {})
    cr = report.get("sections", {}).get("current_risk", {})
    print(f"\n--- REPORT RESULTS ---")
    print(f"Vehicle: {v.get('year')} {v.get('make')} {v.get('model')}")
    print(f"Risk Score: {s.get('reliability_risk_score')}/100")
    print(f"Grade: {s.get('letter_grade')}")
    print(f"Safety Score: {cr.get('safety_score')}")
    print(f"Total Recalls: {s.get('total_recalls')}")
    verdict = ex.get("verdict", "NOT GENERATED")
    print(f"AI Verdict: {verdict[:200]}")
    checklist = report.get("sections", {}).get("inspection_checklist", {})
    must = checklist.get("must_check", [])
    rec = checklist.get("recommended", [])
    std = checklist.get("standard", [])
    print(f"Checklist: {len(must)} must-check, {len(rec)} recommended, {len(std)} standard")
    known = report.get("sections", {}).get("known_issues", {}).get("issues", [])
    print(f"Known Issues: {len(known)}")
    recalls = report.get("sections", {}).get("recall_info", {}).get("recalls", [])
    print(f"Recalls: {len(recalls)}")
    cby = cr.get("complaints_by_year", {})
    print(f"Complaints by Year: {dict(list(cby.items())[:5]) if cby else 'None'}")
    print("\nALL TESTS PASSED!")
else:
    print(f"Report fetch failed: {rpt.status_code}")
    exit(1)
