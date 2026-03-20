"""Full end-to-end test of Railway deployment after bug fixes."""
import requests
import time

BASE = "https://web-production-f8e9b.up.railway.app"

print("=" * 50)
print("DEPLOYMENT SMOKE TEST")
print("=" * 50)

# 1. Health
print("\n[1] Health check...")
r = requests.get(f"{BASE}/health", timeout=15)
print(f"    Status: {r.status_code} | {r.json()}")
assert r.status_code == 200

# 2. Homepage
print("\n[2] Homepage...")
r = requests.get(f"{BASE}/", timeout=15)
print(f"    Status: {r.status_code}")
assert r.status_code == 200
assert "progressLoader" not in r.text, "ERROR: old progressLoader() still present!"
assert "carFormSteps" in r.text, "ERROR: carFormSteps not found!"
assert "$parent" not in r.text, "ERROR: $parent still referenced!"
print("    HTML structure: OK (no $parent, no progressLoader, carFormSteps present)")

# 3. Submit a fresh analysis
print("\n[3] Submitting analysis: 2019 Honda Accord @ 55k...")
r = requests.post(f"{BASE}/api/analyze", json={
    "make": "Honda", "model": "Accord", "year": 2019, "mileage": 55000
}, timeout=15)
data = r.json()
print(f"    Status: {r.status_code} | Response: {data}")
report_id = data.get("report_id")
cached = data.get("cached", False)

# 4. SSE progress stream
if not cached:
    print(f"\n[4] Polling SSE progress (report_id: {report_id})...")
    start_time = time.time()
    done = False
    with requests.get(f"{BASE}/api/progress/{report_id}", timeout=300, stream=True) as resp:
        print(f"    SSE connection status: {resp.status_code}")
        assert resp.status_code == 200, f"ERROR: SSE returned {resp.status_code}!"
        for line in resp.iter_lines(decode_unicode=True):
            elapsed = int(time.time() - start_time)
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            print(f"    [{elapsed}s] {payload[:140]}")
            if '"done"' in payload:
                print(f"    Completed in ~{elapsed}s")
                done = True
                break
            if '"error"' in payload:
                print("    ERROR in analysis!")
                exit(1)
            if elapsed > 180:
                print("    TIMEOUT")
                break
    assert done, "ERROR: Never received 'done' event!"
else:
    print(f"\n[4] Cached result; skipping SSE test")

# 5. Fetch report
print(f"\n[5] Fetching report...")
time.sleep(1)
rpt = requests.get(f"{BASE}/api/report/{report_id}", timeout=15)
assert rpt.status_code == 200, f"ERROR: Report fetch returned {rpt.status_code}"
report = rpt.json()
v = report.get("vehicle", {})
s = report.get("sections", {}).get("vehicle_summary", {})
ex = report.get("sections", {}).get("executive_summary", {})
cr = report.get("sections", {}).get("current_risk", {})
cl = report.get("sections", {}).get("inspection_checklist", {})

print(f"    Vehicle: {v.get('year')} {v.get('make')} {v.get('model')}")
print(f"    Risk Score: {s.get('reliability_risk_score')}/100")
print(f"    Grade: {s.get('letter_grade')}")
print(f"    Total Recalls: {s.get('total_recalls')}")
print(f"    Recalls data: {len(s.get('recalls', []))} items")

verdict_text = ex.get("text", "")
if verdict_text:
    print(f"    AI Verdict: {verdict_text[:150]}...")
    print(f"    Verdict reasoning: {len(ex.get('verdict_reasoning', []))} bullets")
else:
    print(f"    AI Verdict: NOT GENERATED (check ANTHROPIC_API_KEY)")

must = cl.get("must_check", [])
if must and must[0].get("llm_enhanced"):
    print(f"    Checklist: LLM-enhanced ({len(must)} must-check items)")
else:
    print(f"    Checklist: Raw (no LLM enhancement)")

known = cr.get("top_issues", [])
print(f"    Known Issues: {len(known)}")

cby = cr.get("complaints_by_year", {})
if cby:
    print(f"    Complaints by Year: {len(cby)} years of data")

# 6. Report page renders
print(f"\n[6] Report page render...")
r = requests.get(f"{BASE}/report/{report_id}", timeout=15)
print(f"    Status: {r.status_code}")
assert r.status_code == 200

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
