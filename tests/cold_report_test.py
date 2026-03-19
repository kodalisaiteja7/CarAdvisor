"""Test a single cold (non-cached) report on production."""
import requests
import json
import sys
import time

base = sys.argv[1] if len(sys.argv) > 1 else "https://www.caradvisr.com"
v = {"make": "LEXUS", "model": "RX 350", "year": 2016, "mileage": 95000}

print(f"Target: {base}")
print(f"Vehicle: {v['year']} {v['make']} {v['model']} @ {v['mileage']} mi")
print("Requesting report...")
t0 = time.time()
r = requests.post(f"{base}/api/analyze", json=v, timeout=15)
data = r.json()
print(f"Response in {time.time()-t0:.2f}s: {json.dumps(data)}")

rid = data.get("report_id")
if not rid:
    print("No report_id")
    sys.exit(1)

if data.get("cached"):
    print("Cache hit - report already exists, try a different vehicle")
    sys.exit(0)

print(f"Polling {rid}...")
t1 = time.time()
with requests.get(f"{base}/api/progress/{rid}", stream=True, timeout=180) as resp:
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        evt = json.loads(line[6:])
        s = evt.get("status", "")
        src = evt.get("source", "")
        msg = evt.get("message", "")
        elapsed = time.time() - t1
        print(f"  [{elapsed:5.1f}s] [{s}] {src}: {msg}")
        if s in ("done", "error"):
            break

total = time.time() - t0
print(f"\nTotal time: {total:.1f}s")

r2 = requests.get(f"{base}/api/report/{rid}", timeout=10)
print(f"Report status: {r2.status_code}, size: {len(r2.text)} bytes")
if r2.status_code == 200:
    print("SUCCESS")
else:
    print("FAIL")
