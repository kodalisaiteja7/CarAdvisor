"""Quick single-report test to verify production is working."""
import requests
import json
import sys

base = sys.argv[1] if len(sys.argv) > 1 else "https://www.caradvisr.com"
v = {"make": "FORD", "model": "MUSTANG", "year": 2017, "mileage": 60000}

print(f"Target: {base}")
print("Requesting report...")
r = requests.post(f"{base}/api/analyze", json=v, timeout=15)
data = r.json()
print(f"Response: {json.dumps(data)}")

rid = data.get("report_id")
if not rid:
    print("No report_id, aborting")
    sys.exit(1)

print(f"Polling {rid}...")
with requests.get(f"{base}/api/progress/{rid}", stream=True, timeout=180) as resp:
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        evt = json.loads(line[6:])
        status = evt.get("status", "")
        source = evt.get("source", "")
        msg = evt.get("message", "")
        print(f"  [{status}] {source}: {msg}")
        if status in ("done", "error"):
            break

print("\nChecking if report exists...")
r2 = requests.get(f"{base}/api/report/{rid}", timeout=10)
print(f"Report status: {r2.status_code}, size: {len(r2.text)} bytes")
if r2.status_code == 200:
    print("SUCCESS - Report generated!")
else:
    print("FAIL - Report not found")
