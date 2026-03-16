"""Scrape Dashboard Light brand-level Quality Index Ratings (QIR).

Fetches each make page from dashboard-light.com, extracts the QIR,
and stores results in vehicle_signals.db and dashboard_light.csv.

Usage:
    python -m scrapers.dashboard_light
"""

from __future__ import annotations

import csv
import re
import sqlite3
import time
from pathlib import Path

import requests

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DB_PATH = DATASET_DIR / "vehicle_signals.db"
CSV_PATH = DATASET_DIR / "dashboard_light.csv"

BASE_URL = "https://www.dashboard-light.com/reports"

MAKES = [
    ("Acura", "Acura"),
    ("Audi", "Audi"),
    ("BMW", "BMW"),
    ("Buick", "Buick"),
    ("Cadillac", "Cadillac"),
    ("Chevrolet", "Chevrolet"),
    ("Chrysler", "Chrysler"),
    ("Dodge", "Dodge"),
    ("Ford", "Ford"),
    ("GMC", "GMC"),
    ("Honda", "Honda"),
    ("Hummer", "Hummer"),
    ("Hyundai", "Hyundai"),
    ("Infiniti", "Infiniti"),
    ("ISUZU", "ISUZU"),
    ("Jaguar", "Jaguar"),
    ("Jeep", "Jeep"),
    ("Kia", "Kia"),
    ("Land Rover", "Land_Rover"),
    ("Lexus", "Lexus"),
    ("Lincoln", "Lincoln"),
    ("Mazda", "Mazda"),
    ("Mercedes-Benz", "Mercedes-Benz"),
    ("Mercury", "Mercury"),
    ("MINI", "MINI"),
    ("Mitsubishi", "Mitsubishi"),
    ("Nissan", "Nissan"),
    ("Oldsmobile", "Oldsmobile"),
    ("Pontiac", "Pontiac"),
    ("Porsche", "Porsche"),
    ("Saab", "Saab"),
    ("Saturn", "Saturn"),
    ("Scion", "Scion"),
    ("Subaru", "Subaru"),
    ("Suzuki", "Suzuki"),
    ("Toyota", "Toyota"),
    ("Volkswagen", "Volkswagen"),
    ("Volvo", "Volvo"),
]

QIR_PATTERN = re.compile(
    r"Manufacturer\s+(?:<[^>]+>)?Quality\s+Index\s+Rating(?:</a>)?:\s*(\d+)",
    re.IGNORECASE,
)


def scrape_all() -> list[tuple[str, int]]:
    """Scrape QIR from every make page. Returns list of (make, qir)."""
    results: list[tuple[str, int]] = []
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CarProjectResearch/1.0"
    )

    for display_name, url_slug in MAKES:
        url = f"{BASE_URL}/{url_slug}.html"
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  SKIP {display_name}: {e}")
            continue

        match = QIR_PATTERN.search(resp.text)
        if match:
            qir = int(match.group(1))
            results.append((display_name.upper(), qir))
            print(f"  {display_name}: QIR {qir}")
        else:
            print(f"  SKIP {display_name}: no QIR found on page")

        time.sleep(0.5)

    return results


def _save_to_db(results: list[tuple[str, int]]):
    conn = sqlite3.connect(str(DB_PATH))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS dashboard_light (
            make TEXT PRIMARY KEY,
            qir INTEGER NOT NULL
        )
    """)
    conn.execute("DELETE FROM dashboard_light")
    conn.executemany(
        "INSERT INTO dashboard_light (make, qir) VALUES (?, ?)", results
    )
    conn.commit()
    cnt = conn.execute("SELECT COUNT(*) FROM dashboard_light").fetchone()[0]
    conn.close()
    print(f"\n  Saved {cnt} rows to {DB_PATH} (dashboard_light table)")


def _save_to_csv(results: list[tuple[str, int]]):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Make", "QIR"])
        for make, qir in sorted(results, key=lambda x: x[0]):
            writer.writerow([make, qir])
    print(f"  Saved {len(results)} rows to {CSV_PATH}")


def main():
    print("Scraping Dashboard Light brand QIR ratings...\n")
    results = scrape_all()

    if not results:
        print("No results scraped!")
        return

    _save_to_db(results)
    _save_to_csv(results)
    print(f"\nDone: {len(results)} makes scraped.")


if __name__ == "__main__":
    main()
