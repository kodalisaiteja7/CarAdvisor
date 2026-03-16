"""One-time ETL: stream-parse TSBs, Investigations, and Manufacturer
Communications into a lightweight SQLite database (vehicle_signals.db).

Streams files line-by-line to handle multi-GB inputs without loading
them into memory.  Only extracts the columns needed for scoring
(make, model, year, component) and skips the large summary text.

Usage:
    python -m data.preprocess_signals
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DB_PATH = DATASET_DIR / "vehicle_signals.db"

PASSENGER_MAKES = {
    "ACURA", "ALFA ROMEO", "AUDI", "BMW", "BUICK", "CADILLAC",
    "CHEVROLET", "CHRYSLER", "DODGE", "FIAT", "FORD", "GENESIS",
    "GMC", "HONDA", "HYUNDAI", "INFINITI", "JAGUAR", "JEEP", "KIA",
    "LAND ROVER", "LEXUS", "LINCOLN", "MASERATI", "MAZDA",
    "MERCEDES-BENZ", "MERCURY", "MINI", "MITSUBISHI", "NISSAN",
    "PONTIAC", "PORSCHE", "RAM", "SAAB", "SATURN", "SCION", "SMART",
    "SUBARU", "SUZUKI", "TESLA", "TOYOTA", "VOLKSWAGEN", "VOLVO",
}

SYSTEM_MAP = {
    "ENGINE": "Engine",
    "POWER TRAIN": "Transmission",
    "TRANSMISSION": "Transmission",
    "ELECTRICAL": "Electrical",
    "ELECTRONIC": "Electrical",
    "SUSPENSION": "Suspension",
    "BRAKE": "Brakes",
    "BODY": "Body/Paint",
    "EXTERIOR": "Body/Paint",
    "INTERIOR": "Interior",
    "SEAT": "Interior",
    "AIR CONDITIONING": "HVAC",
    "HVAC": "HVAC",
    "HEATING": "HVAC",
    "STEERING": "Steering",
    "FUEL": "Fuel System",
    "EXHAUST": "Exhaust",
    "COOLING": "Cooling",
    "AIR BAG": "Safety",
    "LIGHTING": "Electrical",
    "VISIBILITY": "Body/Paint",
    "TIRES": "Suspension",
    "WHEELS": "Suspension",
    "LATCHES": "Body/Paint",
    "STRUCTURE": "Body/Paint",
}


def _map_component(raw: str) -> str:
    """Map a raw NHTSA component string to a system category."""
    upper = raw.upper()
    for key, system in SYSTEM_MAP.items():
        if key in upper:
            return system
    return "Other"


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        DROP TABLE IF EXISTS tsb_counts;
        DROP TABLE IF EXISTS investigations;
        DROP TABLE IF EXISTS mfr_comm_counts;

        CREATE TABLE tsb_counts (
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            system TEXT NOT NULL,
            tsb_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (make, model, year, system)
        );

        CREATE TABLE investigations (
            inv_id TEXT NOT NULL,
            inv_type TEXT NOT NULL,
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            component TEXT NOT NULL,
            subject TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE mfr_comm_counts (
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            comm_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (make, model, year)
        );

        CREATE INDEX IF NOT EXISTS idx_tsb_make_model ON tsb_counts(make, model, year);
        CREATE INDEX IF NOT EXISTS idx_inv_make_model ON investigations(make, model, year);
        CREATE INDEX IF NOT EXISTS idx_mfr_make_model ON mfr_comm_counts(make, model, year);
    """)


def _process_tsbs(conn: sqlite3.Connection):
    """Stream-parse TSB files: extract (tsb_id, make, model, year, component)."""
    tsb_dir = DATASET_DIR / "tsbs"
    files = sorted(tsb_dir.glob("TSBS_RECEIVED_*.txt"))

    if not files:
        print("  No TSB files found, skipping.")
        return

    seen_keys: set[tuple] = set()
    batch: list[tuple] = []
    total_rows = 0
    skipped = 0

    for fpath in files:
        fname = fpath.name
        sys.stdout.write(f"  Parsing {fname}...")
        sys.stdout.flush()
        file_rows = 0

        with open(fpath, encoding="utf-8", errors="replace") as f:
            for line in f:
                cols = line.split("\t")
                if len(cols) < 11:
                    skipped += 1
                    continue

                tsb_id = cols[0].strip()
                make = cols[7].strip().upper()
                model = cols[8].strip().upper()
                year_str = cols[9].strip()
                component = cols[10].strip()

                if make not in PASSENGER_MAKES:
                    continue

                try:
                    year = int(year_str)
                except ValueError:
                    continue

                if year < 1990 or year > 2026:
                    continue

                system = _map_component(component)
                key = (make, model, year, system, tsb_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                batch.append((make, model, year, system))
                file_rows += 1

                if len(batch) >= 50_000:
                    _flush_tsb_batch(conn, batch)
                    batch.clear()

        total_rows += file_rows
        print(f" {file_rows:,} rows")

    if batch:
        _flush_tsb_batch(conn, batch)

    print(f"  TSBs total: {total_rows:,} rows ({skipped:,} skipped)")


def _flush_tsb_batch(conn: sqlite3.Connection, batch: list[tuple]):
    conn.executemany(
        """INSERT INTO tsb_counts (make, model, year, system, tsb_count)
           VALUES (?, ?, ?, ?, 1)
           ON CONFLICT(make, model, year, system)
           DO UPDATE SET tsb_count = tsb_count + 1""",
        batch,
    )
    conn.commit()


def _process_investigations(conn: sqlite3.Connection):
    """Stream-parse FLAT_INV.txt for defect investigations."""
    inv_path = DATASET_DIR / "FLAT_INV.txt"
    if not inv_path.exists():
        print("  FLAT_INV.txt not found, skipping.")
        return

    sys.stdout.write("  Parsing FLAT_INV.txt...")
    sys.stdout.flush()

    type_map = {"PE": "PE", "EA": "EA", "RQ": "RQ", "DP": "DP"}
    batch: list[tuple] = []
    total = 0

    with open(inv_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            cols = line.split("\t")
            if len(cols) < 10:
                continue

            inv_id = cols[0].strip()
            make = cols[1].strip().upper()
            model = cols[2].strip().upper()
            year_str = cols[3].strip()
            component = cols[4].strip()
            subject = cols[9].strip() if len(cols) > 9 else ""

            if make not in PASSENGER_MAKES:
                continue

            inv_type = ""
            for prefix, t in type_map.items():
                if inv_id[:2].upper() == prefix:
                    inv_type = t
                    break
            if not inv_type:
                continue

            try:
                year = int(year_str)
            except ValueError:
                continue

            if year < 1990 or year > 2026:
                continue

            batch.append((inv_id, inv_type, make, model, year,
                          _map_component(component), subject[:200]))
            total += 1

            if len(batch) >= 10_000:
                conn.executemany(
                    "INSERT INTO investigations VALUES (?,?,?,?,?,?,?)", batch)
                conn.commit()
                batch.clear()

    if batch:
        conn.executemany(
            "INSERT INTO investigations VALUES (?,?,?,?,?,?,?)", batch)
        conn.commit()

    print(f" {total:,} rows")


def _process_mfr_comms(conn: sqlite3.Connection):
    """Parse manufacturer communications CSVs."""
    comms_dir = DATASET_DIR / "mfr_comms"
    files = sorted(comms_dir.glob("MFR_COMMS_RECEIVED_*.csv"))

    if not files:
        print("  No mfr_comms files found, skipping.")
        return

    batch: list[tuple] = []
    total = 0

    for fpath in files:
        fname = fpath.name
        sys.stdout.write(f"  Parsing {fname}...")
        sys.stdout.flush()
        file_rows = 0

        with open(fpath, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                print(" empty")
                continue

            for row in reader:
                if len(row) < 4:
                    continue

                make = row[1].strip().upper()
                model = row[2].strip().upper()
                years_str = row[3].strip()

                if make not in PASSENGER_MAKES:
                    continue

                years = []
                for part in re.split(r"[,;\s]+", years_str):
                    part = part.strip()
                    try:
                        y = int(part)
                        if 1990 <= y <= 2026:
                            years.append(y)
                    except ValueError:
                        pass

                for y in years:
                    batch.append((make, model, y))
                    file_rows += 1

                if len(batch) >= 50_000:
                    _flush_mfr_batch(conn, batch)
                    batch.clear()

        total += file_rows
        print(f" {file_rows:,} rows")

    if batch:
        _flush_mfr_batch(conn, batch)

    print(f"  Mfr comms total: {total:,} rows")


def _flush_mfr_batch(conn: sqlite3.Connection, batch: list[tuple]):
    conn.executemany(
        """INSERT INTO mfr_comm_counts (make, model, year, comm_count)
           VALUES (?, ?, ?, 1)
           ON CONFLICT(make, model, year)
           DO UPDATE SET comm_count = comm_count + 1""",
        batch,
    )
    conn.commit()


def main():
    print(f"Pre-processing signals into {DB_PATH}")
    t0 = time.time()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    _init_db(conn)

    print("\n[1/3] Technical Service Bulletins")
    _process_tsbs(conn)

    print("\n[2/3] Defect Investigations")
    _process_investigations(conn)

    print("\n[3/3] Manufacturer Communications")
    _process_mfr_comms(conn)

    row_counts = {}
    for table in ("tsb_counts", "investigations", "mfr_comm_counts"):
        cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        row_counts[table] = cnt

    conn.close()

    elapsed = time.time() - t0
    db_size = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Database: {DB_PATH} ({db_size:.1f} MB)")
    for table, cnt in row_counts.items():
        print(f"  {table}: {cnt:,} rows")


if __name__ == "__main__":
    main()
