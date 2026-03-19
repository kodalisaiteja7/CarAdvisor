"""US vehicle sales volume lookup.

Uses a SQLite database (dataset/us_vehicle_sales.db) built from GCBC
and Kaggle data, covering 2007-2025 with model-level annual sales.
Falls back to the legacy Kaggle CSV if the DB isn't available.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "dataset" / "us_vehicle_sales.db"
_CSV_PATH = Path(__file__).resolve().parent.parent / "dataset" / "us_car_model_sales_2013_2022.csv"

_local = threading.local()
_csv_cache: dict[str, dict[int, int]] | None = None
_csv_lock = threading.Lock()

_BRAND_MAP = {
    "VOLKSWAGEN": "VW",
    "MERCEDES BENZ": "MERCEDES-BENZ",
}

_MODEL_ALIASES = {
    "C-HR": "C0HR",
    "E-TRON": "E0TRON",
    "E-TRON GT": "E0TRON GT",
    "Q4 E-TRON": "Q4 E0TRON",
}


def _get_db() -> sqlite3.Connection | None:
    """Get a thread-local connection to the sales DB, or None if unavailable."""
    conn = getattr(_local, "db_conn", None)
    if conn is not None:
        return conn
    if not _DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.db_conn = conn
        cnt = conn.execute("SELECT COUNT(*) FROM vehicle_sales").fetchone()[0]
        logger.info("Sales DB loaded: %d entries from %s", cnt, _DB_PATH.name)
        return conn
    except Exception as e:
        logger.warning("Failed to open sales DB: %s", e)
        return None


def _strip_for_compare(s: str) -> str:
    return "".join(c for c in s if c.isalnum()).upper()


def _lookup_db(make: str, model: str, year: int) -> int | None:
    """Look up sales from the SQLite database."""
    conn = _get_db()
    if conn is None:
        return None

    row = conn.execute(
        "SELECT units_sold FROM vehicle_sales WHERE UPPER(make)=? AND UPPER(model)=? AND year=?",
        (make.upper(), model.upper(), year),
    ).fetchone()
    if row:
        return row[0]

    rows = conn.execute(
        "SELECT model, units_sold FROM vehicle_sales WHERE UPPER(make)=? AND year=?",
        (make.upper(), year),
    ).fetchall()
    target = _strip_for_compare(model)
    for db_model, units in rows:
        if target in _strip_for_compare(db_model) or _strip_for_compare(db_model) in target:
            return units

    for nearby in [year - 1, year + 1, year - 2, year + 2]:
        row = conn.execute(
            "SELECT units_sold FROM vehicle_sales WHERE UPPER(make)=? AND UPPER(model)=? AND year=?",
            (make.upper(), model.upper(), nearby),
        ).fetchone()
        if row:
            return row[0]

    return None


def _load_csv() -> dict[str, dict[int, int]]:
    """Load the legacy Kaggle CSV as fallback (thread-safe)."""
    global _csv_cache
    if _csv_cache is not None:
        return _csv_cache

    with _csv_lock:
        if _csv_cache is not None:
            return _csv_cache
        _csv_cache = {}
    if not _CSV_PATH.exists():
        return _csv_cache

    with open(_CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        year_cols = []
        for col in header[2:]:
            try:
                year_cols.append(int(col))
            except ValueError:
                pass

        for row in reader:
            if len(row) < 3:
                continue
            brand = row[0].strip().upper()
            brand_model = row[1].strip()

            model_part = brand_model
            if model_part.upper().startswith(brand + " "):
                model_part = model_part[len(brand) + 1:]
            elif model_part.upper().startswith(brand):
                model_part = model_part[len(brand):]
            model_upper = model_part.strip().upper()

            mapped_brand = _BRAND_MAP.get(brand, brand)
            key = f"{mapped_brand}|{model_upper}"

            year_sales: dict[int, int] = {}
            for i, yr in enumerate(year_cols):
                val_str = row[2 + i].strip() if (2 + i) < len(row) else "0"
                if val_str in ("", "N/A", "n/a", "-"):
                    continue
                try:
                    val = int(val_str.replace(",", ""))
                    if val > 0:
                        year_sales[yr] = val
                except ValueError:
                    continue

            if year_sales:
                if key in _csv_cache:
                    for yr, v in year_sales.items():
                        _csv_cache[key][yr] = _csv_cache[key].get(yr, 0) + v
                else:
                    _csv_cache[key] = year_sales

    return _csv_cache


def _lookup_csv(make: str, model: str, year: int) -> int | None:
    """Look up sales from the legacy CSV."""
    data = _load_csv()
    if not data:
        return None

    brand = _BRAND_MAP.get(make.upper(), make.upper())
    model_upper = model.strip().upper()
    model_upper = _MODEL_ALIASES.get(model_upper, model_upper)

    key = f"{brand}|{model_upper}"
    entry = data.get(key)

    if entry is None:
        model_stripped = _strip_for_compare(model_upper)
        for k, v in data.items():
            b, m = k.split("|", 1)
            if b != brand:
                continue
            if model_upper in m or m in model_upper:
                entry = v
                break
            if model_stripped == _strip_for_compare(m):
                entry = v
                break

    if entry is None:
        return None

    sales = entry.get(year)
    if sales and sales > 0:
        return sales

    for ny in [year - 1, year + 1, year - 2, year + 2]:
        s = entry.get(ny)
        if s and s > 0:
            return s

    return None


def get_sales_volume(make: str, model: str, year: int) -> int | None:
    """Look up annual US sales for a make/model/year.

    Tries the SQLite database first (broader coverage), then falls
    back to the Kaggle CSV.
    """
    result = _lookup_db(make, model, year)
    if result:
        return result

    result = _lookup_csv(make, model, year)
    return result


_computed_baseline: float | None = None


def get_complaints_per_1k_baseline() -> float:
    """Compute the median complaints-per-1000-sold across all models.

    Uses NHTSA bulk data (ModelStats) joined with sales data.
    Falls back to 1.5 if bulk data is unavailable.
    """
    global _computed_baseline
    if _computed_baseline is not None:
        return _computed_baseline

    try:
        from data.stats_builder import ModelStats, _get_bulk_engine
        from sqlalchemy.orm import sessionmaker

        engine = _get_bulk_engine()
        Session = sessionmaker(bind=engine)
        session = Session()

        all_stats = session.query(ModelStats).all()
        rates = []
        for ms in all_stats:
            if ms.year < 2007 or ms.year > 2025:
                continue
            sv = get_sales_volume(ms.make, ms.model, ms.year)
            if sv and sv >= 500 and ms.total_complaints >= 1:
                rates.append((ms.total_complaints / sv) * 1000)

        session.close()

        if len(rates) >= 50:
            rates.sort()
            mid = len(rates) // 2
            _computed_baseline = rates[mid]
            logger.info(
                "Computed real baseline: %.2f per 1k (from %d models)",
                _computed_baseline, len(rates),
            )
            return _computed_baseline
    except Exception as exc:
        logger.warning("Could not compute real baseline: %s", exc)

    _computed_baseline = 1.5
    return _computed_baseline
