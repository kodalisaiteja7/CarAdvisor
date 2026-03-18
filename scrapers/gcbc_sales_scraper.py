"""Scrape US vehicle sales data from GoodCarBadCar.net model pages.

Each model has a dedicated page (e.g. /toyota-camry-sales-figures/) with
a clean "US Annual Sales" table. This scraper:
1. Fetches the model index page to discover all model URLs
2. Filters to US-market brands only
3. Fetches each model page and extracts the US Annual Sales table
4. Stores results in dataset/us_vehicle_sales.db
"""

from __future__ import annotations

import csv
import logging
import re
import sqlite3
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "dataset" / "us_vehicle_sales.db"
KAGGLE_CSV = Path(__file__).resolve().parent.parent / "dataset" / "us_car_model_sales_2013_2022.csv"
INDEX_URL = "https://www.goodcarbadcar.net/sales-by-model/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

US_BRANDS = {
    "acura", "alfa romeo", "aston martin", "audi", "bentley", "bmw", "buick",
    "cadillac", "chevrolet", "chrysler", "dodge", "ferrari", "fiat", "fisker",
    "ford", "genesis", "gmc", "honda", "hummer", "hyundai", "infiniti", "isuzu",
    "jaguar", "jeep", "kia", "lamborghini", "land rover", "lexus", "lincoln",
    "lotus", "lucid", "maserati", "mazda", "mclaren", "mercedes-benz", "mercury",
    "mini", "mitsubishi", "nissan", "oldsmobile", "polestar", "pontiac", "porsche",
    "ram", "rivian", "rolls-royce", "saab", "saturn", "scion", "smart", "subaru",
    "suzuki", "tesla", "toyota", "vinfast", "volkswagen", "volvo",
}

MAKE_NORMALIZE = {
    "mercedes benz": "MERCEDES-BENZ",
    "rolls royce": "ROLLS-ROYCE",
    "land rover": "LAND ROVER",
    "alfa romeo": "ALFA ROMEO",
    "aston martin": "ASTON MARTIN",
}


def _parse_int(s: str) -> int | None:
    s = s.strip().replace(",", "").replace("\u2013", "").replace("-", "")
    if not s or s.lower() in ("n/a", "na", ""):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _fetch(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def discover_model_urls() -> list[tuple[str, str, str]]:
    """Fetch the index page and extract (make, model, url) tuples for US brands."""
    html = _fetch(INDEX_URL)
    if not html:
        raise RuntimeError("Failed to fetch the GCBC model index page")

    soup = BeautifulSoup(html, "lxml")
    results = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "sales-figures" not in href:
            continue

        title = a.get_text(strip=True)
        if not title or "Sales Figures" not in title:
            continue

        title_clean = title.replace("Sales Figures", "").strip()
        if not title_clean:
            continue

        title_lower = title_clean.lower()
        make = None
        model = None
        for brand in sorted(US_BRANDS, key=lambda b: -len(b)):
            if title_lower.startswith(brand + " "):
                make = brand
                model = title_clean[len(brand):].strip()
                break
            if title_lower == brand:
                make = brand
                model = ""
                break

        if not make or not model:
            continue

        make_upper = MAKE_NORMALIZE.get(make, make.upper())
        model_upper = model.upper()

        full_url = href if href.startswith("http") else f"https://www.goodcarbadcar.net{href}"
        key = f"{make_upper}|{model_upper}"
        if key in seen:
            continue
        seen.add(key)
        results.append((make_upper, model_upper, full_url))

    return results


def _find_us_section_tables(soup: BeautifulSoup) -> list:
    """Find tables that belong to the US sales section of a model page."""
    us_heading = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        text = h.get_text(strip=True).lower()
        if ("u.s" in text or "us " in text) and "sales" in text:
            us_heading = h
            break

    if not us_heading:
        return []

    tables = []
    for sibling in us_heading.find_all_next():
        if sibling.name in ("h2",):
            break
        if sibling.name == "table":
            tables.append(sibling)
    return tables


def _parse_annual_table(table) -> list[dict]:
    """Parse a simple Year | Sales Units table."""
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]

    year_col = 0
    sales_col = None
    for i, h in enumerate(headers):
        if "year" in h:
            year_col = i
        if "sales" in h or "units" in h:
            sales_col = i

    if sales_col is None:
        if len(headers) == 2:
            sales_col = 1
        else:
            return []

    records = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(year_col, sales_col):
            continue

        year_text = cells[year_col].get_text(strip=True)
        sales_text = cells[sales_col].get_text(strip=True)

        year_match = re.search(r"(19|20)\d{2}", year_text)
        if not year_match:
            continue
        year = int(year_match.group())
        if year < 2000:
            continue

        sales = _parse_int(sales_text)
        if sales is None or sales <= 0:
            continue
        records.append({"year": year, "units_sold": sales})

    return records


def _parse_monthly_table(table) -> list[dict]:
    """Parse a Year | Jan | Feb | ... | Dec table, summing months."""
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]

    MONTHS = {"jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"}
    month_cols = [i for i, h in enumerate(headers) if h[:3] in MONTHS]

    if len(month_cols) < 6:
        return []

    year_col = 0
    for i, h in enumerate(headers):
        if "year" in h:
            year_col = i
            break

    records = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(month_cols):
            continue

        year_text = cells[year_col].get_text(strip=True)
        year_match = re.search(r"(19|20)\d{2}", year_text)
        if not year_match:
            continue
        year = int(year_match.group())
        if year < 2000:
            continue

        total = 0
        valid_months = 0
        for mc in month_cols:
            if mc < len(cells):
                val = _parse_int(cells[mc].get_text(strip=True))
                if val is not None and val > 0:
                    total += val
                    valid_months += 1

        if total > 0 and valid_months >= 1:
            records.append({"year": year, "units_sold": total})

    return records


def scrape_model_page(url: str) -> list[dict]:
    """Fetch a model page and extract US sales data."""
    html = _fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")

    tables = _find_us_section_tables(soup)
    if not tables:
        return []

    for table in tables:
        headers = [th.get_text(strip=True).lower()
                    for th in table.find("tr").find_all(["th", "td"])] if table.find("tr") else []

        has_months = any(h[:3] in {"jan","feb","mar","apr","may","jun",
                                    "jul","aug","sep","oct","nov","dec"}
                         for h in headers)

        if not has_months:
            records = _parse_annual_table(table)
            if records:
                return records

    for table in tables:
        records = _parse_monthly_table(table)
        if records:
            return records

    return []


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            units_sold INTEGER NOT NULL,
            source TEXT DEFAULT 'gcbc',
            source_url TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(year, make, model)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_year ON vehicle_sales(year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_make_model ON vehicle_sales(make, model)")
    conn.commit()
    return conn


def save_records(conn: sqlite3.Connection, records: list[dict], source: str = "gcbc"):
    for r in records:
        conn.execute("""
            INSERT INTO vehicle_sales (year, make, model, units_sold, source, source_url)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(year, make, model) DO UPDATE SET
                units_sold = excluded.units_sold,
                source = excluded.source,
                source_url = excluded.source_url,
                scraped_at = CURRENT_TIMESTAMP
        """, (r["year"], r["make"], r["model"], r["units_sold"],
              source, r.get("source_url", "")))
    conn.commit()


def import_kaggle_csv(conn: sqlite3.Connection):
    """Import data from the existing Kaggle CSV to fill gaps."""
    if not KAGGLE_CSV.exists():
        logger.warning("Kaggle CSV not found at %s", KAGGLE_CSV)
        return 0

    count = 0
    with open(KAGGLE_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        year_cols: list[int] = []
        for col in header[2:]:
            try:
                year_cols.append(int(col))
            except ValueError:
                pass

        for row in reader:
            if len(row) < 3:
                continue

            brand_raw = row[0].strip()
            brand_model_raw = row[1].strip()

            make = brand_raw
            model_part = brand_model_raw
            if model_part.upper().startswith(make.upper() + " "):
                model_part = model_part[len(make) + 1:]
            elif model_part.upper().startswith(make.upper()):
                model_part = model_part[len(make):]
            model_part = model_part.strip()

            if not model_part:
                continue

            make_upper = MAKE_NORMALIZE.get(make.lower(), make.upper())

            for i, yr in enumerate(year_cols):
                val_str = row[2 + i].strip() if (2 + i) < len(row) else "0"
                if val_str in ("", "N/A", "n/a", "-"):
                    continue
                try:
                    val = int(val_str.replace(",", ""))
                    if val > 0:
                        conn.execute("""
                            INSERT INTO vehicle_sales (year, make, model, units_sold, source)
                            VALUES (?, ?, ?, ?, 'kaggle')
                            ON CONFLICT(year, make, model) DO NOTHING
                        """, (yr, make_upper, model_part, val))
                        count += 1
                except ValueError:
                    continue

    conn.commit()
    logger.info("Imported %d records from Kaggle CSV", count)
    return count


def scrape_all():
    """Main scraping routine: model-by-model approach."""
    conn = init_db()

    print("=" * 60)
    print("  Step 1: Importing Kaggle CSV data...")
    print("=" * 60)
    kaggle_count = import_kaggle_csv(conn)
    print(f"  -> Imported {kaggle_count} Kaggle records\n")

    print("=" * 60)
    print("  Step 2: Discovering model pages from GCBC index...")
    print("=" * 60)
    models = discover_model_urls()
    print(f"  -> Found {len(models)} US-brand model pages\n")

    print("=" * 60)
    print("  Step 3: Scraping individual model pages...")
    print("=" * 60)

    total_records = 0
    successes = 0
    failures = 0
    batch_size = 10

    for i, (make, model, url) in enumerate(models):
        label = f"  [{i+1}/{len(models)}] {make} {model}"
        try:
            yearly = scrape_model_page(url)
            if yearly:
                records = [
                    {"year": y["year"], "make": make, "model": model,
                     "units_sold": y["units_sold"], "source_url": url}
                    for y in yearly
                ]
                save_records(conn, records, source="gcbc")
                total_records += len(records)
                successes += 1
                years = sorted(y["year"] for y in yearly)
                print(f"{label}: {len(yearly)} years ({years[0]}-{years[-1]})")
            else:
                print(f"{label}: no US annual sales table found")
                failures += 1
        except Exception as e:
            print(f"{label}: ERROR - {e}")
            failures += 1

        if (i + 1) % batch_size == 0:
            time.sleep(1.5)
        else:
            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Models scraped: {successes} success, {failures} no data")
    print(f"  GCBC records: {total_records}")
    print(f"  Kaggle records: {kaggle_count}")
    print(f"  Database: {DB_PATH}")
    print(f"{'='*60}")

    cursor = conn.execute("""
        SELECT year, COUNT(*) as models, SUM(units_sold) as total_sales
        FROM vehicle_sales
        GROUP BY year
        ORDER BY year
    """)
    print(f"\n{'Year':>6} {'Models':>8} {'Total Sales':>14}")
    print("-" * 32)
    for row in cursor:
        print(f"{row[0]:>6} {row[1]:>8} {row[2]:>14,}")

    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scrape_all()
