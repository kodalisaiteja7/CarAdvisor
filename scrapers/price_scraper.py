"""Vehicle price estimator — MarketCheck API with MSRP-based fallback.

Uses MarketCheck's professional automotive data API:
  1. MarketCheck Price endpoint (ML-predicted, requires VIN)
  2. MarketCheck Inventory Search + Stats (make/model/year/mileage)
  3. fueleconomy.gov MSRP + depreciation curve (fallback)
  4. Segment-average MSRP + depreciation curve (last resort)
"""

from __future__ import annotations

import logging
from datetime import date
from urllib.parse import quote

import requests

from database.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

_SOURCE = "marketcheck_price"

_MC_BASE = "https://api.marketcheck.com/v2"


def _get_mc_key() -> str:
    from config.settings import MARKETCHECK_API_KEY
    return MARKETCHECK_API_KEY

_MIN_LISTINGS = 3

_FUEL_ECO_BASE = "https://www.fueleconomy.gov/ws/rest"


def fetch_avg_price(
    make: str,
    model: str,
    year: int,
    mileage: int,
    trim: str | None = None,
    engine: str | None = None,
    vin: str | None = None,
    zip_code: str | None = None,
) -> dict:
    """Return an estimated average market price -- always returns a result.

    Tries MarketCheck API first (Price endpoint if VIN available, else
    Inventory Search + Stats), then falls back to MSRP depreciation.
    """
    cache_key_parts = [make, model, str(year), str(mileage // 5000)]
    if trim:
        cache_key_parts.append(trim)
    if vin:
        cache_key_parts.append(vin)
    if zip_code:
        cache_key_parts.append(zip_code)
    cache_suffix = "|".join(cache_key_parts)

    cached = get_cached(_SOURCE, make, model, year)
    if cached is not None and cached.get("_cache_suffix") == cache_suffix:
        return cached

    result = None
    search_stats = None
    api_key = _get_mc_key()
    zc = zip_code or "10001"

    if vin and api_key:
        result = _try_mc_price(vin, mileage, api_key, zc)

    if api_key:
        search_stats = _try_mc_search(make, model, year, mileage, trim, api_key, zc)

    if result is None and search_stats is not None:
        result = search_stats
        search_stats = None
    elif result is not None and search_stats is not None:
        result["percentiles"] = search_stats.get("percentiles")
        result["days_on_market"] = search_stats.get("days_on_market")
        result["listings_count"] = search_stats.get("listings_count", 0)
        result["price_range"] = search_stats.get("price_range")

    if result is None:
        result = _estimate_from_msrp(make, model, year, mileage, trim, engine)

    result["_cache_suffix"] = cache_suffix
    set_cached(_SOURCE, make, model, year, result)
    return result


_volume_cache: dict[str, int | None] = {}


def fetch_market_volume(make: str, model: str, year: int) -> int | None:
    """Return the total active listing count for a make/model/year.

    Used as a proxy for sales volume to normalize complaint counts in scoring.
    Higher listings = more popular vehicle = complaints should be weighed
    relative to volume. Returns None if MarketCheck is unavailable.
    """
    cache_key = f"{make}|{model}|{year}"
    if cache_key in _volume_cache:
        return _volume_cache[cache_key]

    api_key = _get_mc_key()
    if not api_key:
        _volume_cache[cache_key] = None
        return None

    try:
        params = {
            "api_key": api_key,
            "car_type": "used",
            "year": str(year),
            "make": make,
            "model": model,
            "rows": "0",
        }
        resp = requests.get(
            f"{_MC_BASE}/search/car/active", params=params, timeout=10,
        )
        resp.raise_for_status()
        count = resp.json().get("num_found", 0)
        _volume_cache[cache_key] = int(count) if count else None
        logger.info("[volume] %d %s %s → %s active listings", year, make, model, count)
        return _volume_cache[cache_key]
    except Exception:
        logger.warning("[volume] Failed to fetch for %d %s %s", year, make, model, exc_info=True)
        _volume_cache[cache_key] = None
        return None


# ------------------------------------------------------------------
# Strategy 1 — MarketCheck Price (ML prediction, requires VIN)
# ------------------------------------------------------------------


def _try_mc_price(vin: str, mileage: int, api_key: str, zip_code: str = "10001") -> dict | None:
    """Use MarketCheck Price endpoint for ML-predicted valuation."""
    try:
        url = f"{_MC_BASE}/predict/car/us/marketcheck_price"
        params = {
            "api_key": api_key,
            "vin": vin,
            "miles": mileage,
            "dealer_type": "independent",
            "zip": zip_code,
        }

        logger.info("[price] MarketCheck Price API: VIN=%s miles=%d", vin, mileage)
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code in (400, 403, 422):
            logger.info(
                "[price] MarketCheck Price unavailable (HTTP %d), falling back to search",
                resp.status_code,
            )
            return None

        resp.raise_for_status()
        data = resp.json()

        mc_price = data.get("marketcheck_price")
        msrp = data.get("msrp")

        if not mc_price or mc_price <= 0:
            return None

        result = {
            "avg_price": int(mc_price),
            "source": "MarketCheck",
            "listings_count": 0,
            "price_range": None,
            "match_level": "ml_predicted",
            "msrp": int(msrp) if msrp else None,
        }

        logger.info(
            "[price] MarketCheck Price: VIN=%s → $%s (MSRP $%s)",
            vin, f"{int(mc_price):,}", f"{int(msrp):,}" if msrp else "N/A",
        )
        return result

    except Exception:
        logger.warning("MarketCheck Price API failed for VIN=%s", vin, exc_info=True)
        return None


# ------------------------------------------------------------------
# Strategy 2 — MarketCheck Inventory Search + Stats
# ------------------------------------------------------------------


def _try_mc_search(
    make: str, model: str, year: int, mileage: int,
    trim: str | None, api_key: str, zip_code: str = "10001",
) -> dict | None:
    """Use MarketCheck Inventory Search to get market stats."""
    try:
        mileage_min = max(0, mileage - 15_000)
        mileage_max = mileage + 15_000

        params: dict = {
            "api_key": api_key,
            "car_type": "used",
            "year": str(year),
            "make": make,
            "model": model,
            "miles_range": f"{mileage_min}-{mileage_max}",
            "rows": "0",
            "stats": "price,miles,dom_active",
        }
        if zip_code and zip_code != "10001":
            params["zip"] = zip_code
        if trim:
            params["trim"] = trim

        url = f"{_MC_BASE}/search/car/active"
        logger.info(
            "[price] MarketCheck Search: %d %s %s (trim=%s, miles=%d±15k, zip=%s)",
            year, make, model, trim, mileage, zip_code,
        )
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        num_found = data.get("num_found", 0)
        stats = data.get("stats", {})
        price_stats = stats.get("price", {})

        if num_found < _MIN_LISTINGS or not price_stats:
            logger.info(
                "[price] MarketCheck Search: only %d listings (min %d)",
                num_found, _MIN_LISTINGS,
            )
            if trim:
                return _try_mc_search(make, model, year, mileage, trim=None,
                                      api_key=api_key, zip_code=zip_code)
            if zip_code != "10001":
                logger.info("[price] Retrying without location filter")
                return _try_mc_search(make, model, year, mileage, trim=None,
                                      api_key=api_key, zip_code="10001")
            return None

        median_price = price_stats.get("median", 0)
        mean_price = price_stats.get("mean", 0)
        avg_price = int(median_price or mean_price)

        if avg_price <= 0:
            return None

        price_min = int(price_stats.get("min", 0))
        price_max = int(price_stats.get("max", 0))
        percentiles = price_stats.get("percentiles", {})

        dom_stats = stats.get("dom_active", {})
        dom_median = dom_stats.get("median")
        dom_mean = dom_stats.get("mean")

        result = {
            "avg_price": avg_price,
            "source": "MarketCheck",
            "listings_count": int(num_found),
            "price_range": {"low": price_min, "high": price_max} if price_min and price_max else None,
            "match_level": "exact" if trim else "broad",
            "percentiles": {
                "p5": int(percentiles.get("5.0", 0)),
                "p25": int(percentiles.get("25.0", 0)),
                "p50": int(percentiles.get("50.0", 0)),
                "p75": int(percentiles.get("75.0", 0)),
                "p90": int(percentiles.get("90.0", 0)),
                "p95": int(percentiles.get("95.0", 0)),
            } if percentiles else None,
            "days_on_market": int(dom_median or dom_mean or 0) if (dom_median or dom_mean) else None,
        }

        logger.info(
            "[price] MarketCheck Search: %d %s %s → $%s median (%d listings, DOM=%s)",
            year, make, model, f"{avg_price:,}", num_found,
            result["days_on_market"],
        )
        return result

    except Exception:
        logger.warning(
            "MarketCheck Search failed for %d %s %s", year, make, model,
            exc_info=True,
        )
        return None


# ------------------------------------------------------------------
# Strategy 3 — MSRP depreciation estimate (guaranteed)
# ------------------------------------------------------------------

_SEGMENT_MSRP: dict[str, int] = {
    "ACURA": 42_000, "ALFA ROMEO": 45_000, "ASTON MARTIN": 155_000,
    "AUDI": 48_000, "BENTLEY": 220_000, "BMW": 55_000,
    "BUICK": 35_000, "CADILLAC": 50_000, "CHEVROLET": 36_000,
    "CHRYSLER": 35_000, "DODGE": 38_000, "FERRARI": 280_000,
    "FIAT": 25_000, "FORD": 38_000, "GENESIS": 45_000,
    "GMC": 45_000, "HONDA": 30_000, "HYUNDAI": 28_000,
    "INFINITI": 45_000, "JAGUAR": 55_000, "JEEP": 40_000,
    "KIA": 28_000, "LAMBORGHINI": 250_000, "LAND ROVER": 60_000,
    "LEXUS": 48_000, "LINCOLN": 50_000, "LOTUS": 80_000,
    "MASERATI": 85_000, "MAZDA": 30_000, "MCLAREN": 220_000,
    "MERCEDES-BENZ": 55_000, "MINI": 32_000, "MITSUBISHI": 26_000,
    "NISSAN": 30_000, "PORSCHE": 75_000, "RAM": 42_000,
    "ROLLS-ROYCE": 350_000, "SUBARU": 32_000, "TESLA": 48_000,
    "TOYOTA": 34_000, "VOLKSWAGEN": 32_000, "VOLVO": 45_000,
}


def _estimate_from_msrp(
    make: str, model: str, year: int, mileage: int,
    trim: str | None, engine: str | None,
) -> dict:
    """Build a depreciation-based estimate.  Always returns a dict."""
    msrp = _lookup_msrp(make, model, year, trim, engine)
    source = "fueleconomy.gov estimate"

    if msrp is None:
        msrp = _SEGMENT_MSRP.get(make.upper(), 35_000)
        source = "depreciation estimate"

    depreciated = _apply_depreciation(msrp, year)
    adjusted = _mileage_adjust(depreciated, year, mileage)

    return {
        "avg_price": adjusted,
        "source": source,
        "listings_count": 0,
        "price_range": None,
        "match_level": "estimate",
    }


def _lookup_msrp(
    make: str, model: str, year: int,
    trim: str | None, engine: str | None,
) -> int | None:
    """Try fueleconomy.gov for the original MSRP (basePrice)."""
    try:
        headers = {"Accept": "application/json"}

        models_url = (
            f"{_FUEL_ECO_BASE}/vehicle/menu/model"
            f"?year={year}&make={quote(make)}"
        )
        resp = requests.get(models_url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("menuItem", [])
        if isinstance(items, dict):
            items = [items]
        if not items:
            return None

        model_lower = model.lower()
        trim_lower = (trim or "").lower()

        best_variant = None
        for item in items:
            text = item.get("text", "")
            if model_lower not in text.lower():
                continue
            if trim_lower and trim_lower in text.lower():
                best_variant = text
                break
            if best_variant is None:
                best_variant = text

        if not best_variant:
            return None

        options_url = (
            f"{_FUEL_ECO_BASE}/vehicle/menu/options"
            f"?year={year}&make={quote(make)}&model={quote(best_variant)}"
        )
        resp = requests.get(options_url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        opts = data.get("menuItem", [])
        if isinstance(opts, dict):
            opts = [opts]
        if not opts:
            return None

        best_id = None
        engine_lower = (engine or "").lower()
        for opt in opts:
            opt_text = opt.get("text", "").lower()
            vid = opt.get("value")
            if engine_lower and engine_lower in opt_text:
                best_id = vid
                break
            if best_id is None:
                best_id = vid

        if not best_id:
            return None

        vehicle_url = f"{_FUEL_ECO_BASE}/vehicle/{best_id}"
        resp = requests.get(vehicle_url, headers=headers, timeout=8)
        resp.raise_for_status()
        vdata = resp.json()
        base_price = vdata.get("basePrice")
        if base_price:
            price = int(float(str(base_price)))
            if price > 0:
                logger.info(
                    "[price] MSRP from fueleconomy.gov: $%s for %s",
                    f"{price:,}", best_variant,
                )
                return price
        return None

    except Exception:
        logger.debug("fueleconomy.gov MSRP lookup failed", exc_info=True)
        return None


def _apply_depreciation(msrp: int, year: int) -> int:
    """Standard used-car depreciation curve applied to MSRP."""
    age = max(date.today().year - year, 0)
    value = float(msrp)
    for yr in range(1, age + 1):
        if yr == 1:
            value *= 0.80
        elif yr <= 5:
            value *= 0.85
        else:
            value *= 0.90
    return max(int(value), 1_000)


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _mileage_adjust(price: int, year: int, mileage: int) -> int:
    """Shift price up/down based on deviation from expected mileage."""
    expected = _expected_mileage(year)
    diff = mileage - expected
    adjustment = int(diff * 0.08)
    return max(price - adjustment, int(price * 0.4))


def _expected_mileage(year: int) -> int:
    age = max(date.today().year - year, 0)
    return age * 12_000
