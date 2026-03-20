"""Risk Score 2 -- Multi-source blended reliability scoring engine.

Combines the existing NHTSA complaint score with three additional
signal sources (TSBs, Investigations, Manufacturer Communications)
and Dashboard Light brand-level QIR to produce a more balanced risk
assessment.

This module does NOT modify the existing scorer.py or the frontend.

Scoring formula (weighted blend, adaptive):
    When NHTSA has data (score > 10):
        NHTSA Complaint Score  35%   existing scorer.py
        TSB Severity Index     25%   vehicle_signals.db tsb_counts
        Investigation Risk     15%   vehicle_signals.db investigations
        Mfr Comm Density       10%   vehicle_signals.db mfr_comm_counts
        Dashboard Light QIR    15%   vehicle_signals.db dashboard_light
    When NHTSA has insufficient data (score <= 10):
        NHTSA weight drops to 0% and is redistributed proportionally
        to TSB (38.5%), Investigation (23%), Mfr (15.4%), DL (23.1%)

Usage:
    from analysis.scorer_v2 import score_vehicle_v2
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "dataset" / "vehicle_signals.db"

import threading
_local = threading.local()

CRITICAL_SYSTEMS = {"Engine", "Transmission", "Brakes", "Steering", "Fuel System"}
PROACTIVE_SYSTEMS = {"Electrical", "HVAC", "Interior", "Body/Paint", "Other"}
TSB_CRITICAL_WEIGHT = 2.5
TSB_NORMAL_WEIGHT = 1.0
TSB_PROACTIVE_WEIGHT = 0.4

INV_SEVERITY = {"RQ": 80, "EA": 60, "PE": 30, "DP": 15}
INV_COMPONENT_BOOST = {"Engine": 1.5, "Transmission": 1.4, "Brakes": 1.2, "Steering": 1.2, "Fuel System": 1.1}

_BRAND_TSB_MEDIAN: dict[str, float] = {}
_BRAND_MFR_MEDIAN: dict[str, float] = {}
_INDUSTRY_TSB_MEDIAN: float = 50.0
_INDUSTRY_MFR_MEDIAN: float = 50.0
_BRAND_STATS_LOADED = False

_TIER_1_BRANDS = {"TOYOTA", "LEXUS", "HONDA", "ACURA", "MAZDA"}
_TIER_3_BRANDS = {
    "BMW", "AUDI", "MERCEDES-BENZ", "JEEP", "CHRYSLER", "DODGE",
    "LAND ROVER", "ALFA ROMEO", "FIAT", "JAGUAR", "MASERATI",
    "MINI", "VOLVO",
}

_BRAND_WEAR_MAX: dict[str, float] = {}
for _b in _TIER_1_BRANDS:
    _BRAND_WEAR_MAX[_b] = 1.45
for _b in _TIER_3_BRANDS:
    _BRAND_WEAR_MAX[_b] = 1.80


@dataclass
class ScoreV2Result:
    risk_score_v2: float
    letter_grade: str
    nhtsa_component: float
    tsb_component: float
    investigation_component: float
    mfr_comm_component: float
    dl_qir_component: float
    tsb_raw_count: int
    inv_raw_count: int
    mfr_comm_raw_count: int
    dl_qir: int | None
    wear_factor: float = 1.0
    mileage_floor: float = 0.0
    weighted_contributions: dict | None = None


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"vehicle_signals.db not found at {DB_PATH}. "
                "Run `python -m data.preprocess_signals` first."
            )
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.conn = conn
    return conn


def _load_brand_stats():
    """Compute and cache brand-level median TSB/MFR counts.

    The brand median becomes the denominator in the log-curve scoring,
    so each vehicle is scored relative to what's *normal for its brand*.
    A Porsche with 300 TSBs (close to Porsche median 326) scores ~17,
    while a Honda with 300 TSBs (way above Honda median 5) scores much higher.
    """
    global _BRAND_TSB_MEDIAN, _BRAND_MFR_MEDIAN, _BRAND_STATS_LOADED
    global _INDUSTRY_TSB_MEDIAN, _INDUSTRY_MFR_MEDIAN
    if _BRAND_STATS_LOADED:
        return

    from collections import defaultdict
    import statistics as _stats

    conn = _get_conn()

    rows = conn.execute(
        "SELECT make, SUM(tsb_count) "
        "FROM tsb_counts GROUP BY make, model, year"
    ).fetchall()
    brand_tsb = defaultdict(list)
    for make, total in rows:
        brand_tsb[make].append(total)

    for make, counts in brand_tsb.items():
        _BRAND_TSB_MEDIAN[make] = max(1.0, _stats.median(counts))

    all_medians = list(_BRAND_TSB_MEDIAN.values())
    _INDUSTRY_TSB_MEDIAN = max(1.0, _stats.median(all_medians)) if all_medians else 50.0

    rows2 = conn.execute(
        "SELECT make, comm_count FROM mfr_comm_counts"
    ).fetchall()
    brand_mfr = defaultdict(list)
    for make, cnt in rows2:
        brand_mfr[make].append(cnt)

    for make, counts in brand_mfr.items():
        _BRAND_MFR_MEDIAN[make] = max(1.0, _stats.median(counts))

    all_mfr_medians = list(_BRAND_MFR_MEDIAN.values())
    _INDUSTRY_MFR_MEDIAN = max(1.0, _stats.median(all_mfr_medians)) if all_mfr_medians else 50.0

    _BRAND_STATS_LOADED = True
    logger.info(
        "Brand stats loaded: %d TSB brands (ind. median=%.0f), "
        "%d MFR brands (ind. median=%.0f)",
        len(_BRAND_TSB_MEDIAN), _INDUSTRY_TSB_MEDIAN,
        len(_BRAND_MFR_MEDIAN), _INDUSTRY_MFR_MEDIAN,
    )


_SERIES_PREFIXES: dict[tuple[str, str], list[str]] = {
    ("BMW", "3 SERIES"): ["320", "328", "330", "335", "340", "M3"],
    ("BMW", "5 SERIES"): ["530", "535", "540", "545", "550", "M5"],
    ("BMW", "7 SERIES"): ["740", "745", "750", "760", "M7"],
    ("BMW", "4 SERIES"): ["430", "435", "440", "M4"],
    ("BMW", "6 SERIES"): ["630", "640", "645", "650", "M6"],
    ("BMW", "2 SERIES"): ["220", "228", "230", "235", "240", "M2"],
    ("MERCEDES-BENZ", "C-CLASS"): ["C ", "C-", "AMG C"],
    ("MERCEDES-BENZ", "E-CLASS"): ["E ", "E-", "AMG E"],
    ("MERCEDES-BENZ", "S-CLASS"): ["S ", "S-", "AMG S"],
    ("MERCEDES-BENZ", "GLC"): ["GLC", "AMG GLC"],
    ("MERCEDES-BENZ", "GLE"): ["GLE", "AMG GLE"],
    ("MERCEDES-BENZ", "GLS"): ["GLS", "AMG GLS"],
    ("MERCEDES-BENZ", "CLA"): ["CLA", "AMG CLA"],
    ("MERCEDES-BENZ", "GLA"): ["GLA", "AMG GLA"],
    ("LAND ROVER", "EVOQUE"): ["RANGE ROVER EVOQUE", "EVOQUE"],
    ("LAND ROVER", "DISCOVERY SPORT"): ["DISCOVERY SPORT"],
    ("LAND ROVER", "RANGE ROVER SPORT"): ["RANGE ROVER SPORT"],
}


def _normalize_model(model: str) -> str:
    """Strip body-style suffixes that NHTSA adds but TSB data doesn't use.

    E.g. 'RANGE ROVER EVOQUE (5-DOOR)' → 'RANGE ROVER EVOQUE'
         'CIVIC (COUPE)'               → 'CIVIC'
    """
    import re
    return re.sub(r"\s*\([^)]*\)\s*$", "", model).strip()


def _build_model_where(make: str, model: str) -> tuple[str, list]:
    """Build a SQL WHERE clause + params for fuzzy model matching.

    Handles four patterns:
    1. Exact match (always tried)
    2. Substring/prefix match (model appears in DB model, or vice versa)
    3. Reverse contains (DB model appears in input; catches NHTSA suffixes)
    4. Series prefix expansion (BMW '3 SERIES' → '320%', '330%', etc.)

    Also normalizes input by stripping parenthetical body-style suffixes
    like (5-DOOR), (CONVERTIBLE), (COUPE) that NHTSA adds.
    """
    make_u = make.upper()
    model_u = model.upper()
    model_clean = _normalize_model(model_u)

    conditions = [
        "model = ?",
        "model LIKE ? || '%'",
        "model LIKE '%' || ? || '%'",
        "? LIKE '%' || model || '%'",
    ]
    params: list[str] = [model_u, model_u, model_u, model_u]

    if model_clean != model_u:
        conditions.extend([
            "model = ?",
            "model LIKE ? || '%'",
            "model LIKE '%' || ? || '%'",
        ])
        params.extend([model_clean, model_clean, model_clean])

    prefixes = _SERIES_PREFIXES.get((make_u, model_u))
    if not prefixes and model_clean != model_u:
        prefixes = _SERIES_PREFIXES.get((make_u, model_clean))
    if prefixes:
        for pfx in prefixes:
            conditions.append("model LIKE ? || '%'")
            params.append(pfx)

    return "(" + " OR ".join(conditions) + ")", params


def _get_tsb_counts(make: str, model: str, year: int) -> dict[str, int]:
    """Get TSB counts by system for a vehicle, with fuzzy model matching."""
    conn = _get_conn()
    model_clause, model_params = _build_model_where(make, model)
    params = [make.upper(), year] + model_params
    rows = conn.execute(
        f"SELECT system, SUM(tsb_count) FROM tsb_counts "
        f"WHERE make=? AND year=? AND {model_clause} GROUP BY system",
        params,
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _get_investigations(make: str, model: str, year: int) -> list[tuple[str, str]]:
    """Get investigation (inv_type, inv_id) pairs, with fuzzy model matching."""
    conn = _get_conn()
    model_clause, model_params = _build_model_where(make, model)
    params = [make.upper(), year] + model_params
    rows = conn.execute(
        f"SELECT DISTINCT inv_type, inv_id FROM investigations "
        f"WHERE make=? AND year=? AND {model_clause}",
        params,
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _get_mfr_comm_count(make: str, model: str, year: int) -> int:
    """Get manufacturer communication count, with fuzzy model matching."""
    conn = _get_conn()
    model_clause, model_params = _build_model_where(make, model)
    params = [make.upper(), year] + model_params
    row = conn.execute(
        f"SELECT SUM(comm_count) FROM mfr_comm_counts "
        f"WHERE make=? AND year=? AND {model_clause}",
        params,
    ).fetchone()
    return row[0] if row[0] else 0


def _get_dl_qir(make: str) -> int | None:
    """Get Dashboard Light QIR for a make."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT qir FROM dashboard_light WHERE make=?",
        (make.upper(),),
    ).fetchone()
    return row[0] if row else None


def _signal_wear_factor(mileage: int, make: str = "") -> float:
    """Mileage-based multiplier for non-NHTSA signals (TSB/INV/MFR/DL).

    Uses a power curve anchored at 75k miles (factor = 1.0). Below 75k the
    factor drops sharply; a 10k-mile car gets signals cut by 55-70%.
    Above 75k the factor rises to max_f at 200k.

    The exponent is derived so that factor(200k) = max_f exactly:
        p = ln(max_f) / ln(200000 / 75000)

    Approximate values (default brand, max_f=1.60):
        10k → 0.38    25k → 0.57    50k → 0.80    75k → 1.00
       100k → 1.15   150k → 1.39   200k → 1.60
    """
    max_f = _BRAND_WEAR_MAX.get(make.upper(), 1.60)
    ref_mi = 75_000
    p = math.log(max_f) / math.log(200_000 / ref_mi)
    raw = (max(mileage, 1_000) / ref_mi) ** p
    return round(min(max_f, max(0.25, raw)), 3)


def _mileage_baseline(mileage: int) -> float:
    """Absolute mileage-based risk floor. Even a perfect model carries
    wear risk at high mileage."""
    if mileage <= 30_000:
        return 0.0
    if mileage <= 60_000:
        return 5.0 * ((mileage - 30_000) / 30_000)
    if mileage <= 100_000:
        return 5.0 + 7.0 * ((mileage - 60_000) / 40_000)
    if mileage <= 150_000:
        return 12.0 + 8.0 * ((mileage - 100_000) / 50_000)
    if mileage <= 200_000:
        return 20.0 + 10.0 * ((mileage - 150_000) / 50_000)
    return 30.0


def _compute_tsb_index(tsb_by_system: dict[str, int], make: str = "") -> tuple[float, int]:
    """Compute TSB Severity Index (0-100) with three normalization layers:

    1. Content filtering: critical systems 2.5x, proactive systems 0.4x
       (firmware updates, cosmetic TSBs weighted down)
    2. Brand baseline: score = log1p(weighted / brand_median) so each
       vehicle is measured against what's normal for its brand
    3. Coefficient 25 keeps the 0-100 range well-distributed:
       - At brand median: score ~17 (normal)
       - At 3x median:    score ~35 (elevated)
       - At 10x median:   score ~60 (high)
    """
    if not tsb_by_system:
        return 0.0, 0

    _load_brand_stats()

    weighted_total = 0.0
    raw_total = 0
    for system, count in tsb_by_system.items():
        if system in CRITICAL_SYSTEMS:
            weight = TSB_CRITICAL_WEIGHT
        elif system in PROACTIVE_SYSTEMS:
            weight = TSB_PROACTIVE_WEIGHT
        else:
            weight = TSB_NORMAL_WEIGHT
        weighted_total += count * weight
        raw_total += count

    raw_med = _BRAND_TSB_MEDIAN.get(make.upper(), _INDUSTRY_TSB_MEDIAN)
    brand_med = max(raw_med, 50.0)
    score = min(100.0, 25.0 * math.log1p(weighted_total / brand_med))

    return round(score, 1), raw_total


def _compute_investigation_risk(
    investigations: list[tuple[str, str]],
    make: str,
    model: str,
    year: int,
) -> tuple[float, int]:
    """Compute Investigation Risk score (0-100).

    Each investigation adds its type-based severity score, boosted
    by 1.2-1.5x for Engine/Transmission/Brakes/Steering investigations.
    """
    if not investigations:
        return 0.0, 0

    conn = _get_conn()
    model_clause, model_params = _build_model_where(make, model)
    params = [make.upper(), year] + model_params

    inv_components: dict[str, str] = {}
    for row in conn.execute(
        f"SELECT DISTINCT inv_id, component FROM investigations "
        f"WHERE make=? AND year=? AND {model_clause}",
        params,
    ).fetchall():
        inv_components[row[0]] = row[1]

    seen_ids: set[str] = set()
    total_score = 0.0

    for inv_type, inv_id in investigations:
        if inv_id in seen_ids:
            continue
        seen_ids.add(inv_id)
        base = INV_SEVERITY.get(inv_type, 15)
        component = inv_components.get(inv_id, "Other")
        boost = INV_COMPONENT_BOOST.get(component, 1.0)
        total_score += base * boost

    score = min(100.0, total_score)
    return round(score, 1), len(seen_ids)


def _compute_mfr_comm_density(comm_count: int, make: str = "") -> float:
    """Compute Manufacturer Communication Density score (0-100).

    Uses brand median as denominator so prolific communicators
    (GM, Porsche, Audi) aren't penalized for routine dealer comms.
    A count equal to the brand median scores ~17; a count 10x
    above median scores ~60.
    """
    if comm_count == 0:
        return 0.0

    _load_brand_stats()

    raw_med = _BRAND_MFR_MEDIAN.get(make.upper(), _INDUSTRY_MFR_MEDIAN)
    brand_med = max(raw_med, 50.0)
    score = min(100.0, 25.0 * math.log1p(comm_count / brand_med))

    return round(score, 1)


def _compute_dl_component(qir: int | None) -> float:
    """Invert QIR so high = bad risk. QIR 0-100 where high = reliable."""
    if qir is None:
        return 50.0
    return round(100.0 - qir, 1)


GRADE_THRESHOLDS = [
    (20, "A"),
    (30, "B"),
    (45, "C"),
    (60, "D"),
    (75, "E"),
    (100, "F"),
]


def _letter_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score <= threshold:
            return grade
    return "F"


def score_vehicle_v2(
    nhtsa_risk_score: float,
    make: str,
    model: str,
    year: int,
    mileage: int = 75_000,
    min_score: float = 0.0,
) -> ScoreV2Result:
    """Compute Risk Score 2 by blending NHTSA score with additional signals.

    Parameters
    ----------
    nhtsa_risk_score : float
        The existing Risk Score 1 from scorer.py (0-100).
    make, model, year : str, str, int
        Vehicle identification for looking up signals in vehicle_signals.db.
    mileage : int
        Current mileage -- used for wear factor scaling and baseline floor.
    min_score : float
        Minimum score (for monotonicity enforcement across mileage points).
    """
    tsb_by_system = _get_tsb_counts(make, model, year)
    investigations = _get_investigations(make, model, year)
    mfr_comm_count = _get_mfr_comm_count(make, model, year)
    dl_qir = _get_dl_qir(make)

    tsb_score, tsb_raw = _compute_tsb_index(tsb_by_system, make)
    inv_score, inv_raw = _compute_investigation_risk(
        investigations, make, model, year,
    )
    mfr_score = _compute_mfr_comm_density(mfr_comm_count, make)
    dl_score = _compute_dl_component(dl_qir)

    nhtsa_capped = min(100.0, max(0.0, nhtsa_risk_score))

    wear = _signal_wear_factor(mileage, make)

    # Adaptive weighting: if NHTSA has insufficient complaint data
    # (score <= 10 means essentially no/minimal complaints found),
    # redistribute the NHTSA weight to the other signals so sparse
    # complaint data doesn't suppress the score for low-volume vehicles.
    nhtsa_has_data = nhtsa_capped > 10.0
    if nhtsa_has_data:
        w_nhtsa = 0.35
        w_tsb, w_inv, w_mfr, w_dl = 0.25, 0.15, 0.10, 0.15
    else:
        w_nhtsa = 0.0
        w_tsb, w_inv, w_mfr, w_dl = 0.385, 0.230, 0.154, 0.231

    nhtsa_weighted = w_nhtsa * nhtsa_capped
    signal_blend = (
        w_tsb * tsb_score
        + w_inv * inv_score
        + w_mfr * mfr_score
        + w_dl * dl_score
    )
    signal_scaled = signal_blend * wear

    risk_score_2 = nhtsa_weighted + signal_scaled

    floor = _mileage_baseline(mileage)
    risk_score_2 = max(risk_score_2, floor, min_score)

    risk_score_2 = round(min(100.0, max(0.0, risk_score_2)), 1)

    weighted_contribs = {
        "nhtsa": round(nhtsa_weighted, 2),
        "tsb": round(w_tsb * tsb_score * wear, 2),
        "investigation": round(w_inv * inv_score * wear, 2),
        "mfr_comm": round(w_mfr * mfr_score * wear, 2),
        "dashboard_light": round(w_dl * dl_score * wear, 2),
        "mileage_floor": round(floor, 2),
    }

    return ScoreV2Result(
        risk_score_v2=risk_score_2,
        letter_grade=_letter_grade(risk_score_2),
        nhtsa_component=round(nhtsa_capped, 1),
        tsb_component=tsb_score,
        investigation_component=inv_score,
        mfr_comm_component=mfr_score,
        dl_qir_component=dl_score,
        tsb_raw_count=tsb_raw,
        inv_raw_count=inv_raw,
        mfr_comm_raw_count=mfr_comm_count,
        dl_qir=dl_qir,
        wear_factor=round(wear, 3),
        mileage_floor=round(floor, 1),
        weighted_contributions=weighted_contribs,
    )


def get_v2_signal_details(make: str, model: str, year: int) -> dict:
    """Return detailed V2 signal data for frontend visualization."""
    tsb_by_system = _get_tsb_counts(make, model, year)
    investigations = _get_investigations(make, model, year)
    mfr_comm_count = _get_mfr_comm_count(make, model, year)
    dl_qir = _get_dl_qir(make)

    tsb_systems = []
    for system, count in sorted(tsb_by_system.items(), key=lambda x: x[1], reverse=True):
        category = "critical" if system in CRITICAL_SYSTEMS else (
            "proactive" if system in PROACTIVE_SYSTEMS else "normal"
        )
        tsb_systems.append({"system": system, "count": count, "category": category})

    inv_list = []
    seen = set()
    for inv_type, inv_id in investigations:
        if inv_id in seen:
            continue
        seen.add(inv_id)
        severity = INV_SEVERITY.get(inv_type, 15)
        type_label = {"RQ": "Recall Query", "EA": "Engineering Analysis",
                      "PE": "Preliminary Evaluation", "DP": "Default/Petition"}.get(inv_type, inv_type)
        inv_list.append({"type": inv_type, "type_label": type_label,
                         "id": inv_id, "severity": severity})

    return {
        "tsb_by_system": tsb_systems,
        "tsb_total": sum(tsb_by_system.values()),
        "investigations": inv_list,
        "inv_count": len(inv_list),
        "mfr_comm_count": mfr_comm_count,
        "dl_qir": dl_qir,
        "dl_qir_label": (
            "Excellent" if dl_qir and dl_qir >= 80 else
            "Good" if dl_qir and dl_qir >= 60 else
            "Average" if dl_qir and dl_qir >= 40 else
            "Below Average" if dl_qir and dl_qir >= 20 else
            "Poor" if dl_qir is not None else "No Data"
        ),
    }
