"""Pre-compute per-model complaint statistics from the NHTSA bulk database.

Generates aggregate stats (complaint counts, mileage percentiles, severity
distributions, crash/fire rates) that calibrate the scoring model and give
the LLM context about what is "normal" for any given vehicle.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, Text, func, case
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine as _sa_create_engine

from data.bulk_loader import (
    BULK_DB_PATH,
    BULK_DB_URL,
    BulkBase,
    NHTSAComplaint,
    _get_bulk_engine,
)

logger = logging.getLogger(__name__)

STATS_CACHE_PATH = Path(__file__).resolve().parent.parent / "dataset" / "bulk_stats_cache.db"


def _get_stats_engine():
    """Return an engine that has model_stats/global_baselines tables.

    Prefers the full nhtsa_bulk.db; falls back to the lightweight
    bulk_stats_cache.db that ships with the repo for production deploys.
    """
    if Path(BULK_DB_PATH).exists():
        return _get_bulk_engine(), True
    if STATS_CACHE_PATH.exists():
        engine = _sa_create_engine(f"sqlite:///{STATS_CACHE_PATH}", echo=False)
        return engine, False
    return None, False


class ModelStats(BulkBase):
    __tablename__ = "model_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    total_complaints = Column(Integer, default=0)
    complaints_percentile = Column(Float, default=50.0)
    system_distribution = Column(Text, default="{}")
    mileage_distribution = Column(Text, default="{}")
    crash_rate = Column(Float, default=0.0)
    fire_rate = Column(Float, default=0.0)
    avg_mileage = Column(Float)
    severity_index = Column(Float, default=0.0)


class GlobalBaseline(BulkBase):
    __tablename__ = "global_baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_key = Column(String(100), unique=True, nullable=False)
    stat_value = Column(Float)
    stat_json = Column(Text)


def build_stats():
    """Pre-compute all model statistics from nhtsa_complaints table."""
    engine = _get_bulk_engine()
    BulkBase.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    ModelStats.__table__.drop(engine, checkfirst=True)
    GlobalBaseline.__table__.drop(engine, checkfirst=True)
    BulkBase.metadata.create_all(engine)

    logger.info("Building per-model statistics...")

    groups = (
        session.query(
            NHTSAComplaint.make,
            NHTSAComplaint.model,
            NHTSAComplaint.year,
            func.count(NHTSAComplaint.id).label("cnt"),
        )
        .group_by(NHTSAComplaint.make, NHTSAComplaint.model, NHTSAComplaint.year)
        .all()
    )

    logger.info("Found %d make/model/year groups", len(groups))

    all_counts = [g.cnt for g in groups]
    all_counts.sort()

    stats_batch: list[ModelStats] = []

    for idx, g in enumerate(groups):
        make, model, year, total = g.make, g.model, g.year, g.cnt

        system_rows = (
            session.query(
                NHTSAComplaint.system,
                func.count(NHTSAComplaint.id),
            )
            .filter_by(make=make, model=model, year=year)
            .group_by(NHTSAComplaint.system)
            .all()
        )
        system_dist = {row[0]: row[1] for row in system_rows}

        mileage_values = [
            r[0]
            for r in session.query(NHTSAComplaint.mileage)
            .filter_by(make=make, model=model, year=year)
            .filter(NHTSAComplaint.mileage.isnot(None))
            .filter(NHTSAComplaint.mileage > 0)
            .filter(NHTSAComplaint.mileage < 500_000)
            .all()
        ]

        mileage_dist = _compute_percentiles(mileage_values)

        crash_count = (
            session.query(func.count(NHTSAComplaint.id))
            .filter_by(make=make, model=model, year=year, crash=True)
            .scalar()
        ) or 0

        fire_count = (
            session.query(func.count(NHTSAComplaint.id))
            .filter_by(make=make, model=model, year=year, fire=True)
            .scalar()
        ) or 0

        crash_rate = crash_count / total if total > 0 else 0
        fire_rate = fire_count / total if total > 0 else 0

        avg_mileage = None
        if mileage_values:
            avg_mileage = sum(mileage_values) / len(mileage_values)

        injury_death_count = (
            session.query(func.count(NHTSAComplaint.id))
            .filter_by(make=make, model=model, year=year)
            .filter(
                (NHTSAComplaint.injured > 0) | (NHTSAComplaint.deaths > 0)
            )
            .scalar()
        ) or 0
        severity_index = (crash_count * 2 + fire_count * 3 + injury_death_count * 5) / max(total, 1)

        percentile = _percentile_rank(all_counts, total)

        stats_batch.append(
            ModelStats(
                make=make,
                model=model,
                year=year,
                total_complaints=total,
                complaints_percentile=round(percentile, 1),
                system_distribution=json.dumps(system_dist),
                mileage_distribution=json.dumps(mileage_dist),
                crash_rate=round(crash_rate, 4),
                fire_rate=round(fire_rate, 4),
                avg_mileage=round(avg_mileage, 0) if avg_mileage else None,
                severity_index=round(severity_index, 4),
            )
        )

        if len(stats_batch) >= 500:
            session.bulk_save_objects(stats_batch)
            session.commit()
            stats_batch.clear()

        if (idx + 1) % 5000 == 0:
            logger.info("  ... processed %d/%d groups", idx + 1, len(groups))

    if stats_batch:
        session.bulk_save_objects(stats_batch)
        session.commit()

    _build_global_baselines(session, all_counts)

    session.close()
    logger.info("Stats build complete: %d model-year entries", len(groups))


def _compute_percentiles(values: list[int | float]) -> dict:
    """Compute p10, p25, median, p75, p90 from a list of numeric values."""
    if not values:
        return {}
    values.sort()
    n = len(values)

    def pct(p: float) -> float:
        k = (p / 100) * (n - 1)
        f = math.floor(k)
        c = min(f + 1, n - 1)
        if f == c:
            return values[f]
        return values[f] + (k - f) * (values[c] - values[f])

    return {
        "count": n,
        "p10": round(pct(10)),
        "p25": round(pct(25)),
        "median": round(pct(50)),
        "p75": round(pct(75)),
        "p90": round(pct(90)),
        "min": values[0],
        "max": values[-1],
    }


def _percentile_rank(sorted_all: list[int], value: int) -> float:
    """Where does `value` sit in the sorted list? Returns 0-100."""
    n = len(sorted_all)
    if n == 0:
        return 50.0
    count_below = 0
    for v in sorted_all:
        if v < value:
            count_below += 1
        else:
            break
    return (count_below / n) * 100


def _build_global_baselines(session, all_counts: list[int]):
    """Store global baseline stats for normalization."""
    if not all_counts:
        return

    n = len(all_counts)
    median_idx = n // 2
    median_complaints = all_counts[median_idx]
    mean_complaints = sum(all_counts) / n

    baselines = [
        GlobalBaseline(stat_key="median_complaints_per_model_year", stat_value=median_complaints),
        GlobalBaseline(stat_key="mean_complaints_per_model_year", stat_value=mean_complaints),
        GlobalBaseline(stat_key="total_model_years", stat_value=n),
        GlobalBaseline(stat_key="total_complaints", stat_value=sum(all_counts)),
        GlobalBaseline(
            stat_key="complaint_distribution",
            stat_json=json.dumps(_compute_percentiles(all_counts)),
        ),
    ]
    session.bulk_save_objects(baselines)
    session.commit()
    logger.info(
        "Global baselines: median=%d, mean=%.1f across %d model-years",
        median_complaints, mean_complaints, n,
    )


def get_model_stats(make: str, model: str, year: int) -> dict | None:
    """Look up pre-computed stats for a specific vehicle. Returns None if not found."""
    engine, _ = _get_stats_engine()
    if engine is None:
        return None

    BulkBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    make_upper = make.strip().upper()
    model_upper = model.strip().upper()

    row = (
        session.query(ModelStats)
        .filter_by(make=make_upper, model=model_upper, year=year)
        .first()
    )

    if not row:
        row = (
            session.query(ModelStats)
            .filter(
                func.upper(ModelStats.make) == make_upper,
                func.upper(ModelStats.model).contains(model_upper),
                ModelStats.year == year,
            )
            .first()
        )

    if not row:
        session.close()
        return None

    global_median = (
        session.query(GlobalBaseline.stat_value)
        .filter_by(stat_key="median_complaints_per_model_year")
        .scalar()
    ) or 10

    global_mean = (
        session.query(GlobalBaseline.stat_value)
        .filter_by(stat_key="mean_complaints_per_model_year")
        .scalar()
    ) or 20

    session.close()

    ratio = row.total_complaints / max(global_median, 1)
    if ratio > 3:
        interpretation = "Significantly above average complaint volume"
    elif ratio > 1.5:
        interpretation = "Above average complaint volume"
    elif ratio > 0.7:
        interpretation = "Average complaint volume"
    elif ratio > 0.3:
        interpretation = "Below average complaint volume"
    else:
        interpretation = "Very low complaint volume"

    return {
        "make": row.make,
        "model": row.model,
        "year": row.year,
        "total_complaints": row.total_complaints,
        "complaints_percentile": row.complaints_percentile,
        "system_distribution": json.loads(row.system_distribution or "{}"),
        "mileage_distribution": json.loads(row.mileage_distribution or "{}"),
        "crash_rate": row.crash_rate,
        "fire_rate": row.fire_rate,
        "avg_mileage": row.avg_mileage,
        "severity_index": row.severity_index,
        "global_median_complaints": global_median,
        "global_mean_complaints": global_mean,
        "complaint_ratio": round(ratio, 2),
        "interpretation": interpretation,
    }


def get_mileage_curve(make: str, model: str, year: int, system: str) -> dict | None:
    """Get mileage failure distribution for a specific vehicle+system."""
    curves = get_all_mileage_curves(make, model, year)
    return curves.get(system)


def get_all_mileage_curves(
    make: str, model: str, year: int,
) -> dict[str, dict]:
    """Fetch mileage curves for ALL systems in a single query.

    Returns {system: {count, p10, p25, median, p75, p90, min, max}}.
    """
    if not Path(BULK_DB_PATH).exists():
        return {}

    engine = _get_bulk_engine()
    BulkBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    make_upper = make.strip().upper()
    model_upper = model.strip().upper()

    rows = (
        session.query(NHTSAComplaint.system, NHTSAComplaint.mileage)
        .filter(
            NHTSAComplaint.make == make_upper,
            NHTSAComplaint.model == model_upper,
            NHTSAComplaint.year.between(year - 3, year + 3),
            NHTSAComplaint.mileage.isnot(None),
            NHTSAComplaint.mileage > 0,
            NHTSAComplaint.mileage < 500_000,
        )
        .all()
    )
    session.close()

    by_system: dict[str, list[int]] = {}
    for system, mileage_val in rows:
        by_system.setdefault(system, []).append(mileage_val)

    curves: dict[str, dict] = {}
    for system, values in by_system.items():
        if len(values) >= 5:
            curves[system] = _compute_percentiles(values)

    return curves


def get_calibrated_weights(make: str, model: str, year: int) -> dict | None:
    """Derive calibrated severity weights from bulk data for this vehicle.

    For vehicles with higher crash/fire rates, safety_impact weight increases.
    For vehicles with more expensive typical repairs, repair_cost weight increases.
    Falls back to None if no bulk data is available.
    """
    stats = get_model_stats(make, model, year)
    if not stats:
        return None

    crash_rate = stats.get("crash_rate", 0)
    fire_rate = stats.get("fire_rate", 0)
    severity_idx = stats.get("severity_index", 0)

    base = {
        "complaint_count": 0.30,
        "severity": 0.25,
        "safety_impact": 0.25,
        "repair_cost": 0.20,
    }

    if crash_rate > 0.05 or fire_rate > 0.02 or severity_idx > 1.0:
        safety_boost = min(0.10, (crash_rate * 0.5 + fire_rate * 1.0 + severity_idx * 0.02))
        base["safety_impact"] += safety_boost
        base["complaint_count"] -= safety_boost / 2
        base["repair_cost"] -= safety_boost / 2

    total = sum(base.values())
    return {k: round(v / total, 4) for k, v in base.items()}


def get_complaint_baseline(make: str, model: str, year: int) -> dict | None:
    """Get complaint count normalization baseline for this vehicle.

    Returns the vehicle's complaint count, the global median, and a ratio
    so the scorer can normalize "is X complaints a lot?"
    """
    stats = get_model_stats(make, model, year)
    if not stats:
        return None

    return {
        "this_model_complaints": stats["total_complaints"],
        "average_model_complaints": stats["global_mean_complaints"],
        "median_model_complaints": stats["global_median_complaints"],
        "percentile": stats["complaints_percentile"],
        "interpretation": stats["interpretation"],
        "complaint_ratio": stats["complaint_ratio"],
    }
