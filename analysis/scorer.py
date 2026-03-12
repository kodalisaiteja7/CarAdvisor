"""Severity & frequency scoring engine.

Produces a composite Reliability Risk Score (0-100) for a vehicle at a
given mileage, ranks the top issues, and assigns a letter grade.

When NHTSA bulk data is available, uses calibrated severity weights and
normalizes complaint counts against per-model baselines instead of using
fixed thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from analysis.mileage_model import MileageAnalysis, MileageClassifiedProblem, MileagePhase
from config.settings import SEVERITY_WEIGHTS

logger = logging.getLogger(__name__)

_cached_weights: dict[str, dict | None] = {}
_cached_baseline: dict[str, dict | None] = {}

GRADE_THRESHOLDS = [
    (15, "A"),
    (30, "B"),
    (50, "C"),
    (70, "D"),
    (100, "F"),
]

DEFAULT_REPAIR_COSTS = {
    "Engine": (1500, 4000),
    "Transmission": (1200, 3500),
    "Electrical": (200, 800),
    "Suspension": (300, 1200),
    "Brakes": (200, 800),
    "Body/Paint": (300, 2000),
    "Interior": (100, 500),
    "HVAC": (300, 1200),
    "Steering": (400, 1500),
    "Fuel System": (300, 1000),
    "Exhaust": (200, 800),
    "Cooling": (400, 1200),
}

SYSTEM_CRITICALITY = {
    "Engine": 1.4,
    "Transmission": 1.2
}


@dataclass
class ScoredProblem:
    classified: MileageClassifiedProblem
    weighted_score: float
    rank: int = 0

    @property
    def probability(self) -> str:
        if self.weighted_score >= 6:
            return "High"
        if self.weighted_score >= 3:
            return "Medium"
        return "Low"


@dataclass
class VehicleScore:
    reliability_risk_score: float  # 0-100
    safety_score: float  # 0-100, higher = more dangerous
    letter_grade: str
    top_issues: list[ScoredProblem] = field(default_factory=list)
    total_problems: int = 0
    mileage_analysis: MileageAnalysis | None = None


def _estimate_repair_cost(problem) -> float:
    """Return an average repair cost, using defaults when data is missing."""
    if problem.repair_cost_low is not None and problem.repair_cost_high is not None:
        return (problem.repair_cost_low + problem.repair_cost_high) / 2

    defaults = DEFAULT_REPAIR_COSTS.get(problem.category, (300, 1000))
    return (defaults[0] + defaults[1]) / 2


def _get_bulk_weights(make: str, model: str, year: int) -> dict | None:
    """Fetch calibrated severity weights from bulk data (cached)."""
    cache_key = f"{make}|{model}|{year}"
    if cache_key in _cached_weights:
        return _cached_weights[cache_key]
    try:
        from data.stats_builder import get_calibrated_weights
        w = get_calibrated_weights(make, model, year)
    except Exception:
        w = None
    _cached_weights[cache_key] = w
    return w


def _get_bulk_baseline(make: str, model: str, year: int) -> dict | None:
    """Fetch complaint count baseline from bulk data (cached)."""
    cache_key = f"{make}|{model}|{year}"
    if cache_key in _cached_baseline:
        return _cached_baseline[cache_key]
    try:
        from data.stats_builder import get_complaint_baseline
        b = get_complaint_baseline(make, model, year)
    except Exception:
        b = None
    _cached_baseline[cache_key] = b
    return b


def _score_single(
    cp: MileageClassifiedProblem,
    weights: dict | None = None,
    baseline: dict | None = None,
) -> float:
    """Compute a weighted score for one classified problem (0-10 scale).

    When bulk data is available:
    - Uses calibrated severity weights instead of hand-tuned defaults
    - Normalizes complaint counts against the per-model baseline
    """
    p = cp.problem

    if baseline and baseline.get("median_model_complaints"):
        median = baseline["median_model_complaints"]
        count_norm = min(10.0, (p.complaint_count / max(median, 1)) * 5)
    else:
        count_norm = min(10.0, p.complaint_count / 50 * 10)

    severity_norm = p.severity
    safety_norm = p.safety_impact

    avg_cost = _estimate_repair_cost(p)
    cost_norm = min(10.0, avg_cost / 500)

    w = weights or SEVERITY_WEIGHTS
    raw = (
        w["complaint_count"] * count_norm
        + w["severity"] * severity_norm
        + w["safety_impact"] * safety_norm
        + w["repair_cost"] * cost_norm
    )

    system_crit = SYSTEM_CRITICALITY.get(p.category, 1.0)
    raw *= system_crit

    phase_multiplier = {
        MileagePhase.CURRENT: 1.2,
        MileagePhase.UPCOMING: 0.8,
        MileagePhase.PAST: 1.0,
        MileagePhase.FUTURE: 0.5,
        MileagePhase.UNKNOWN: 0.6,
    }
    raw *= phase_multiplier.get(cp.phase, 1.0)
    raw *= cp.relevance_score

    return min(10.0, raw)


def _mileage_wear_factor(mileage: int) -> float:
    """Higher mileage = more accumulated wear = higher baseline risk.

    Range: ~0.5x at 0 miles to ~2.0x at 150k+ miles.
    This makes the same vehicle score roughly 3-4x higher at 150k vs 30k.
    """
    factor = 0.5 + 1.5 * min(1.0, mileage / 150_000)
    return round(min(2.0, factor), 3)


def _compute_inherent_risk(scored: list[ScoredProblem], mileage: int) -> float:
    """Compute a mileage-scaled floor based on inherent vehicle risk.

    When all problems are PAST (high mileage), per-problem scores drop
    because relevance is low. But a car that has been through ALL failure
    zones carries accumulated mechanical wear. This function computes a
    baseline from raw severity data (ignoring mileage phase entirely)
    and scales it with mileage to guarantee monotonically increasing risk.

    The formula: floor = avg_severity * (base + growth * mileage_ratio)
    This ensures a car at 150k with 10 known issues is never rated safer
    than the same car at 25k.
    """
    if not scored:
        return 0.0

    raw_severities = []
    total_complaints = 0
    for sp in scored:
        p = sp.classified.problem
        crit = SYSTEM_CRITICALITY.get(p.category, 1.0)
        sev = (p.severity * 0.3
               + p.safety_impact * 0.2
               + min(10.0, p.complaint_count / 20) * 0.3
               + crit * 2.0 * 0.2)
        raw_severities.append(sev)
        total_complaints += p.complaint_count

    raw_severities.sort(reverse=True)
    top_n = raw_severities[:min(10, len(raw_severities))]
    avg_severity = sum(top_n) / len(top_n)

    volume_boost = min(2.0, (total_complaints / 50) ** 0.5)

    mileage_ratio = 1.0 + min(2.0, mileage / 100_000)
    floor = avg_severity * 4.0 * volume_boost * (mileage_ratio ** 1.3)

    return min(85.0, floor)


def _mileage_baseline(mileage: int) -> float:
    """Direct mileage-based risk floor.

    Even a perfectly reliable model carries inherent wear risk at high mileage.
    Returns 0 at 0 miles, scaling up to ~20 at 150k+.
    """
    if mileage <= 30_000:
        return 0.0
    if mileage <= 60_000:
        return 5.0 * ((mileage - 30_000) / 30_000)
    if mileage <= 100_000:
        return 5.0 + 7.0 * ((mileage - 60_000) / 40_000)
    if mileage <= 150_000:
        return 12.0 + 8.0 * ((mileage - 100_000) / 50_000)
    return 20.0


def _letter_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score <= threshold:
            return grade
    return "F"


_SAFETY_CATEGORIES = {
    "Engine", "Transmission", "Brakes", "Steering",
    "Suspension", "Fuel System", "Electrical", "Cooling",
}

_HIGH_SAFETY_CATEGORIES = {"Brakes", "Steering", "Fuel System", "Suspension"}


def _compute_safety_score(
    scored: list[ScoredProblem],
    num_recalls: int,
) -> float:
    """Compute a safety-focused risk score (0-100, higher = more dangerous).

    Components (each capped, summed to 100 max):
      - Crash/fire impact from NHTSA data           (up to 25)
      - Number of recalls                           (up to 25)
      - Complaint volume on safety-critical systems  (up to 25)
      - Severity of safety-critical system issues    (up to 25)
    """
    if not scored and num_recalls == 0:
        return 0.0

    impact_sum = 0.0
    safety_complaint_count = 0
    severity_sum = 0.0
    safety_system_count = 0

    for sp in scored:
        p = sp.classified.problem
        if p.safety_impact > 0:
            impact_sum += p.safety_impact * min(3.0, p.complaint_count / 20)

        if p.category in _SAFETY_CATEGORIES and p.complaint_count > 0:
            safety_complaint_count += p.complaint_count
            weight = 1.5 if p.category in _HIGH_SAFETY_CATEGORIES else 1.0
            severity_sum += p.severity * weight
            safety_system_count += 1

    crash_component = min(25.0, impact_sum * 2.0)

    recall_component = min(25.0, num_recalls * 6.0)

    volume_component = min(25.0, (safety_complaint_count / 5) ** 0.6 * 3.0)

    if safety_system_count > 0:
        avg_severity = severity_sum / safety_system_count
        severity_component = min(25.0, avg_severity * 2.5)
    else:
        severity_component = 0.0

    raw = crash_component + recall_component + volume_component + severity_component
    return round(min(100.0, raw), 1)


def score_vehicle(
    mileage_analysis: MileageAnalysis,
    make: str = "",
    model: str = "",
    year: int = 0,
    num_recalls: int = 0,
) -> VehicleScore:
    """Score a vehicle's reliability at its analysed mileage point.

    When make/model/year are provided, attempts to load calibrated weights
    and complaint baselines from bulk NHTSA data for more accurate scoring.
    """
    weights = None
    baseline = None
    if make and model and year:
        weights = _get_bulk_weights(make, model, year)
        baseline = _get_bulk_baseline(make, model, year)
        if weights:
            logger.info("Using calibrated weights for %s %s %d", make, model, year)
        if baseline:
            logger.info(
                "Using complaint baseline for %s %s %d: %s",
                make, model, year, baseline.get("interpretation", ""),
            )

    scored: list[ScoredProblem] = []
    for cp in mileage_analysis.classified_problems:
        ws = _score_single(cp, weights=weights, baseline=baseline)
        scored.append(ScoredProblem(classified=cp, weighted_score=ws))

    scored.sort(key=lambda s: s.weighted_score, reverse=True)
    for i, sp in enumerate(scored, start=1):
        sp.rank = i

    top_10 = scored[:10]

    mileage = mileage_analysis.mileage
    mileage_baseline = _mileage_baseline(mileage)

    if scored:
        top_n = min(15, len(scored))
        top_scores = [s.weighted_score for s in scored[:top_n]]
        weighted_avg = sum(top_scores) / len(top_scores)

        source_count = len({
            src
            for sp in scored
            for src in sp.classified.problem.sources
        })
        source_factor = min(1.3, 0.7 + source_count * 0.15)

        mileage_factor = _mileage_wear_factor(mileage)
        risk = weighted_avg * 10 * source_factor * mileage_factor

        inherent_floor = _compute_inherent_risk(scored, mileage)
        risk = max(risk, inherent_floor)

        risk = max(risk, mileage_baseline)
        risk = min(100.0, risk)
    else:
        risk = mileage_baseline

    safety = _compute_safety_score(scored, num_recalls)

    return VehicleScore(
        reliability_risk_score=round(risk, 1),
        safety_score=safety,
        letter_grade=_letter_grade(risk),
        top_issues=top_10,
        total_problems=len(scored),
        mileage_analysis=mileage_analysis,
    )
