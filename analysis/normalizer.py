"""Normalize heterogeneous scraper output into a unified internal schema."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field

from config.settings import VEHICLE_SYSTEMS

logger = logging.getLogger(__name__)

SEVERITY_SCALE = {
    "low": 2,
    "minor": 3,
    "medium": 5,
    "moderate": 5,
    "high": 8,
    "severe": 9,
    "critical": 10,
}

SYSTEM_ALIASES: dict[str, str] = {
    "ac": "HVAC",
    "a/c": "HVAC",
    "air conditioning": "HVAC",
    "heating": "HVAC",
    "heat": "HVAC",
    "body": "Body/Paint",
    "paint": "Body/Paint",
    "body/interior": "Interior",
    "drivetrain": "Transmission",
    "powertrain": "Transmission",
    "gearbox": "Transmission",
    "fuel": "Fuel System",
    "fuel system": "Fuel System",
    "electronics": "Electrical",
    "wiring": "Electrical",
    "lights": "Electrical",
    "wheels": "Suspension",
    "tires": "Suspension",
    "other": "Other",
}

_VALID_SYSTEMS = set(VEHICLE_SYSTEMS) | {"Other"}


@dataclass
class NormalizedProblem:
    category: str
    description: str
    severity: float  # 1-10
    safety_impact: float  # 0-10
    mileage_low: int | None
    mileage_high: int | None
    repair_cost_low: float | None
    repair_cost_high: float | None
    complaint_count: int
    frequency: str
    sources: list[str] = field(default_factory=list)
    user_reports: list[str] = field(default_factory=list)


@dataclass
class NormalizedRecall:
    campaign_number: str
    component: str
    summary: str
    consequence: str
    remedy: str
    report_date: str
    source: str


@dataclass
class NormalizedVehicleData:
    make: str
    model: str
    year: int
    problems: list[NormalizedProblem] = field(default_factory=list)
    recalls: list[NormalizedRecall] = field(default_factory=list)
    ratings: dict = field(default_factory=dict)
    sources_used: list[str] = field(default_factory=list)


def normalize_severity(raw: str | float | int | None) -> float:
    """Convert a source-specific severity label or number to a 1-10 scale."""
    if raw is None:
        return 5.0
    if isinstance(raw, (int, float)):
        return max(1.0, min(10.0, float(raw)))
    return float(SEVERITY_SCALE.get(raw.lower().strip(), 5))


def normalize_category(raw: str) -> str:
    """Map a raw category string to one of the 12 standard vehicle systems."""
    if not raw:
        return "Other"
    if raw in _VALID_SYSTEMS:
        return raw
    lower = raw.lower().strip()
    if lower in SYSTEM_ALIASES:
        return SYSTEM_ALIASES[lower]
    for alias, system in SYSTEM_ALIASES.items():
        if alias in lower:
            return system
    best = difflib.get_close_matches(raw, VEHICLE_SYSTEMS, n=1, cutoff=0.5)
    if best:
        return best[0]
    return "Other"


def _parse_cost_string(cost_str: str | None) -> tuple[float | None, float | None]:
    if not cost_str:
        return None, None
    import re
    nums = re.findall(r"[\d,]+(?:\.\d+)?", cost_str)
    values = [float(n.replace(",", "")) for n in nums]
    if len(values) >= 2:
        return min(values), max(values)
    if len(values) == 1:
        return values[0], values[0]
    return None, None


def normalize_source_data(raw: dict) -> NormalizedVehicleData:
    """Transform a single source's raw JSON into normalized dataclasses."""
    source = raw.get("source", "unknown")
    vehicle = NormalizedVehicleData(
        make=raw.get("make", ""),
        model=raw.get("model", ""),
        year=raw.get("year", 0),
        sources_used=[source],
    )

    for p in raw.get("problems", []):
        mileage_range = p.get("typical_mileage_range")
        mileage_low = mileage_range[0] if mileage_range and len(mileage_range) >= 1 else None
        mileage_high = mileage_range[1] if mileage_range and len(mileage_range) >= 2 else None

        cost_low, cost_high = _parse_cost_string(p.get("estimated_repair_cost"))

        problem = NormalizedProblem(
            category=normalize_category(p.get("category", "")),
            description=p.get("description", ""),
            severity=normalize_severity(p.get("severity")),
            safety_impact=float(p.get("safety_impact") or 0),
            mileage_low=int(mileage_low) if mileage_low is not None else None,
            mileage_high=int(mileage_high) if mileage_high is not None else None,
            repair_cost_low=cost_low,
            repair_cost_high=cost_high,
            complaint_count=int(p.get("complaint_count", 0)),
            frequency=p.get("frequency", "unknown"),
            sources=[source],
            user_reports=list(p.get("user_reports", [])),
        )
        vehicle.problems.append(problem)

    for r in raw.get("recalls", []):
        recall = NormalizedRecall(
            campaign_number=r.get("campaign_number", ""),
            component=r.get("component", ""),
            summary=r.get("summary", ""),
            consequence=r.get("consequence", ""),
            remedy=r.get("remedy", ""),
            report_date=r.get("report_date", ""),
            source=source,
        )
        vehicle.recalls.append(recall)

    vehicle.ratings = raw.get("ratings", {})
    return vehicle


def are_problems_similar(a: NormalizedProblem, b: NormalizedProblem) -> bool:
    """Heuristic check: do two problems describe the same underlying issue?"""
    if a.category != b.category:
        return False
    ratio = difflib.SequenceMatcher(
        None, a.description.lower(), b.description.lower()
    ).ratio()
    return ratio > 0.55
