"""Merge and cross-reference normalized data from multiple scraper sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from analysis.normalizer import (
    NormalizedProblem,
    NormalizedRecall,
    NormalizedVehicleData,
    are_problems_similar,
    normalize_source_data,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregatedVehicleData:
    make: str
    model: str
    year: int
    problems: list[NormalizedProblem] = field(default_factory=list)
    recalls: list[NormalizedRecall] = field(default_factory=list)
    ratings: dict = field(default_factory=dict)
    sources_used: list[str] = field(default_factory=list)
    source_count: int = 0
    complaint_dates: list[str] = field(default_factory=list)

    @property
    def total_complaints(self) -> int:
        return sum(p.complaint_count for p in self.problems)


CROSS_REFERENCE_BOOST = 1.3


def aggregate(source_results: list[dict]) -> AggregatedVehicleData:
    """Merge raw results from all scrapers into a single aggregated dataset.

    When the same problem appears in multiple sources its severity and
    complaint count are boosted to reflect higher confidence.
    """
    if not source_results:
        return AggregatedVehicleData(make="", model="", year=0)

    first = source_results[0]
    agg = AggregatedVehicleData(
        make=first.get("make", ""),
        model=first.get("model", ""),
        year=first.get("year", 0),
    )

    normalized_sets: list[NormalizedVehicleData] = []
    for raw in source_results:
        try:
            nd = normalize_source_data(raw)
            normalized_sets.append(nd)
            agg.sources_used.append(raw.get("source", "unknown"))
        except Exception:
            logger.exception("Failed to normalize data from %s", raw.get("source"))

    agg.source_count = len(agg.sources_used)

    merged_problems: list[NormalizedProblem] = []
    for nd in normalized_sets:
        for problem in nd.problems:
            matched = _find_similar(problem, merged_problems)
            if matched is not None:
                _merge_into(matched, problem)
            else:
                merged_problems.append(problem)

    agg.problems = merged_problems

    seen_campaigns: set[str] = set()
    for nd in normalized_sets:
        for recall in nd.recalls:
            key = recall.campaign_number or recall.summary[:60]
            if key not in seen_campaigns:
                seen_campaigns.add(key)
                agg.recalls.append(recall)

    for nd in normalized_sets:
        source = nd.sources_used[0] if nd.sources_used else "unknown"
        if nd.ratings:
            agg.ratings[source] = nd.ratings

    for raw in source_results:
        agg.complaint_dates.extend(raw.get("complaint_dates", []))

    return agg


def _find_similar(
    problem: NormalizedProblem, existing: list[NormalizedProblem]
) -> NormalizedProblem | None:
    for e in existing:
        if are_problems_similar(e, problem):
            return e
    return None


def _merge_into(target: NormalizedProblem, new: NormalizedProblem):
    """Merge a duplicate problem into an existing one, boosting scores."""
    target.complaint_count += new.complaint_count
    target.severity = min(10, target.severity * CROSS_REFERENCE_BOOST)
    target.safety_impact = max(target.safety_impact, new.safety_impact)

    for s in new.sources:
        if s not in target.sources:
            target.sources.append(s)

    target.user_reports.extend(new.user_reports)

    if new.mileage_low is not None:
        if target.mileage_low is None:
            target.mileage_low = new.mileage_low
        else:
            target.mileage_low = min(target.mileage_low, new.mileage_low)

    if new.mileage_high is not None:
        if target.mileage_high is None:
            target.mileage_high = new.mileage_high
        else:
            target.mileage_high = max(target.mileage_high, new.mileage_high)

    if new.repair_cost_low is not None:
        if target.repair_cost_low is None:
            target.repair_cost_low = new.repair_cost_low
        else:
            target.repair_cost_low = min(target.repair_cost_low, new.repair_cost_low)

    if new.repair_cost_high is not None:
        if target.repair_cost_high is None:
            target.repair_cost_high = new.repair_cost_high
        else:
            target.repair_cost_high = max(target.repair_cost_high, new.repair_cost_high)

    if not target.description or len(new.description) > len(target.description):
        target.description = new.description
