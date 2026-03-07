"""Mileage-aware risk analysis.

Classifies every known problem relative to the user's current mileage
so that the same vehicle at different mileages produces meaningfully
different risk profiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from analysis.normalizer import NormalizedProblem
from analysis.aggregator import AggregatedVehicleData
from config.settings import MILEAGE_BRACKETS, VEHICLE_SYSTEMS

logger = logging.getLogger(__name__)


class MileagePhase(str, Enum):
    PAST = "past"
    CURRENT = "current"
    UPCOMING = "upcoming"
    FUTURE = "future"
    UNKNOWN = "unknown"


UPCOMING_WINDOW = 40_000  # miles ahead to consider "upcoming"


@dataclass
class MileageClassifiedProblem:
    problem: NormalizedProblem
    phase: MileagePhase
    distance_to_onset: int | None = None  # negative = already past, 0 = now
    relevance_score: float = 0.0  # 0-1, how relevant at this mileage


@dataclass
class SystemRisk:
    system: str
    risk_score: float  # 0-100
    problem_count: int
    top_problems: list[MileageClassifiedProblem] = field(default_factory=list)


@dataclass
class MileageAnalysis:
    mileage: int
    bracket: str
    phase_counts: dict[str, int] = field(default_factory=dict)
    classified_problems: list[MileageClassifiedProblem] = field(default_factory=list)
    system_risks: list[SystemRisk] = field(default_factory=list)

    @property
    def current_problems(self) -> list[MileageClassifiedProblem]:
        return [p for p in self.classified_problems if p.phase == MileagePhase.CURRENT]

    @property
    def upcoming_problems(self) -> list[MileageClassifiedProblem]:
        return [p for p in self.classified_problems if p.phase == MileagePhase.UPCOMING]

    @property
    def past_problems(self) -> list[MileageClassifiedProblem]:
        return [p for p in self.classified_problems if p.phase == MileagePhase.PAST]

    @property
    def future_problems(self) -> list[MileageClassifiedProblem]:
        return [p for p in self.classified_problems if p.phase == MileagePhase.FUTURE]


def _get_bracket_label(mileage: int) -> str:
    for low, high in MILEAGE_BRACKETS:
        if low <= mileage < high:
            if high == float("inf"):
                return f"{low // 1000}k+"
            return f"{low // 1000}k-{high // 1000}k"
    return "unknown"


def classify_problem(
    problem: NormalizedProblem, user_mileage: int
) -> MileageClassifiedProblem:
    """Determine how a problem relates to the user's current mileage."""
    low = problem.mileage_low
    high = problem.mileage_high

    if low is None or high is None:
        return MileageClassifiedProblem(
            problem=problem,
            phase=MileagePhase.UNKNOWN,
            distance_to_onset=None,
            relevance_score=0.4,
        )

    midpoint = (low + high) / 2

    if user_mileage > high:
        distance = user_mileage - high
        relevance = max(0.1, 1.0 - distance / 50_000)
        return MileageClassifiedProblem(
            problem=problem,
            phase=MileagePhase.PAST,
            distance_to_onset=-(user_mileage - int(midpoint)),
            relevance_score=relevance,
        )

    if low <= user_mileage <= high:
        range_size = max(high - low, 1)
        position = (user_mileage - low) / range_size
        relevance = 0.7 + 0.3 * (1 - abs(position - 0.5) * 2)
        return MileageClassifiedProblem(
            problem=problem,
            phase=MileagePhase.CURRENT,
            distance_to_onset=0,
            relevance_score=relevance,
        )

    distance_to_start = low - user_mileage
    if distance_to_start <= UPCOMING_WINDOW:
        relevance = max(0.3, 1.0 - distance_to_start / UPCOMING_WINDOW)
        return MileageClassifiedProblem(
            problem=problem,
            phase=MileagePhase.UPCOMING,
            distance_to_onset=distance_to_start,
            relevance_score=relevance,
        )

    relevance = max(0.05, 1.0 - distance_to_start / 100_000)
    return MileageClassifiedProblem(
        problem=problem,
        phase=MileagePhase.FUTURE,
        distance_to_onset=distance_to_start,
        relevance_score=relevance,
    )


def _compute_system_risk(
    system: str, classified: list[MileageClassifiedProblem]
) -> SystemRisk:
    system_problems = [c for c in classified if c.problem.category == system]
    if not system_problems:
        return SystemRisk(system=system, risk_score=0, problem_count=0)

    scores = []
    for cp in system_problems:
        base = cp.problem.severity
        phase_mult = {
            MileagePhase.CURRENT: 1.4,
            MileagePhase.UPCOMING: 1.15,
            MileagePhase.PAST: 0.35,
            MileagePhase.FUTURE: 0.25,
            MileagePhase.UNKNOWN: 0.5,
        }
        base *= phase_mult.get(cp.phase, 1.0)
        base *= cp.relevance_score
        count_factor = min(1.3, 1.0 + cp.problem.complaint_count / 500)
        scores.append(base * count_factor)

    scores.sort(reverse=True)
    top_n = scores[:3]
    avg_score = sum(top_n) / len(top_n)
    risk_score = min(100.0, avg_score * 8)

    system_problems.sort(key=lambda c: c.relevance_score, reverse=True)

    return SystemRisk(
        system=system,
        risk_score=round(risk_score, 1),
        problem_count=len(system_problems),
        top_problems=system_problems[:5],
    )


def analyze_mileage(
    data: AggregatedVehicleData, user_mileage: int
) -> MileageAnalysis:
    """Run the full mileage-aware analysis on aggregated vehicle data."""
    classified = [classify_problem(p, user_mileage) for p in data.problems]

    phase_counts = {}
    for phase in MileagePhase:
        phase_counts[phase.value] = sum(
            1 for c in classified if c.phase == phase
        )

    all_systems = list(VEHICLE_SYSTEMS) + ["Other"]
    system_risks = [_compute_system_risk(s, classified) for s in all_systems]
    system_risks = [sr for sr in system_risks if sr.problem_count > 0]
    system_risks.sort(key=lambda sr: sr.risk_score, reverse=True)

    classified.sort(key=lambda c: (
        c.phase != MileagePhase.CURRENT,
        c.phase != MileagePhase.UPCOMING,
        -c.relevance_score,
    ))

    return MileageAnalysis(
        mileage=user_mileage,
        bracket=_get_bracket_label(user_mileage),
        phase_counts=phase_counts,
        classified_problems=classified,
        system_risks=system_risks,
    )
