"""Report generator — builds the full structured buying report.

Produces a dict with all 7 report sections ready for rendering by the
web UI or CLI.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from analysis.aggregator import AggregatedVehicleData
from analysis.llm_enhancer import enhance_inspection_checklist, enhance_report_sections
from analysis.mileage_model import MileageAnalysis, MileagePhase
from analysis.scorer import VehicleScore, ScoredProblem

logger = logging.getLogger(__name__)

FORECAST_WINDOWS = [20_000, 40_000, 60_000]


def generate_report(
    agg: AggregatedVehicleData,
    mileage_analysis: MileageAnalysis,
    vehicle_score: VehicleScore,
    user_mileage: int,
) -> dict:
    """Build the complete report dict with all 7 sections."""
    vehicle = {
        "make": agg.make,
        "model": agg.model,
        "year": agg.year,
        "mileage": user_mileage,
    }

    checklist = _build_inspection_checklist(
        mileage_analysis, vehicle_score, user_mileage
    )
    checklist = enhance_inspection_checklist(vehicle, checklist)

    report = {
        "vehicle": vehicle,
        "meta": {
            "sources_used": agg.sources_used,
            "source_count": agg.source_count,
            "total_problems_found": len(agg.problems),
            "total_recalls_found": len(agg.recalls),
            "mileage_bracket": mileage_analysis.bracket,
        },
        "sections": {
            "vehicle_summary": _build_vehicle_summary(agg, vehicle_score),
            "inspection_checklist": checklist,
            "current_risk": _build_current_risk(mileage_analysis, vehicle_score),
            "future_forecast": _build_future_forecast(
                mileage_analysis, agg, user_mileage
            ),
            "owner_experience": _build_owner_experience(agg),
            "red_flags": _build_red_flags(agg, vehicle_score, mileage_analysis),
            "negotiation": _build_negotiation(
                mileage_analysis, vehicle_score, user_mileage
            ),
        },
    }

    report = enhance_report_sections(vehicle, report)

    return report


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------


def _build_vehicle_summary(
    agg: AggregatedVehicleData, score: VehicleScore
) -> dict:
    ratings_summary = {}
    for source, rating_data in agg.ratings.items():
        if isinstance(rating_data, dict):
            overall = rating_data.get("overall") or rating_data.get("overall_text")
            if overall:
                ratings_summary[source] = overall

    return {
        "title": f"{agg.year} {agg.make} {agg.model}",
        "reliability_risk_score": score.reliability_risk_score,
        "letter_grade": score.letter_grade,
        "total_complaints": agg.total_complaints,
        "total_recalls": len(agg.recalls),
        "ratings_by_source": ratings_summary,
        "recalls": [
            {
                "campaign": r.campaign_number,
                "component": r.component,
                "summary": r.summary,
                "consequence": r.consequence,
                "remedy": r.remedy,
                "date": r.report_date,
                "source": r.source,
            }
            for r in agg.recalls
        ],
        "sources": agg.sources_used,
    }


def _build_inspection_checklist(
    ma: MileageAnalysis, score: VehicleScore, user_mileage: int
) -> dict:
    must_check = []
    recommended = []
    standard = []

    for cp in ma.classified_problems:
        item = {
            "system": cp.problem.category,
            "description": cp.problem.description,
            "what_to_look_for": _inspection_guidance(cp),
            "estimated_cost_if_bad": _format_cost_range(
                cp.problem.repair_cost_low, cp.problem.repair_cost_high
            ),
            "sources": cp.problem.sources,
        }

        if cp.phase == MileagePhase.CURRENT and cp.problem.severity >= 6:
            must_check.append(item)
        elif cp.phase in (MileagePhase.CURRENT, MileagePhase.UPCOMING):
            recommended.append(item)
        elif cp.phase == MileagePhase.PAST and cp.problem.severity >= 7:
            recommended.append(item)
        else:
            standard.append(item)

    standard.extend(_standard_checks())

    return {
        "must_check": must_check[:10],
        "recommended": recommended[:10],
        "standard": standard[:10],
        "total_items": len(must_check) + len(recommended) + len(standard),
    }


def _build_current_risk(ma: MileageAnalysis, score: VehicleScore) -> dict:
    issues = []
    for sp in score.top_issues:
        cp = sp.classified
        issues.append({
            "rank": sp.rank,
            "system": cp.problem.category,
            "description": cp.problem.description,
            "probability": sp.probability,
            "severity": round(cp.problem.severity, 1),
            "phase": cp.phase.value,
            "complaint_count": cp.problem.complaint_count,
            "test_drive_tips": _test_drive_tips(cp),
            "diagnostic_tests": _diagnostic_suggestions(cp),
            "sources": cp.problem.sources,
        })

    system_risks = [
        {
            "system": sr.system,
            "risk_score": sr.risk_score,
            "problem_count": sr.problem_count,
        }
        for sr in ma.system_risks
    ]

    return {
        "mileage": ma.mileage,
        "bracket": ma.bracket,
        "top_issues": issues,
        "system_risks": system_risks,
        "phase_summary": ma.phase_counts,
    }


def _build_future_forecast(
    ma: MileageAnalysis, agg: AggregatedVehicleData, user_mileage: int
) -> dict:
    windows = []
    for window in FORECAST_WINDOWS:
        target = user_mileage + window
        predicted = []
        total_cost_low = 0.0
        total_cost_high = 0.0

        for cp in ma.classified_problems:
            p = cp.problem
            if p.mileage_low is None or p.mileage_high is None:
                continue
            if p.mileage_low <= target and p.mileage_low > user_mileage:
                cost_low = p.repair_cost_low or 0
                cost_high = p.repair_cost_high or 0
                total_cost_low += cost_low
                total_cost_high += cost_high
                predicted.append({
                    "system": p.category,
                    "description": p.description,
                    "expected_mileage": f"{p.mileage_low:,}-{p.mileage_high:,}",
                    "estimated_cost": _format_cost_range(
                        p.repair_cost_low, p.repair_cost_high
                    ),
                    "complaint_count": p.complaint_count,
                    "sources": p.sources,
                })

        windows.append({
            "window_miles": window,
            "window_label": f"Next {window // 1000}k miles",
            "target_mileage": target,
            "predicted_issues": predicted[:10],
            "estimated_total_cost": _format_cost_range(
                total_cost_low if total_cost_low else None,
                total_cost_high if total_cost_high else None,
            ),
        })

    return {"forecast_windows": windows}


def _build_owner_experience(agg: AggregatedVehicleData) -> dict:
    all_reports = []
    for p in agg.problems:
        all_reports.extend(p.user_reports[:3])

    return {
        "sample_reports": all_reports[:15],
        "report_count": len(all_reports),
        "note": (
            "Owner experience data is currently sourced from NHTSA complaints "
            "and CarComplaints.com. Reddit and forum data will be added in a "
            "future update for richer sentiment analysis."
        ),
    }


def _build_red_flags(
    agg: AggregatedVehicleData,
    score: VehicleScore,
    ma: MileageAnalysis,
) -> dict:
    catastrophic = []
    for sp in score.top_issues:
        p = sp.classified.problem
        if p.severity >= 8 or p.safety_impact >= 7:
            catastrophic.append({
                "system": p.category,
                "description": p.description,
                "severity": round(p.severity, 1),
                "safety_impact": round(p.safety_impact, 1),
                "complaint_count": p.complaint_count,
                "sources": p.sources,
            })

    open_recalls = [
        {
            "campaign": r.campaign_number,
            "component": r.component,
            "summary": r.summary,
            "consequence": r.consequence,
        }
        for r in agg.recalls
    ]

    return {
        "catastrophic_failures": catastrophic,
        "open_recalls": open_recalls,
        "vin_check_recommended": len(agg.recalls) > 0,
        "recommendations": [
            "Run a VIN check to verify all recalls have been completed",
            "Request a pre-purchase inspection from an independent mechanic",
            "Check for signs of flood damage (musty smell, water lines, corroded electronics)",
            "Verify the odometer reading matches service history",
        ],
    }


def _build_negotiation(
    ma: MileageAnalysis, score: VehicleScore, user_mileage: int
) -> dict:
    talking_points = []
    total_upcoming_cost_low = 0.0
    total_upcoming_cost_high = 0.0

    for cp in ma.upcoming_problems + ma.current_problems:
        p = cp.problem
        cost_low = p.repair_cost_low or 0
        cost_high = p.repair_cost_high or 0
        total_upcoming_cost_low += cost_low
        total_upcoming_cost_high += cost_high

        if cost_low > 0 or p.severity >= 6:
            talking_points.append({
                "issue": p.description,
                "system": p.category,
                "estimated_cost": _format_cost_range(
                    p.repair_cost_low, p.repair_cost_high
                ),
                "complaint_count": p.complaint_count,
                "leverage": _negotiation_leverage(cp),
            })

    talking_points.sort(
        key=lambda t: float(
            (t.get("estimated_cost") or "$0")
            .replace("$", "")
            .replace(",", "")
            .split("-")[0]
            or "0"
        ),
        reverse=True,
    )

    return {
        "talking_points": talking_points[:8],
        "total_upcoming_maintenance": _format_cost_range(
            total_upcoming_cost_low if total_upcoming_cost_low else None,
            total_upcoming_cost_high if total_upcoming_cost_high else None,
        ),
        "summary": (
            f"Based on {score.total_problems} known issues and the current "
            f"mileage of {user_mileage:,} miles, this vehicle may need "
            f"{_format_cost_range(total_upcoming_cost_low, total_upcoming_cost_high)} "
            f"in maintenance over the next 30k miles."
        ),
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _format_cost_range(low: float | None, high: float | None) -> str | None:
    if low is None and high is None:
        return None
    if low == high or high is None:
        return f"${low:,.0f}" if low else None
    if low is None:
        return f"${high:,.0f}"
    return f"${low:,.0f}-${high:,.0f}"


def _inspection_guidance(cp) -> str:
    p = cp.problem
    cat = p.category.lower()
    guides = {
        "engine": "Check for unusual noises, smoke from exhaust, oil leaks. Request a compression test.",
        "transmission": "Test all gears during the drive. Feel for slipping, hard shifts, or delays.",
        "electrical": "Test all electronics — windows, locks, lights, infotainment. Check for warning lights.",
        "suspension": "Listen for clunks over bumps. Check tire wear for uneven patterns.",
        "brakes": "Test braking feel — pulsation, pulling, or grinding indicates problems.",
        "steering": "Check for play in the steering wheel. Listen for whine during turns.",
        "cooling": "Check coolant level and color. Look for leaks under the car.",
        "hvac": "Test both heat and AC at all settings. Check for unusual smells.",
        "fuel system": "Check for fuel smell. Monitor fuel gauge behavior during drive.",
        "exhaust": "Listen for rattles or unusual exhaust sound. Check for visible rust.",
    }
    for key, guide in guides.items():
        if key in cat:
            return guide
    return "Visually inspect for damage, wear, or signs of previous repair."


def _test_drive_tips(cp) -> list[str]:
    tips = []
    cat = cp.problem.category.lower()
    if "engine" in cat:
        tips.extend([
            "Cold-start the engine and listen for knocking or ticking",
            "Watch for blue/white smoke from the exhaust",
            "Monitor the temperature gauge during the drive",
        ])
    elif "transmission" in cat:
        tips.extend([
            "Drive through all gears and feel for hesitation",
            "Test reverse — listen for clunks",
            "Try stop-and-go driving to check for jerky shifts",
        ])
    elif "electrical" in cat:
        tips.extend([
            "Test every button, switch, and screen",
            "Check if any warning lights stay on",
            "Test the battery voltage if possible",
        ])
    elif "brakes" in cat:
        tips.extend([
            "Brake firmly from 40 mph — feel for pulsation",
            "Check if the car pulls to one side under braking",
        ])
    elif "suspension" in cat:
        tips.extend([
            "Drive over speed bumps and listen for rattles",
            "Check if the car tracks straight on the highway",
        ])
    else:
        tips.append(f"Pay attention to the {cp.problem.category} system during the test drive")
    return tips


def _diagnostic_suggestions(cp) -> list[str]:
    suggestions = []
    cat = cp.problem.category.lower()
    if "engine" in cat:
        suggestions.extend([
            "Request a compression test",
            "Check for OBD-II codes (especially P0300-P0306 misfires, P0420 catalytic)",
            "Oil analysis if available",
        ])
    elif "transmission" in cat:
        suggestions.extend([
            "Check transmission fluid color and smell",
            "Scan for transmission-related DTCs",
            "Check for transmission adaption resets in scan data",
        ])
    elif "electrical" in cat:
        suggestions.append("Full OBD-II scan for stored and pending codes")
    else:
        suggestions.append("Full OBD-II diagnostic scan recommended")
    return suggestions


def _negotiation_leverage(cp) -> str:
    p = cp.problem
    if p.complaint_count > 100:
        return f"Well-documented issue with {p.complaint_count}+ complaints filed"
    if p.severity >= 8:
        return "Severe known issue — significant repair cost likely"
    if p.repair_cost_high and p.repair_cost_high > 1000:
        return f"Potential repair cost up to ${p.repair_cost_high:,.0f}"
    return "Known issue to factor into pricing"


def _standard_checks() -> list[dict]:
    """Baseline checks applicable to any used vehicle."""
    return [
        {
            "system": "General",
            "description": "Verify title status (clean, salvage, rebuilt)",
            "what_to_look_for": "Request title and check for any branding",
            "estimated_cost_if_bad": None,
            "sources": ["standard"],
        },
        {
            "system": "General",
            "description": "Check service history documentation",
            "what_to_look_for": "Look for consistent maintenance at recommended intervals",
            "estimated_cost_if_bad": None,
            "sources": ["standard"],
        },
        {
            "system": "Body/Paint",
            "description": "Check for paint inconsistencies (accident repair)",
            "what_to_look_for": "Use a paint depth gauge or look for overspray, mismatched panels",
            "estimated_cost_if_bad": "$500-$5,000+",
            "sources": ["standard"],
        },
        {
            "system": "Brakes",
            "description": "Check brake pad and rotor condition",
            "what_to_look_for": "Measure pad thickness (replace below 3mm). Check rotors for scoring.",
            "estimated_cost_if_bad": "$300-$800",
            "sources": ["standard"],
        },
        {
            "system": "Suspension",
            "description": "Check tire condition and tread depth",
            "what_to_look_for": "Look for uneven wear patterns (alignment/suspension issues)",
            "estimated_cost_if_bad": "$400-$1,200 for a full set",
            "sources": ["standard"],
        },
    ]
