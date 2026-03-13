"""Report generator — builds the structured buying report.

Produces a dict with report sections ready for rendering by the web UI or CLI.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from analysis.aggregator import AggregatedVehicleData
from analysis.llm_enhancer import enhance_inspection_checklist, enhance_report_sections
from analysis.mileage_model import MileageAnalysis, MileagePhase
from analysis.scorer import VehicleScore
from utils.trace import get_trace, _thread_local

logger = logging.getLogger(__name__)

def generate_report(
    agg: AggregatedVehicleData,
    mileage_analysis: MileageAnalysis,
    vehicle_score: VehicleScore,
    user_mileage: int,
    options: dict | None = None,
    vector_complaints: list[dict] | None = None,
    bulk_stats: dict | None = None,
    price_data: dict | None = None,
) -> dict:
    """Build the complete report dict."""
    options = options or {}
    vehicle = {
        "make": agg.make,
        "model": agg.model,
        "year": agg.year,
        "mileage": user_mileage,
        "trim": options.get("trim"),
        "engine": options.get("engine"),
        "transmission": options.get("transmission"),
        "drivetrain": options.get("drivetrain"),
    }

    raw_checklist = _build_inspection_checklist(
        mileage_analysis, vehicle_score, user_mileage
    )
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
            "inspection_checklist": raw_checklist,
            "current_risk": _build_current_risk(mileage_analysis, vehicle_score, agg),
            "owner_experience": _build_owner_experience(agg),
            "pricing": _build_pricing_section(options.get("asking_price"), price_data),
        },
    }

    trace = get_trace()
    if trace:
        trace.log_sections("pre_llm", report["sections"])

    def _run_with_trace(fn, *args, **kwargs):
        if trace:
            _thread_local.trace = trace
        return fn(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        checklist_future = pool.submit(
            _run_with_trace, enhance_inspection_checklist, vehicle, raw_checklist,
        )
        verdict_future = pool.submit(
            _run_with_trace, enhance_report_sections, vehicle, report,
            vector_complaints=vector_complaints, bulk_stats=bulk_stats,
            price_data=price_data,
        )

        try:
            report["sections"]["inspection_checklist"] = checklist_future.result(timeout=120)
        except Exception:
            logger.warning("Checklist LLM failed — using raw checklist")
            report["sections"]["inspection_checklist"] = raw_checklist

        try:
            report = verdict_future.result(timeout=120)
        except Exception:
            logger.warning("Verdict LLM failed — report will lack executive summary")

    if trace:
        trace.log_sections("post_llm", report["sections"])

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
        "safety_score": score.safety_score,
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


def _build_current_risk(ma: MileageAnalysis, score: VehicleScore, agg: AggregatedVehicleData | None = None) -> dict:
    merged: dict[str, dict] = {}
    for sp in score.top_issues:
        cp = sp.classified
        sys = cp.problem.category
        if sys == "Other":
            continue
        if sys not in merged:
            merged[sys] = {
                "system": sys,
                "descriptions": [cp.problem.description],
                "probability": sp.probability,
                "severity": round(cp.problem.severity, 1),
                "weighted_score": sp.weighted_score,
                "phase": cp.phase.value,
                "complaint_count": cp.problem.complaint_count,
                "test_drive_tips": list(_test_drive_tips(cp)),
                "diagnostic_tests": list(_diagnostic_suggestions(cp)),
                "sources": list(cp.problem.sources),
            }
        else:
            m = merged[sys]
            m["complaint_count"] += cp.problem.complaint_count
            if cp.problem.description not in m["descriptions"]:
                m["descriptions"].append(cp.problem.description)
            m["severity"] = max(m["severity"], round(cp.problem.severity, 1))
            m["weighted_score"] = max(m["weighted_score"], sp.weighted_score)
            if m["weighted_score"] >= 6:
                m["probability"] = "High"
            elif m["weighted_score"] >= 3:
                m["probability"] = "Medium"
            else:
                m["probability"] = "Low"
            phase_priority = {"current": 0, "upcoming": 1, "past": 2, "future": 3, "unknown": 4}
            if phase_priority.get(cp.phase.value, 5) < phase_priority.get(m["phase"], 5):
                m["phase"] = cp.phase.value
            for tip in _test_drive_tips(cp):
                if tip not in m["test_drive_tips"]:
                    m["test_drive_tips"].append(tip)
            for test in _diagnostic_suggestions(cp):
                if test not in m["diagnostic_tests"]:
                    m["diagnostic_tests"].append(test)
            for src in cp.problem.sources:
                if src not in m["sources"]:
                    m["sources"].append(src)

    issues = sorted(merged.values(), key=lambda x: x["complaint_count"], reverse=True)
    for i, issue in enumerate(issues, start=1):
        issue["rank"] = i
        issue["description"] = " | ".join(issue.pop("descriptions"))
        issue.pop("weighted_score", None)

    system_risks = [
        {
            "system": sr.system,
            "risk_score": sr.risk_score,
            "problem_count": sr.problem_count,
            "total_complaints": sr.total_complaints,
        }
        for sr in ma.system_risks
        if sr.system != "Other"
    ]

    complaints_by_year: dict[int, int] = {}
    if agg:
        for date_str in agg.complaint_dates:
            try:
                yr = int(date_str.split("/")[-1][:4]) if "/" in date_str else int(date_str[:4])
                if 1990 <= yr <= 2030:
                    complaints_by_year[yr] = complaints_by_year.get(yr, 0) + 1
            except (ValueError, IndexError):
                continue

    return {
        "mileage": ma.mileage,
        "bracket": ma.bracket,
        "top_issues": issues,
        "system_risks": system_risks,
        "phase_summary": ma.phase_counts,
        "complaints_by_year": dict(sorted(complaints_by_year.items())),
    }


def _build_pricing_section(asking_price: int | None, price_data: dict | None) -> dict:
    """Build the pricing comparison section."""
    if not price_data:
        return {"available": False}

    avg_price = price_data.get("avg_price", 0)
    price_diff = None
    price_verdict = None

    if asking_price and avg_price:
        price_diff = asking_price - avg_price
        pct = abs(price_diff) / avg_price if avg_price else 0
        if pct <= 0.05:
            price_verdict = "at_market"
        elif price_diff < 0:
            price_verdict = "below_market"
        else:
            price_verdict = "above_market"

    return {
        "available": True,
        "asking_price": asking_price,
        "avg_market_price": avg_price,
        "source": price_data.get("source", ""),
        "listings_count": price_data.get("listings_count", 0),
        "price_range": price_data.get("price_range"),
        "match_level": price_data.get("match_level", "estimate"),
        "price_difference": price_diff,
        "price_verdict": price_verdict,
    }


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
