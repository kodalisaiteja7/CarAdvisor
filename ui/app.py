"""Flask application — web UI and API routes for Car Advisor."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from threading import Thread

from flask import Flask, Response, jsonify, render_template, request

from config.settings import FLASK_SECRET_KEY
from database.models import init_db
from scrapers.nhtsa import NHTSAScraper
from scrapers.carcomplaints import CarComplaintsScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from reports.generator import generate_report
from utils.trace import start_trace, end_trace

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.secret_key = FLASK_SECRET_KEY

_reports: dict[str, dict] = {}
_progress: dict[str, list[dict]] = {}
_traces: dict[str, dict] = {}


SCRAPERS = [
    ("NHTSA", NHTSAScraper),
    ("CarComplaints", CarComplaintsScraper),
]


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report/<report_id>")
def report_page(report_id: str):
    report = _reports.get(report_id)
    if not report:
        return render_template("index.html", error="Report not found"), 404
    _postprocess_current_risk(report)
    return render_template("report.html", report=report, report_id=report_id)


def _postprocess_current_risk(report: dict):
    """Merge same-system issues, filter out 'Other', sort by complaint count."""
    cr = report.get("sections", {}).get("current_risk", {})
    if not cr:
        return

    top_issues = cr.get("top_issues", [])
    merged: dict[str, dict] = {}
    for issue in top_issues:
        sys = issue.get("system", "")
        if sys == "Other":
            continue
        if sys not in merged:
            merged[sys] = {
                "system": sys,
                "description": issue.get("description", ""),
                "probability": issue.get("probability", "Low"),
                "severity": issue.get("severity", 0),
                "phase": issue.get("phase", "unknown"),
                "complaint_count": issue.get("complaint_count", 0),
                "test_drive_tips": list(issue.get("test_drive_tips") or []),
                "diagnostic_tests": list(issue.get("diagnostic_tests") or []),
                "sources": list(issue.get("sources") or []),
                "test_drive_narrative": issue.get("test_drive_narrative"),
                "what_to_listen_for": issue.get("what_to_listen_for"),
                "llm_enhanced": issue.get("llm_enhanced", False),
            }
        else:
            m = merged[sys]
            m["complaint_count"] += issue.get("complaint_count", 0)
            new_sev = issue.get("severity", 0)
            if new_sev > m["severity"]:
                m["severity"] = new_sev
            if new_sev >= 7:
                m["probability"] = "High"
            elif new_sev >= 4 and m["probability"] != "High":
                m["probability"] = "Medium"
            desc = issue.get("description", "")
            if desc and desc not in m["description"]:
                m["description"] += " | " + desc
            phase_priority = {"current": 0, "upcoming": 1, "past": 2, "future": 3, "unknown": 4}
            if phase_priority.get(issue.get("phase"), 5) < phase_priority.get(m["phase"], 5):
                m["phase"] = issue["phase"]
            for tip in issue.get("test_drive_tips") or []:
                if tip not in m["test_drive_tips"]:
                    m["test_drive_tips"].append(tip)
            for test in issue.get("diagnostic_tests") or []:
                if test not in m["diagnostic_tests"]:
                    m["diagnostic_tests"].append(test)
            for src in issue.get("sources") or []:
                if src not in m["sources"]:
                    m["sources"].append(src)
            if issue.get("test_drive_narrative") and not m["test_drive_narrative"]:
                m["test_drive_narrative"] = issue["test_drive_narrative"]
            if issue.get("what_to_listen_for") and not m["what_to_listen_for"]:
                m["what_to_listen_for"] = issue["what_to_listen_for"]
            if issue.get("llm_enhanced"):
                m["llm_enhanced"] = True

    issues = sorted(merged.values(), key=lambda x: x["complaint_count"], reverse=True)
    for i, issue in enumerate(issues, start=1):
        issue["rank"] = i
    cr["top_issues"] = issues

    cr["system_risks"] = [
        sr for sr in cr.get("system_risks", []) if sr.get("system") != "Other"
    ]


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------


@app.route("/api/years")
def api_years():
    scraper = NHTSAScraper()
    years = scraper.get_years()
    return jsonify(years)


@app.route("/api/makes")
def api_makes():
    year = request.args.get("year", type=int)
    scraper = NHTSAScraper()
    makes = scraper.get_makes(year)
    return jsonify(makes)


@app.route("/api/models")
def api_models():
    make = request.args.get("make", "")
    year = request.args.get("year", type=int)
    if not make or not year:
        return jsonify([])
    scraper = NHTSAScraper()
    models = scraper.get_models(make, year)
    return jsonify(models)


@app.route("/api/vehicle-trims")
def api_vehicle_trims():
    """Return trim / variant names from fueleconomy.gov for a year/make/model."""
    year = request.args.get("year", type=int)
    make = request.args.get("make", "")
    model_name = request.args.get("model", "")
    if not all([year, make, model_name]):
        return jsonify([])

    try:
        trims = _fetch_trims(year, make, model_name)
    except Exception:
        logger.warning("fueleconomy.gov trim lookup failed", exc_info=True)
        trims = []
    return jsonify(trims)


@app.route("/api/vehicle-engines")
def api_vehicle_engines():
    """Return engine options from fueleconomy.gov for a specific trim variant."""
    year = request.args.get("year", type=int)
    make = request.args.get("make", "")
    trim_variant = request.args.get("trim", "")
    if not all([year, make, trim_variant]):
        return jsonify([])

    try:
        engines = _fetch_engines(year, make, trim_variant)
    except Exception:
        logger.warning("fueleconomy.gov engine lookup failed", exc_info=True)
        engines = []
    return jsonify(engines)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json() or {}
    make = data.get("make", "").strip()
    model = data.get("model", "").strip()
    year = data.get("year")
    mileage = data.get("mileage")

    if not all([make, model, year, mileage]):
        return jsonify({"error": "make, model, year, and mileage are required"}), 400

    try:
        year = int(year)
        mileage = int(mileage)
    except (ValueError, TypeError):
        return jsonify({"error": "year and mileage must be integers"}), 400

    options = {
        "trim": (data.get("trim") or "").strip() or None,
        "engine": (data.get("engine") or "").strip() or None,
        "transmission": (data.get("transmission") or "").strip() or None,
        "drivetrain": (data.get("drivetrain") or "").strip() or None,
    }

    report_id = str(uuid.uuid4())[:8]
    _progress[report_id] = []

    thread = Thread(
        target=_run_analysis,
        args=(report_id, make, model, year, mileage, options),
        daemon=True,
    )
    thread.start()

    return jsonify({"report_id": report_id})


@app.route("/api/progress/<report_id>")
def api_progress(report_id: str):
    """Server-Sent Events stream for analysis progress."""

    def stream():
        last_idx = 0
        while True:
            events = _progress.get(report_id, [])
            for evt in events[last_idx:]:
                yield f"data: {json.dumps(evt)}\n\n"
                last_idx = len(events)
                if evt.get("status") in ("done", "error"):
                    return
            import time
            time.sleep(0.3)

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/report/<report_id>")
def api_report(report_id: str):
    report = _reports.get(report_id)
    if not report:
        return jsonify({"error": "Report not found or still processing"}), 404
    return jsonify(report)


@app.route("/api/trace/<report_id>")
def api_trace(report_id: str):
    """Return the debug trace for a report (JSON)."""
    trace_data = _traces.get(report_id)
    if trace_data:
        return jsonify(trace_data)

    trace_file = Path(__file__).resolve().parent.parent / "logs" / f"{report_id}.json"
    if trace_file.exists():
        return Response(
            trace_file.read_text(encoding="utf-8"),
            mimetype="application/json",
        )

    return jsonify({"error": "Trace not found"}), 404


@app.route("/trace/<report_id>")
def trace_page(report_id: str):
    """Render the debug trace viewer."""
    return render_template("trace.html", report_id=report_id)


# ------------------------------------------------------------------
# Background analysis
# ------------------------------------------------------------------


def _emit(report_id: str, source: str, status: str, message: str = ""):
    _progress.setdefault(report_id, []).append({
        "source": source,
        "status": status,
        "message": message,
    })


def _run_analysis(
    report_id: str, make: str, model: str, year: int, mileage: int,
    options: dict | None = None,
):
    """Run scraping + analysis in a background thread."""
    options = options or {}

    trace = start_trace(report_id)
    trace.log_user_query(
        make=make, model=model, year=year, mileage=mileage, options=options,
    )

    results = []

    for name, scraper_cls in SCRAPERS:
        _emit(report_id, name, "scraping", f"Fetching data from {name}...")
        try:
            scraper = scraper_cls()
            data = scraper.fetch(make, model, year)
            results.append(data)
            trace.log_scraper(name, "success", data=data)
            _emit(report_id, name, "complete", f"{name} data collected")
        except Exception as exc:
            logger.exception("Scraper %s failed", name)
            trace.log_scraper(name, "failed", error=exc)
            _emit(report_id, name, "failed", str(exc))

    if not results:
        _emit(report_id, "system", "error", "All scrapers failed")
        end_trace()
        return

    _emit(report_id, "Analysis", "scraping", "Aggregating and analyzing data...")

    try:
        agg = aggregate(results)
        ma = analyze_mileage(agg, mileage)
        vs = score_vehicle(ma, make=make, model=model, year=year)

        trace.log_analysis(
            total_complaints=agg.total_complaints,
            total_problems=len(agg.problems),
            total_recalls=len(agg.recalls),
            sources_used=agg.sources_used,
            mileage_bracket=ma.bracket,
            phase_counts=ma.phase_counts,
            reliability_risk_score=vs.reliability_risk_score,
            letter_grade=vs.letter_grade,
            top_issues_count=len(vs.top_issues),
        )

        _emit(report_id, "Analysis", "complete", "Data analysis complete")

        bulk_stats = None
        vector_complaints = None
        try:
            _emit(report_id, "Bulk Data", "scraping", "Looking up NHTSA bulk statistics...")
            from data.stats_builder import get_model_stats
            from data.vector_search import search_similar_complaints
            bulk_stats = get_model_stats(make, model, year)
            if bulk_stats:
                logger.info(
                    "Bulk stats found: %d complaints, %sth percentile",
                    bulk_stats.get("total_complaints", 0),
                    bulk_stats.get("complaints_percentile", "?"),
                )
            vector_complaints = search_similar_complaints(
                make, model, year, mileage=mileage,
            )
            if vector_complaints:
                logger.info("Retrieved %d similar complaints from vector store", len(vector_complaints))
            _emit(report_id, "Bulk Data", "complete", "Bulk data retrieved")
        except Exception as exc:
            logger.info("Bulk data not available (this is OK if not set up): %s", exc)
            _emit(report_id, "Bulk Data", "complete", "Bulk data not available (using scraper data only)")

        _emit(report_id, "AI Insights", "scraping", "Generating AI-powered insights and guidance...")
        report = generate_report(
            agg, ma, vs, mileage, options=options,
            vector_complaints=vector_complaints,
            bulk_stats=bulk_stats,
        )
        _emit(report_id, "AI Insights", "complete", "AI insights generated")

        _reports[report_id] = report
        finished_trace = end_trace()
        if finished_trace:
            _traces[report_id] = finished_trace.to_dict()
        _emit(report_id, "system", "done", report_id)
    except Exception as exc:
        logger.exception("Analysis failed")
        end_trace()
        _emit(report_id, "system", "error", str(exc))


# ------------------------------------------------------------------
# Fueleconomy.gov helpers
# ------------------------------------------------------------------

_FUEL_ECO_BASE = "https://www.fueleconomy.gov/ws/rest/vehicle/menu"
_FUEL_ECO_HEADERS = {"Accept": "application/json"}


def _fuel_eco_get(path: str) -> list[dict]:
    """GET a fueleconomy.gov menu endpoint and return the items list."""
    import requests as http_client
    from urllib.parse import quote

    resp = http_client.get(
        f"{_FUEL_ECO_BASE}/{path}",
        headers=_FUEL_ECO_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return []
    items = data.get("menuItem", [])
    if isinstance(items, dict):
        items = [items]
    return items


def _fetch_trims(year: int, make: str, model_name: str) -> list[dict]:
    """Return trim/variant labels derived from fueleconomy.gov model names."""
    from urllib.parse import quote

    items = _fuel_eco_get(f"model?year={year}&make={quote(make)}")
    model_lower = model_name.lower()
    matching = [m["text"] for m in items if model_lower in m.get("text", "").lower()]

    if not matching:
        return []

    trims = []
    for variant in sorted(matching):
        idx = variant.lower().find(model_lower)
        if idx != -1:
            suffix = variant[idx + len(model_lower) :].strip()
            label = suffix if suffix else "(Base)"
        else:
            label = variant
        trims.append({"label": label, "value": variant})

    if len(trims) == 1 and trims[0]["label"] == "(Base)":
        return []

    return trims


def _parse_engine(option_text: str) -> str:
    """Parse engine description from fueleconomy.gov option text.

    Input:  'Auto (AV-S7), 4 cyl, 1.5 L, Turbo'
    Output: '1.5L 4-cyl Turbo'
    """
    parts = [p.strip() for p in option_text.split(",")]
    if len(parts) < 3:
        return option_text

    cyl = parts[1].strip().replace(" cyl", "-cyl")
    disp = parts[2].strip().replace(" L", "L").replace(" l", "L")
    engine = f"{disp} {cyl}"

    extras = [p.strip() for p in parts[3:]]
    if extras:
        engine += f" {' '.join(extras)}"
    return engine


def _fetch_engines(year: int, make: str, trim_variant: str) -> list[str]:
    """Return unique engine descriptions for a specific fueleconomy.gov model variant."""
    from urllib.parse import quote

    items = _fuel_eco_get(
        f"options?year={year}&make={quote(make)}&model={quote(trim_variant)}"
    )
    engines = set()
    for opt in items:
        engine = _parse_engine(opt.get("text", ""))
        if engine:
            engines.add(engine)
    return sorted(engines)


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------


def create_app() -> Flask:
    init_db()
    return app
