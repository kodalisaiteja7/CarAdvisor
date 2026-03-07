"""Flask application — web UI and API routes for Car Advisor."""

from __future__ import annotations

import json
import logging
import uuid
from threading import Thread

from flask import Flask, Response, jsonify, render_template, request

from config.settings import FLASK_SECRET_KEY
from database.models import init_db
from scrapers.nhtsa import NHTSAScraper
from scrapers.carcomplaints import CarComplaintsScraper
from scrapers.repairpal import RepairPalScraper
from scrapers.edmunds import EdmundsScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from reports.generator import generate_report

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.secret_key = FLASK_SECRET_KEY

_reports: dict[str, dict] = {}
_progress: dict[str, list[dict]] = {}


SCRAPERS = [
    ("NHTSA", NHTSAScraper),
    ("CarComplaints", CarComplaintsScraper),
    ("RepairPal", RepairPalScraper),
    ("Edmunds", EdmundsScraper),
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
    return render_template("report.html", report=report, report_id=report_id)


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

    report_id = str(uuid.uuid4())[:8]
    _progress[report_id] = []

    thread = Thread(
        target=_run_analysis,
        args=(report_id, make, model, year, mileage),
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
    report_id: str, make: str, model: str, year: int, mileage: int
):
    """Run scraping + analysis in a background thread."""
    results = []

    for name, scraper_cls in SCRAPERS:
        _emit(report_id, name, "scraping", f"Fetching data from {name}...")
        try:
            scraper = scraper_cls()
            data = scraper.fetch(make, model, year)
            results.append(data)
            _emit(report_id, name, "complete", f"{name} data collected")
        except Exception as exc:
            logger.exception("Scraper %s failed", name)
            _emit(report_id, name, "failed", str(exc))

    if not results:
        _emit(report_id, "system", "error", "All scrapers failed")
        return

    _emit(report_id, "analysis", "scraping", "Aggregating and analyzing data...")

    try:
        agg = aggregate(results)
        ma = analyze_mileage(agg, mileage)
        vs = score_vehicle(ma)
        report = generate_report(agg, ma, vs, mileage)
        _reports[report_id] = report
        _emit(report_id, "system", "done", report_id)
    except Exception as exc:
        logger.exception("Analysis failed")
        _emit(report_id, "system", "error", str(exc))


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------


def create_app() -> Flask:
    init_db()
    return app
