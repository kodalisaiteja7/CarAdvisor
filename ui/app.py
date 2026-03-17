"""Flask application — web UI and API routes for Car Advisor."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from threading import Thread

from flask import Flask, Response, jsonify, render_template, request

from config.settings import FLASK_SECRET_KEY
from database.models import init_db
from cache.store import (
    init_store, get_report, set_report, get_progress, push_progress,
    init_progress, get_trace, set_trace, get_cached_report_id,
    set_cached_report_id,
)
from scrapers.nhtsa import NHTSAScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from analysis.scorer_v2 import score_vehicle_v2, get_v2_signal_details
from reports.generator import generate_report
from utils.trace import start_trace, end_trace

logger = logging.getLogger(__name__)

_REPORT_ID_RE = re.compile(r"^[a-f0-9\-]{1,36}$")


def _valid_report_id(report_id: str) -> bool:
    return bool(_REPORT_ID_RE.match(report_id))

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.secret_key = FLASK_SECRET_KEY



SCRAPERS = [
    ("NHTSA", NHTSAScraper),
]


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------


@app.route("/health")
def health():
    from cache.store import health_check as store_health
    from database.models import engine as db_engine
    from sqlalchemy import text

    checks = {}
    checks["store"] = store_health()

    try:
        with db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    healthy = all(checks.values())
    return jsonify({"status": "ok" if healthy else "degraded", "checks": checks}), 200 if healthy else 503


@app.route("/api/admin/clear-cache", methods=["POST"])
def admin_clear_cache():
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403
    from cache.store import clear_vehicle_cache
    count = clear_vehicle_cache()
    logger.info("Admin cache clear: removed %d entries", count)
    return jsonify({"cleared": count})


_bulk_download_status = {"running": False, "progress": "", "error": ""}

GDRIVE_BULK_DB_ID = "1CR4-W4ZRfhrTRfruzZWo4gPsPGEAL4L5"


@app.route("/api/admin/download-bulk-db", methods=["POST"])
def admin_download_bulk_db():
    """Download nhtsa_bulk.db from Google Drive to the Railway volume."""
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403

    if _bulk_download_status["running"]:
        return jsonify({"status": "already_running", "progress": _bulk_download_status["progress"]})

    vol_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
    if not vol_path or not Path(vol_path).is_dir():
        return jsonify({"error": "No Railway volume mounted"}), 400

    dest = Path(vol_path) / "nhtsa_bulk.db"

    def _download():
        _bulk_download_status["running"] = True
        _bulk_download_status["progress"] = "Starting download..."
        _bulk_download_status["error"] = ""
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_BULK_DB_ID}"
            _bulk_download_status["progress"] = "Downloading from Google Drive..."
            logger.info("Downloading nhtsa_bulk.db to %s", dest)
            output = gdown.download(url, str(dest), quiet=False, fuzzy=True)
            if output is None:
                _bulk_download_status["progress"] = "gdown failed, trying direct download..."
                logger.warning("gdown returned None, trying requests fallback")
                _download_gdrive_direct(str(dest))
            size_mb = dest.stat().st_size / 1024 / 1024
            _bulk_download_status["progress"] = f"Done! {size_mb:.0f} MB downloaded"
            logger.info("Download complete: %.0f MB at %s", size_mb, dest)

            import config.settings
            config.settings.BULK_DB_PATH = dest
            import data.bulk_loader
            data.bulk_loader.BULK_DB_PATH = dest
            data.bulk_loader.BULK_DB_URL = f"sqlite:///{dest}"
        except Exception as e:
            _bulk_download_status["error"] = str(e)
            logger.exception("Bulk DB download failed")
        finally:
            _bulk_download_status["running"] = False

    def _download_gdrive_direct(dest_path: str):
        """Fallback: download large Google Drive file with confirmation bypass."""
        import requests as req
        session = req.Session()
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_BULK_DB_ID}"
        r = session.get(url, stream=True)
        for key, value in r.cookies.items():
            if key.startswith("download_warning"):
                url = f"https://drive.google.com/uc?export=download&confirm={value}&id={GDRIVE_BULK_DB_ID}"
                r = session.get(url, stream=True)
                break
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=32 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    mb = downloaded / 1024 / 1024
                    _bulk_download_status["progress"] = f"Downloading... {mb:.0f} MB"

    thread = Thread(target=_download, daemon=True)
    thread.start()
    return jsonify({"status": "started", "destination": str(dest)})


@app.route("/api/admin/download-bulk-db/status")
def admin_download_status():
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403
    return jsonify(_bulk_download_status)


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report/<report_id>")
def report_page(report_id: str):
    if not _valid_report_id(report_id):
        return render_template("index.html", error="Invalid report ID"), 400
    import time
    report = get_report(report_id)
    if not report:
        for _ in range(10):
            time.sleep(0.5)
            report = get_report(report_id)
            if report:
                break
    if not report:
        return render_template("index.html", error="Report not found"), 404
    _postprocess_current_risk(report)
    _ensure_safety_score(report)
    _strip_extra_llm_content(report)
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

    issues = sorted(merged.values(), key=lambda x: x["complaint_count"], reverse=True)
    for i, issue in enumerate(issues, start=1):
        issue["rank"] = i
    cr["top_issues"] = issues

    cr["system_risks"] = [
        sr for sr in cr.get("system_risks", []) if sr.get("system") != "Other"
    ]


_SAFETY_CATS = {
    "Engine", "Transmission", "Brakes", "Steering",
    "Suspension", "Fuel System", "Electrical", "Cooling",
}
_HIGH_SAFETY_CATS = {"Brakes", "Steering", "Fuel System", "Suspension"}


def _ensure_safety_score(report: dict):
    """Guarantee vehicle_summary has a safety_score for legacy cached reports."""
    vs = report.get("sections", {}).get("vehicle_summary", {})
    if vs.get("safety_score") is not None:
        return

    num_recalls = vs.get("total_recalls", 0)
    recall_component = min(25.0, num_recalls * 6.0)

    cr = report.get("sections", {}).get("current_risk", {})
    issues = cr.get("top_issues", [])

    safety_complaints = 0
    severity_sum = 0.0
    count = 0
    for iss in issues:
        cat = iss.get("system", "")
        cc = iss.get("complaint_count", 0)
        if cat in _SAFETY_CATS and cc > 0:
            safety_complaints += cc
            weight = 1.5 if cat in _HIGH_SAFETY_CATS else 1.0
            severity_sum += iss.get("severity", 5) * weight
            count += 1

    volume_component = min(25.0, (safety_complaints / 5) ** 0.6 * 3.0)
    severity_component = min(25.0, (severity_sum / count) * 2.5) if count else 0.0

    score = round(min(100.0, recall_component + volume_component + severity_component), 1)
    vs["safety_score"] = score


def _strip_extra_llm_content(report: dict):
    """Remove LLM-generated fields from sections except Buyer's Verdict
    and Inspection Checklist, and drop removed sections from legacy cache."""
    sections = report.get("sections", {})

    for issue in sections.get("current_risk", {}).get("top_issues", []):
        issue.pop("test_drive_narrative", None)
        issue.pop("what_to_listen_for", None)
        issue.pop("llm_enhanced", None)

    oe = sections.get("owner_experience", {})
    oe.pop("owner_themes", None)

    sections.pop("future_forecast", None)
    sections.pop("red_flags", None)
    sections.pop("negotiation", None)


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------


@app.route("/api/vin-decode")
def api_vin_decode():
    """Decode a VIN using the NHTSA vPIC API and return vehicle details."""
    vin = (request.args.get("vin") or "").strip().upper()
    if not vin or not re.match(r"^[A-HJ-NPR-Z0-9]{17}$", vin):
        return jsonify({"error": "Please enter a valid 17-character VIN"}), 400

    try:
        result = _decode_vin(vin)
        if not result:
            return jsonify({"error": "Could not decode VIN. Please check and try again."}), 404
        return jsonify(result)
    except Exception as exc:
        logger.warning("VIN decode failed for %s: %s", vin, exc)
        return jsonify({"error": "VIN decode service unavailable. Please try again or enter details manually."}), 503


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

    if len(make) > 50 or len(model) > 50:
        return jsonify({"error": "make and model must be under 50 characters"}), 400

    try:
        year = int(year)
        mileage = int(mileage)
    except (ValueError, TypeError):
        return jsonify({"error": "year and mileage must be integers"}), 400

    if not (1960 <= year <= 2027):
        return jsonify({"error": "year must be between 1960 and 2027"}), 400
    if not (0 <= mileage <= 500_000):
        return jsonify({"error": "mileage must be between 0 and 500,000"}), 400

    asking_price = data.get("asking_price")
    if asking_price is not None:
        try:
            asking_price = int(asking_price)
            if asking_price <= 0 or asking_price > 10_000_000:
                asking_price = None
        except (ValueError, TypeError):
            asking_price = None

    vin = (data.get("vin") or "").strip().upper() or None
    if vin and (len(vin) != 17 or not re.match(r"^[A-HJ-NPR-Z0-9]{17}$", vin)):
        vin = None

    zip_code = (data.get("zip_code") or "").strip() or None
    if zip_code and not re.match(r"^\d{5}$", zip_code):
        zip_code = None

    options = {
        "trim": (data.get("trim") or "").strip() or None,
        "engine": (data.get("engine") or "").strip() or None,
        "transmission": (data.get("transmission") or "").strip() or None,
        "drivetrain": (data.get("drivetrain") or "").strip() or None,
        "asking_price": asking_price,
        "vin": vin,
        "zip_code": zip_code,
    }

    cache_key = _make_cache_key(make, model, year, mileage, options)
    cached_id = get_cached_report_id(cache_key)
    if cached_id and get_report(cached_id):
        report_id = cached_id
        init_progress(report_id)
        push_progress(report_id, {"source": "Cache", "status": "complete", "message": "Report loaded from cache"})
        push_progress(report_id, {"source": "system", "status": "done", "message": report_id})
        logger.info("Cache hit for %s — returning report %s", cache_key, report_id)
        return jsonify({"report_id": report_id, "cached": True})

    report_id = str(uuid.uuid4())[:8]
    init_progress(report_id)

    thread = Thread(
        target=_run_analysis,
        args=(report_id, make, model, year, mileage, options, cache_key),
        daemon=True,
    )
    thread.start()

    return jsonify({"report_id": report_id})


@app.route("/api/progress/<report_id>")
def api_progress(report_id: str):
    """Server-Sent Events stream for analysis progress."""
    if not _valid_report_id(report_id):
        return jsonify({"error": "Invalid report ID"}), 400

    def stream():
        import time
        last_idx = 0
        deadline = time.time() + 170
        while time.time() < deadline:
            events = get_progress(report_id)
            for evt in events[last_idx:]:
                yield f"data: {json.dumps(evt)}\n\n"
                last_idx = len(events)
                if evt.get("status") in ("done", "error"):
                    return
            time.sleep(0.3)
        yield f'data: {json.dumps({"source": "system", "status": "error", "message": "Analysis timed out"})}\n\n'

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/report/<report_id>")
def api_report(report_id: str):
    if not _valid_report_id(report_id):
        return jsonify({"error": "Invalid report ID"}), 400
    report = get_report(report_id)
    if not report:
        return jsonify({"error": "Report not found or still processing"}), 404
    return jsonify(report)


@app.route("/api/trace/<report_id>")
def api_trace(report_id: str):
    """Return the debug trace for a report (JSON)."""
    if not _valid_report_id(report_id):
        return jsonify({"error": "Invalid report ID"}), 400

    trace_data = get_trace(report_id)
    if trace_data:
        return jsonify(trace_data)

    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    trace_file = (logs_dir / f"{report_id}.json").resolve()
    if not str(trace_file).startswith(str(logs_dir.resolve())):
        return jsonify({"error": "Invalid report ID"}), 400
    if trace_file.exists():
        return Response(
            trace_file.read_text(encoding="utf-8"),
            mimetype="application/json",
        )

    return jsonify({"error": "Trace not found"}), 404


@app.route("/trace/<report_id>")
def trace_page(report_id: str):
    """Render the debug trace viewer."""
    if not _valid_report_id(report_id):
        return "Invalid report ID", 400
    return render_template("trace.html", report_id=report_id)


# ------------------------------------------------------------------
# Background analysis
# ------------------------------------------------------------------


def _make_cache_key(
    make: str, model: str, year: int, mileage: int, options: dict,
) -> str:
    """Build a deterministic cache key from vehicle parameters."""
    parts = [
        make.upper(), model.upper(), str(year), str(mileage),
        (options.get("trim") or "").upper(),
        (options.get("engine") or "").upper(),
        (options.get("transmission") or "").upper(),
        (options.get("drivetrain") or "").upper(),
        str(options.get("asking_price") or ""),
        str(options.get("zip_code") or ""),
    ]
    return "|".join(parts)


def _emit(report_id: str, source: str, status: str, message: str = ""):
    push_progress(report_id, {
        "source": source,
        "status": status,
        "message": message,
    })


def _run_analysis(
    report_id: str, make: str, model: str, year: int, mileage: int,
    options: dict | None = None, cache_key: str | None = None,
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

    sales_vol = None
    try:
        from data.sales_data import get_sales_volume
        sales_vol = get_sales_volume(make, model, year)
        if sales_vol:
            logger.info("Sales volume for %s %s %d: %d units", make, model, year, sales_vol)
    except Exception as exc:
        logger.warning("Sales volume lookup failed: %s", exc)

    try:
        agg = aggregate(results)
        ma = analyze_mileage(agg, mileage)
        vs = score_vehicle(ma, make=make, model=model, year=year,
                          num_recalls=len(agg.recalls),
                          sales_volume=sales_vol,
                          complaint_dates=agg.complaint_dates)

        v2 = score_vehicle_v2(
            nhtsa_risk_score=vs.reliability_risk_score,
            make=make, model=model, year=year, mileage=mileage,
        )
        vs.reliability_risk_score = v2.risk_score_v2
        vs.letter_grade = v2.letter_grade

        v2_signals = get_v2_signal_details(make, model, year)
        v2_signals["score_components"] = {
            "nhtsa": {"score": v2.nhtsa_component, "weight": 35, "label": "NHTSA Complaints"},
            "tsb": {"score": v2.tsb_component, "weight": 25, "label": "Technical Service Bulletins"},
            "investigation": {"score": v2.investigation_component, "weight": 15, "label": "NHTSA Investigations"},
            "mfr_comm": {"score": v2.mfr_comm_component, "weight": 10, "label": "Manufacturer Communications"},
            "dashboard_light": {"score": v2.dl_qir_component, "weight": 15, "label": "Dashboard Light QIR"},
        }
        v2_signals["wear_factor"] = v2.wear_factor
        v2_signals["mileage_floor"] = v2.mileage_floor
        v2_signals["weighted_contributions"] = v2.weighted_contributions

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
        try:
            _emit(report_id, "Bulk Data", "scraping", "Looking up NHTSA bulk statistics...")
            from data.stats_builder import get_model_stats
            bulk_stats = get_model_stats(make, model, year)
            if bulk_stats:
                logger.info(
                    "Bulk stats found: %d complaints, %sth percentile",
                    bulk_stats.get("total_complaints", 0),
                    bulk_stats.get("complaints_percentile", "?"),
                )
            _emit(report_id, "Bulk Data", "complete", "Bulk data retrieved")
        except Exception as exc:
            logger.info("Bulk data not available (this is OK if not set up): %s", exc)
            _emit(report_id, "Bulk Data", "complete", "Bulk data not available (using scraper data only)")

        price_data = None
        try:
            _emit(report_id, "Pricing", "scraping", "Fetching market prices from MarketCheck...")
            from scrapers.price_scraper import fetch_avg_price
            price_data = fetch_avg_price(
                make, model, year, mileage,
                trim=options.get("trim"),
                engine=options.get("engine"),
                vin=options.get("vin"),
                zip_code=options.get("zip_code"),
            )
            source = price_data.get("source", "estimate")
            count = price_data.get("listings_count", 0)
            match = price_data.get("match_level", "estimate")
            _emit(report_id, "Pricing", "complete",
                  f"Market price: ${price_data.get('avg_price', 0):,} ({source}, {match})")
        except Exception as exc:
            logger.warning("Price fetch failed: %s", exc)
            _emit(report_id, "Pricing", "complete", "Price data not available")

        _emit(report_id, "AI Insights", "scraping", "Generating AI-powered insights and guidance...")
        report = generate_report(
            agg, ma, vs, mileage, options=options,
            bulk_stats=bulk_stats,
            price_data=price_data,
            v2_signals=v2_signals,
        )
        _emit(report_id, "AI Insights", "complete", "AI insights generated")

        set_report(report_id, report)
        if cache_key:
            set_cached_report_id(cache_key, report_id)
            logger.info("Cached report %s under key %s", report_id, cache_key)
        finished_trace = end_trace()
        if finished_trace:
            set_trace(report_id, finished_trace.to_dict())
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
# VIN decoder helper
# ------------------------------------------------------------------

_VIN_DECODE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues"


def _decode_vin(vin: str) -> dict | None:
    """Call NHTSA vPIC to decode a VIN into year/make/model/trim/engine."""
    import requests as http_client

    resp = http_client.get(
        f"{_VIN_DECODE_URL}/{vin}",
        params={"format": "json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("Results", [])
    if not results:
        return None
    r = results[0]

    error_code = r.get("ErrorCode", "")
    error_codes = [c.strip() for c in str(error_code).split(",") if c.strip()]
    if error_codes == ["6"]:
        return None

    make = (r.get("Make") or "").strip()
    model = (r.get("Model") or "").strip()
    year = (r.get("ModelYear") or "").strip()
    if not make or not model or not year:
        return None

    displacement = r.get("DisplacementL") or ""
    cylinders = r.get("EngineCylinders") or ""
    engine_parts = []
    if displacement:
        try:
            engine_parts.append(f"{float(displacement):.1f}L")
        except (ValueError, TypeError):
            pass
    if cylinders:
        engine_parts.append(f"{cylinders}-cyl")
    turbo = (r.get("Turbo") or "").strip()
    if turbo and turbo.lower() not in ("", "no"):
        engine_parts.append("Turbo")
    supercharger = (r.get("OtherEngineInfo") or "").strip()
    if "supercharg" in supercharger.lower():
        engine_parts.append("Supercharged")

    fuel = (r.get("FuelTypePrimary") or "").strip()
    if fuel and fuel.lower() not in ("gasoline", ""):
        engine_parts.append(fuel)

    drive_type = (r.get("DriveType") or "").strip()
    drivetrain = ""
    dl = drive_type.lower()
    if "front" in dl:
        drivetrain = "FWD"
    elif "rear" in dl:
        drivetrain = "RWD"
    elif "all" in dl:
        drivetrain = "AWD"
    elif "4" in dl:
        drivetrain = "4WD"

    trans = (r.get("TransmissionStyle") or "").strip()
    transmission = ""
    tl = trans.lower()
    if "automatic" in tl:
        transmission = "Automatic"
    elif "manual" in tl:
        transmission = "Manual"
    elif "cvt" in tl or "continuously" in tl:
        transmission = "CVT"
    elif "dual" in tl or "dct" in tl:
        transmission = "DCT"

    return {
        "vin": vin,
        "year": year,
        "make": make.upper(),
        "model": model.upper(),
        "trim": (r.get("Trim") or "").strip(),
        "engine": " ".join(engine_parts) if engine_parts else "",
        "transmission": transmission,
        "drivetrain": drivetrain,
        "body_class": (r.get("BodyClass") or "").strip(),
        "vehicle_type": (r.get("VehicleType") or "").strip(),
        "plant_country": (r.get("PlantCountry") or "").strip(),
    }


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------


def create_app() -> Flask:
    sentry_dsn = os.environ.get("SENTRY_DSN", "")
    if sentry_dsn:
        import sentry_sdk
        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=0.1,
            send_default_pii=False,
        )
        logger.info("Sentry error tracking enabled")

    init_db()
    init_store()
    return app
