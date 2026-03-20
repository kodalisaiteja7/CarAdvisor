"""Flask application — web UI and API routes for CarAdvisr."""

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


@app.route("/api/admin/volume", methods=["GET"])
def admin_volume_list():
    """List all files on the Railway volume with sizes."""
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403

    vol_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
    if not vol_path or not Path(vol_path).is_dir():
        return jsonify({"error": "No volume mounted", "volume_env": vol_path}), 400

    files = []
    total = 0
    for root, dirs, filenames in os.walk(vol_path):
        for fname in filenames:
            fp = Path(root) / fname
            try:
                size = fp.stat().st_size
                total += size
                files.append({
                    "path": str(fp.relative_to(vol_path)),
                    "size_mb": round(size / 1024 / 1024, 2),
                })
            except OSError:
                pass
    files.sort(key=lambda f: f["size_mb"], reverse=True)
    return jsonify({"volume": vol_path, "total_mb": round(total / 1024 / 1024, 2), "files": files})


@app.route("/api/admin/volume/clean", methods=["POST"])
def admin_volume_clean():
    """Delete all files on the Railway volume to free space."""
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403

    vol_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
    if not vol_path or not Path(vol_path).is_dir():
        return jsonify({"error": "No volume mounted"}), 400

    import shutil
    deleted = []
    errors = []
    for root, dirs, filenames in os.walk(vol_path, topdown=False):
        for fname in filenames:
            fp = Path(root) / fname
            try:
                size = fp.stat().st_size
                fp.unlink()
                deleted.append({"path": str(fp.relative_to(vol_path)), "size_mb": round(size / 1024 / 1024, 2)})
            except Exception as e:
                errors.append({"path": str(fp.relative_to(vol_path)), "error": str(e)})
        for d in dirs:
            dp = Path(root) / d
            try:
                dp.rmdir()
            except OSError:
                pass

    # Recreate cache_store directories so the app keeps working
    for sub in ("reports", "progress", "traces", "vcache"):
        (Path(vol_path) / "cache_store" / sub).mkdir(parents=True, exist_ok=True)

    freed = sum(f["size_mb"] for f in deleted)
    return jsonify({"deleted": len(deleted), "freed_mb": round(freed, 2), "files": deleted, "errors": errors})


import threading as _threading
_bulk_download_lock = _threading.Lock()
import threading
_bulk_download_lock = threading.Lock()
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
            import requests as req
            session = req.Session()
            file_id = GDRIVE_BULK_DB_ID

            if dest.exists():
                dest.unlink()

            _bulk_download_status["progress"] = "Requesting file from Google Drive..."
            logger.info("Downloading nhtsa_bulk.db to %s", dest)

            url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
            r = session.get(url, stream=True, timeout=60)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if "text/html" in content_type:
                logger.warning("Got HTML instead of file, trying alt URL...")
                _bulk_download_status["progress"] = "Bypassing confirmation page..."
                r.close()
                url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
                r = session.get(url, stream=True, timeout=60)
                r.raise_for_status()
            _bulk_download_status["progress"] = "Downloading..."

            downloaded = 0
            with open(str(dest), "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        mb = downloaded / 1024 / 1024
                        _bulk_download_status["progress"] = f"Downloading... {mb:.0f} MB"

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

    thread = Thread(target=_download, daemon=True)
    thread.start()
    return jsonify({"status": "started", "destination": str(dest)})


@app.route("/api/admin/download-bulk-db/status")
def admin_download_status():
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403
    return jsonify(_bulk_download_status)


@app.route("/api/admin/download-bulk-db/test")
def admin_download_test():
    """Synchronous: test Google Drive URL and write 1MB sample to volume."""
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403

    import requests as req
    diag = {}

    vol_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
    diag["volume_env"] = vol_path
    diag["volume_exists"] = bool(vol_path and Path(vol_path).is_dir())

    if not diag["volume_exists"]:
        return jsonify(diag)

    existing = Path(vol_path) / "nhtsa_bulk.db"
    diag["bulk_db_exists"] = existing.exists()
    if existing.exists():
        diag["bulk_db_size_mb"] = round(existing.stat().st_size / 1024 / 1024, 1)

    url = f"https://drive.usercontent.google.com/download?id={GDRIVE_BULK_DB_ID}&export=download&confirm=t"
    try:
        r = req.get(url, stream=True, timeout=30)
        diag["gdrive_status"] = r.status_code
        diag["gdrive_content_type"] = r.headers.get("Content-Type", "")
        cl = r.headers.get("Content-Length", "0")
        diag["gdrive_size_mb"] = round(int(cl) / 1024 / 1024, 1) if cl.isdigit() else "unknown"

        # Write a 1MB sample to verify streaming + volume write works
        if "octet-stream" in diag["gdrive_content_type"]:
            sample_path = Path(vol_path) / "_test_1mb.tmp"
            written = 0
            with open(str(sample_path), "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
                        if written >= 1024 * 1024:
                            break
            diag["sample_written_bytes"] = written
            sample_path.unlink(missing_ok=True)
            diag["streaming_works"] = True
        r.close()
    except Exception as e:
        diag["gdrive_error"] = str(e)

    return jsonify(diag)


@app.route("/api/admin/download-bulk-db/run", methods=["POST"])
def admin_download_sync():
    """Synchronous full download — blocks until complete (may take minutes)."""
    secret = request.headers.get("X-Admin-Key") or request.args.get("key")
    if secret != os.environ.get("ADMIN_KEY", "car-advisor-clear-2026"):
        return jsonify({"error": "unauthorized"}), 403

    import requests as req

    vol_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
    if not vol_path or not Path(vol_path).is_dir():
        return jsonify({"error": "No volume"}), 400

    dest = Path(vol_path) / "nhtsa_bulk.db"
    if dest.exists():
        dest.unlink()

    try:
        url = f"https://drive.usercontent.google.com/download?id={GDRIVE_BULK_DB_ID}&export=download&confirm=t"
        r = req.get(url, stream=True, timeout=60)
        r.raise_for_status()

        downloaded = 0
        with open(str(dest), "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        r.close()

        size_mb = dest.stat().st_size / 1024 / 1024
        import config.settings
        config.settings.BULK_DB_PATH = dest
        import data.bulk_loader
        data.bulk_loader.BULK_DB_PATH = dest
        data.bulk_loader.BULK_DB_URL = f"sqlite:///{dest}"

        return jsonify({"status": "done", "size_mb": round(size_mb, 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------


@app.route("/api/subscribe", methods=["POST"])
def api_subscribe():
    """Collect email for newsletter / report delivery, and send report via Resend."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"error": "Valid email required"}), 400

    report_id = (data.get("report_id") or "").strip()
    source = data.get("source", "report")

    from datetime import datetime
    try:
        from database.models import get_session, engine as db_engine
        from sqlalchemy import text
        with db_engine.connect() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS subscribers ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "timestamp TEXT, email TEXT, source TEXT, report_id TEXT)"
            ))
            conn.execute(
                text("INSERT INTO subscribers (timestamp, email, source, report_id) VALUES (:ts, :em, :src, :rid)"),
                {"ts": datetime.utcnow().isoformat(), "em": email, "src": source, "rid": report_id},
            )
            conn.commit()
    except Exception:
        logger.warning("DB subscriber insert failed, falling back to CSV")
        subscribers_file = Path(__file__).resolve().parent.parent / "subscribers.csv"
        line = f"{datetime.utcnow().isoformat()},{email},{source},{report_id}\n"
        is_new = not subscribers_file.exists()
        with open(str(subscribers_file), "a", encoding="utf-8") as f:
            if is_new:
                f.write("timestamp,email,source,report_id\n")
            f.write(line)

    logger.info("New subscriber: %s (source=%s)", email, source)

    if report_id:
        rpt = get_report(report_id)
        if rpt:
            base_url = request.url_root.rstrip("/")
            def _send_bg(em, r, rid, bu):
                try:
                    from services.email_service import send_report_email
                    send_report_email(em, r, rid, bu)
                except Exception:
                    logger.exception("Background email send failed for %s", em)
            Thread(target=_send_bg, args=(email, rpt, report_id, base_url), daemon=True).start()

    return jsonify({"status": "subscribed", "email_sent": True})


_BLOG_ARTICLES = {
    "is-toyota-camry-reliable": {
        "title": "Is the Toyota Camry Reliable? Common Problems, Recalls & Risk Score",
        "excerpt": "We analyzed every NHTSA complaint ever filed against the Toyota Camry. Here's what the data says about its reliability across all model years.",
        "date": "2026-03-09",
        "read_time": 8,
        "category": "Reliability Report",
        "content": """
<p>The Toyota Camry is consistently one of the <strong>best-selling sedans in America</strong>, and for good reason — it has a well-earned reputation for reliability. But is that reputation backed by data? We analyzed <strong>every NHTSA complaint</strong> ever filed against the Camry to find out.</p>

<h2>What the NHTSA Data Shows</h2>
<p>Across all model years, the Camry has fewer complaints per vehicle than the industry average. However, some model years stand out as significantly worse than others.</p>

<h3>Best Model Years (Fewest Complaints)</h3>
<ul>
<li><strong>2020-2024</strong> — Very few complaints, modern safety tech, excellent reliability</li>
<li><strong>2012-2014</strong> — Mature platform, most issues well-documented and resolved</li>
<li><strong>2005-2006</strong> — Bulletproof generation, minimal issues</li>
</ul>

<h3>Model Years to Avoid</h3>
<ul>
<li><strong>2007-2009</strong> — Higher engine oil consumption complaints, dashboard cracking</li>
<li><strong>2018</strong> — First year of new generation, typical first-year teething issues</li>
<li><strong>2002-2003</strong> — Transmission hesitation complaints, some engine sludge reports</li>
</ul>

<h2>Most Common Problems</h2>
<p>Based on NHTSA complaint data, these are the most frequently reported issues:</p>
<ol>
<li><strong>Engine</strong> — Oil consumption, especially in 2007-2011 4-cylinder models</li>
<li><strong>Electrical</strong> — Dashboard rattles, infotainment bugs in newer models</li>
<li><strong>Transmission</strong> — Hesitation on older models, generally reliable on newer ones</li>
<li><strong>Brakes</strong> — Brake pulsation/warping, relatively common but inexpensive to fix</li>
</ol>

<h2>Recall History</h2>
<p>Toyota has issued recalls for airbag inflators (Takata), floor mat entrapment, and fuel pump issues. Most are well-addressed by now, but always verify with a <a href="/">free CarAdvisr report</a> to check for open recalls on a specific vehicle.</p>

<h2>Bottom Line</h2>
<p>The Toyota Camry remains one of the <strong>most reliable sedans you can buy</strong>. Stick with the recommended model years, avoid first-year redesigns, and always run a <a href="/">risk report</a> before purchasing.</p>

<p><strong>Want to check a specific Camry?</strong> <a href="/">Run a free risk report</a> — it takes 30 seconds and analyzes every complaint, recall, and TSB on record.</p>
""",
    },
    "is-honda-civic-reliable": {
        "title": "Is the Honda Civic Reliable? NHTSA Complaints, Known Issues & Buying Guide",
        "excerpt": "The Honda Civic is a perennial favorite for first-time buyers. We analyzed NHTSA data to reveal which model years are safest and which to avoid.",
        "date": "2026-03-09",
        "read_time": 7,
        "category": "Reliability Report",
        "content": """
<p>The Honda Civic is one of the <strong>most popular compact cars</strong> in the world, known for fuel efficiency, low maintenance costs, and strong resale value. But some model years have significant problems. Here's what the NHTSA data tells us.</p>

<h2>Best Civic Model Years</h2>
<ul>
<li><strong>2022-2024 (11th gen)</strong> — Excellent reliability, modern safety, refined platform</li>
<li><strong>2014-2015 (9th gen refresh)</strong> — Proven platform with sorted issues</li>
<li><strong>2009-2011 (8th gen)</strong> — Simple, reliable, and affordable to maintain</li>
</ul>

<h2>Civic Model Years to Avoid</h2>
<ul>
<li><strong>2016-2018 (10th gen)</strong> — Oil dilution in 1.5T engines, AC condenser leaks, infotainment bugs</li>
<li><strong>2006-2008</strong> — Cracked engine blocks (1.8L), paint peeling issues</li>
<li><strong>2001-2003</strong> — Transmission failures, especially in automatic models</li>
</ul>

<h2>Most Common Complaints</h2>
<ol>
<li><strong>Engine (1.5T Oil Dilution)</strong> — Fuel mixing with oil in cold climates, 2016-2018 models</li>
<li><strong>AC Condenser</strong> — Premature failure requiring $800-1,200 replacement</li>
<li><strong>Paint/Clear Coat</strong> — Peeling and fading on 2006-2011 models</li>
<li><strong>Transmission</strong> — Shuddering in CVT models, some older automatics fail</li>
</ol>

<h2>Is the Honda Civic a Good Used Car?</h2>
<p><strong>Yes</strong>, if you pick the right model year. The Civic holds its value exceptionally well and is inexpensive to maintain. Avoid the 2016-2018 1.5-turbo if you live in a cold climate, and always <a href="/">check the risk report</a> before buying.</p>
""",
    },
    "is-ford-f150-reliable": {
        "title": "Is the Ford F-150 Reliable? Complaints, Common Problems & What to Inspect",
        "excerpt": "The Ford F-150 is America's best-selling vehicle. We analyzed NHTSA complaints to find which years are bulletproof and which have costly problems.",
        "date": "2026-03-09",
        "read_time": 8,
        "category": "Reliability Report",
        "content": """
<p>The Ford F-150 has been <strong>America's best-selling vehicle for over 40 years</strong>. With millions on the road, there's a huge used market. But not all F-150s are created equal — some model years have serious issues.</p>

<h2>Best F-150 Model Years</h2>
<ul>
<li><strong>2021-2023 (14th gen)</strong> — Refined platform, PowerBoost hybrid option, strong reliability</li>
<li><strong>2018-2020 (13th gen refresh)</strong> — Sorted out early 13th-gen issues, 5.0L V8 is rock-solid</li>
<li><strong>2012-2014 (12th gen)</strong> — Proven platform, widely available, affordable</li>
</ul>

<h2>F-150 Model Years to Avoid</h2>
<ul>
<li><strong>2015-2017</strong> — First aluminum body years, some panel fit issues, 2.7L EcoBoost carbon buildup</li>
<li><strong>2004-2006</strong> — Spark plug ejection issues (5.4L), cam phaser problems</li>
<li><strong>2010-2011</strong> — Some powertrain issues, but generally acceptable</li>
</ul>

<h2>Most Common Complaints by Engine</h2>
<h3>5.0L Coyote V8</h3>
<p>Generally very reliable. Some oil consumption complaints in early years. Tick noise is common but usually harmless.</p>

<h3>3.5L EcoBoost</h3>
<p>Powerful and efficient. Watch for timing chain issues in early models (2011-2013) and condensation/intercooler leaks.</p>

<h3>2.7L EcoBoost</h3>
<p>Carbon buildup on intake valves is the main concern. Regular maintenance (walnut blasting every 60-80k miles) prevents this.</p>

<h2>What to Inspect Before Buying</h2>
<ol>
<li>Check for cam phaser noise on startup (5.0L)</li>
<li>Inspect for oil leaks around the turbo oil lines (EcoBoost)</li>
<li>Test the transmission — 10-speed should shift smoothly without hunting</li>
<li>Check for rust on frame and cab corners (especially northern trucks)</li>
<li>Verify all recalls have been completed with a <a href="/">free risk report</a></li>
</ol>
""",
    },
    "free-carfax-alternative": {
        "title": "Free Carfax Alternative: How to Check a Used Car Without Paying $40",
        "excerpt": "You don't need to spend $40-$100 on Carfax. Here are free tools and methods to check any used car's history, complaints, and reliability.",
        "date": "2026-03-09",
        "read_time": 6,
        "category": "Buyer Guide",
        "content": """
<p>Carfax charges <strong>$39.99 for a single report</strong> or $99.99 for six. If you're shopping for a used car and checking multiple vehicles, those costs add up fast. The good news? You can get most of the same information — and even more — for free.</p>

<h2>What Carfax Tells You (and Its Limitations)</h2>
<p>Carfax provides accident history, ownership count, service records, and title information. However, it has significant gaps:</p>
<ul>
<li>Only shows accidents reported to insurance or police — many aren't</li>
<li>Service records depend on shops that report to Carfax (many don't)</li>
<li>Doesn't tell you what <em>will</em> go wrong based on complaint patterns</li>
<li>Doesn't provide risk scores or reliability ratings</li>
</ul>

<h2>Free Alternatives</h2>

<h3>1. CarAdvisr (This Site)</h3>
<p><a href="/">CarAdvisr</a> analyzes <strong>2.18 million NHTSA complaints</strong> to give you a comprehensive risk report including:</p>
<ul>
<li>Risk score (0-100) based on real complaint data</li>
<li>Every recall on record</li>
<li>AI-powered inspection checklist tailored to the mileage</li>
<li>Price analysis comparing to market data</li>
<li>Buyer's verdict: Buy or Walk Away</li>
</ul>
<p>It's completely free, no signup required.</p>

<h3>2. NHTSA Recall Check</h3>
<p>The government's <a href="https://www.nhtsa.gov/recalls" target="_blank">recall lookup tool</a> lets you search by VIN for open recalls. This is free and authoritative.</p>

<h3>3. NMVTIS (Title Check)</h3>
<p>The National Motor Vehicle Title Information System provides title history for about $2-10 per check through approved providers.</p>

<h3>4. VINCheck by NICB</h3>
<p>The National Insurance Crime Bureau offers a free VIN check for theft and total loss records at <a href="https://www.nicb.org/vincheck" target="_blank">nicb.org/vincheck</a>.</p>

<h2>The Best Free Strategy</h2>
<ol>
<li><strong>Run a <a href="/">CarAdvisr report</a></strong> for complaints, recalls, and risk assessment</li>
<li><strong>Check NHTSA.gov</strong> for open recalls by VIN</li>
<li><strong>Check NICB VINCheck</strong> for theft/total loss</li>
<li><strong>Google "[car year model] problems"</strong> for forum discussions</li>
<li><strong>Get a pre-purchase inspection</strong> from an independent mechanic ($100-200)</li>
</ol>

<p>This combination gives you <strong>more actionable information</strong> than a Carfax report — and the only cost is the mechanic inspection.</p>
""",
    },
    "used-car-inspection-checklist": {
        "title": "How to Inspect a Used Car Before Buying: Complete 2026 Checklist",
        "excerpt": "Don't buy a used car without checking these 15 critical items. A complete inspection guide that catches problems before they cost you thousands.",
        "date": "2026-03-09",
        "read_time": 10,
        "category": "Buyer Guide",
        "content": """
<p>Buying a used car without a proper inspection is like buying a house without a home inspection — it's a gamble that can cost you thousands. This checklist covers <strong>everything you should check</strong> before handing over your money.</p>

<h2>Before You Visit: Research Phase</h2>
<ol>
<li><strong>Run a <a href="/">CarAdvisr risk report</a></strong> — see all NHTSA complaints, recalls, and known issues for the specific year/make/model</li>
<li><strong>Check the VIN</strong> for recalls at NHTSA.gov</li>
<li><strong>Research fair market price</strong> — know what the car should cost before negotiating</li>
</ol>

<h2>Exterior Inspection</h2>
<ol>
<li><strong>Paint consistency</strong> — Mismatched colors or texture between panels indicates bodywork/accident repair</li>
<li><strong>Panel gaps</strong> — Uneven gaps between doors, hood, and trunk suggest collision damage</li>
<li><strong>Tire wear</strong> — Uneven wear indicates alignment issues or suspension problems. Check all four tires.</li>
<li><strong>Rust</strong> — Check wheel wells, rocker panels, door bottoms, and under the car. Surface rust is cosmetic; structural rust is a deal-breaker.</li>
<li><strong>Glass</strong> — Check windshield for chips/cracks. Verify all windows match (same manufacturer stamp).</li>
</ol>

<h2>Under the Hood</h2>
<ol>
<li><strong>Oil condition</strong> — Pull the dipstick. Black is normal; milky/frothy suggests a head gasket leak (expensive). Low oil suggests consumption issues.</li>
<li><strong>Coolant</strong> — Should be green, orange, or pink (not brown or milky). Brown coolant means neglected maintenance; milky means oil contamination.</li>
<li><strong>Belts and hoses</strong> — Look for cracks, fraying, or bulging. Replacement is $200-600.</li>
<li><strong>Battery terminals</strong> — Heavy corrosion suggests electrical issues or an old battery.</li>
</ol>

<h2>Interior Check</h2>
<ol>
<li><strong>All electronics</strong> — Test every button, switch, window, lock, mirror, and light</li>
<li><strong>AC/Heat</strong> — Run the AC on max cold for 5 minutes. Should blow cold within 2 minutes. AC repair is $500-1,500+.</li>
<li><strong>Odors</strong> — Musty smell means water leaks or flood damage. Sweet smell could be a coolant leak.</li>
</ol>

<h2>Test Drive (Most Important!)</h2>
<ol>
<li><strong>Cold start</strong> — Visit when the car hasn't been warmed up. Listen for unusual noises on startup.</li>
<li><strong>Transmission</strong> — Should shift smoothly. Jerking, hesitation, or slipping needs immediate attention.</li>
<li><strong>Brakes</strong> — Should stop straight without pulling, pulsation, or grinding</li>
<li><strong>Steering</strong> — Should track straight on a level road. No vibration at highway speed.</li>
<li><strong>Highway driving</strong> — Get up to 60+ mph. Listen for wind noise, vibrations, or drone.</li>
</ol>

<h2>Final Steps</h2>
<ul>
<li><strong>Get a pre-purchase inspection</strong> from an independent mechanic ($100-200). This is the best money you'll spend.</li>
<li><strong>Review the risk report</strong> from <a href="/">CarAdvisr</a> to understand what issues are most likely at the current mileage</li>
<li><strong>Negotiate based on findings</strong> — use any issues found as leverage for a lower price</li>
</ul>
""",
    },
}


@app.route("/blog")
def blog_index():
    return render_template("blog_index.html", articles=_BLOG_ARTICLES)


@app.route("/blog/<slug>")
def blog_article(slug: str):
    article = _BLOG_ARTICLES.get(slug)
    if not article:
        return render_template("index.html", error="Article not found"), 404
    related = {k: v for k, v in list(_BLOG_ARTICLES.items())[:4] if k != slug}
    return render_template("blog_article.html", article=article, slug=slug, related=related)


@app.route("/privacy")
def privacy_policy():
    return render_template("privacy.html")


@app.route("/terms")
def terms_of_service():
    return render_template("terms.html")


@app.route("/robots.txt")
def robots_txt():
    content = "User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /trace/\nDisallow: /api/admin/\n\nSitemap: " + request.host_url.rstrip("/") + "/sitemap.xml\n"
    return Response(content, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml():
    from datetime import datetime
    base = request.host_url.rstrip("/")
    now = datetime.utcnow().strftime("%Y-%m-%d")

    urls = [
        {"loc": base + "/", "changefreq": "weekly", "priority": "1.0"},
        {"loc": base + "/blog", "changefreq": "weekly", "priority": "0.8"},
    ]

    for slug in _BLOG_ARTICLES:
        urls.append({"loc": f"{base}/blog/{slug}", "changefreq": "monthly", "priority": "0.7"})

    xml_entries = []
    for u in urls:
        xml_entries.append(
            f"  <url>\n    <loc>{u['loc']}</loc>\n    <lastmod>{now}</lastmod>\n"
            f"    <changefreq>{u['changefreq']}</changefreq>\n    <priority>{u['priority']}</priority>\n  </url>"
        )

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + "\n".join(xml_entries) + "\n</urlset>\n"
    return Response(xml, mimetype="application/xml")


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
    tpl = "report_v1.html" if request.args.get("layout") == "v1" else "report.html"
    return render_template(tpl, report=report, report_id=report_id)


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

    with _inflight_lock:
        if cache_key in _inflight_reports:
            inflight_id = _inflight_reports[cache_key]
            logger.info("Dedup: attaching to in-flight report %s for %s", inflight_id, cache_key)
            return jsonify({"report_id": inflight_id, "cached": False})

        report_id = str(uuid.uuid4())[:8]
        _inflight_reports[cache_key] = report_id

    init_progress(report_id)

    def _wrapped_analysis():
        try:
            _run_analysis(report_id, make, model, year, mileage, options, cache_key)
        finally:
            with _inflight_lock:
                _inflight_reports.pop(cache_key, None)

    thread = Thread(target=_wrapped_analysis, daemon=True)
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

_REPORT_SEMAPHORE = threading.Semaphore(20)
_inflight_reports: dict[str, str] = {}
_inflight_lock = threading.Lock()


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
    acquired = _REPORT_SEMAPHORE.acquire(timeout=60)
    if not acquired:
        _emit(report_id, "system", "error", "Server is busy, please try again in a moment.")
        return

    options = options or {}

    try:
        _run_analysis_inner(report_id, make, model, year, mileage, options, cache_key)
    finally:
        _REPORT_SEMAPHORE.release()


def _run_analysis_inner(
    report_id: str, make: str, model: str, year: int, mileage: int,
    options: dict, cache_key: str | None = None,
):
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
            "brand_reliability": {"score": v2.dl_qir_component, "weight": 15, "label": "Brand Reliability Index"},
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

        from concurrent.futures import ThreadPoolExecutor

        def _fetch_bulk():
            _emit(report_id, "Bulk Data", "scraping", "Looking up NHTSA bulk statistics...")
            try:
                from data.stats_builder import get_model_stats
                stats = get_model_stats(make, model, year)
                if stats:
                    logger.info(
                        "Bulk stats found: %d complaints, %sth percentile",
                        stats.get("total_complaints", 0),
                        stats.get("complaints_percentile", "?"),
                    )
                _emit(report_id, "Bulk Data", "complete", "Bulk data retrieved")
                return stats
            except Exception as exc:
                logger.info("Bulk data not available: %s", exc)
                _emit(report_id, "Bulk Data", "complete", "Bulk data not available (using scraper data only)")
                return None

        def _fetch_price():
            _emit(report_id, "Pricing", "scraping", "Fetching market prices from MarketCheck...")
            try:
                from scrapers.price_scraper import fetch_avg_price
                pd = fetch_avg_price(
                    make, model, year, mileage,
                    trim=options.get("trim"),
                    engine=options.get("engine"),
                    vin=options.get("vin"),
                    zip_code=options.get("zip_code"),
                )
                _emit(report_id, "Pricing", "complete",
                      "Market pricing data collected")
                return pd
            except Exception as exc:
                logger.warning("Price fetch failed: %s", exc)
                _emit(report_id, "Pricing", "complete", "Price data not available")
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_bulk = pool.submit(_fetch_bulk)
            f_price = pool.submit(_fetch_price)

        bulk_stats = f_bulk.result()
        price_data = f_price.result()

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

    raw_trim = (r.get("Trim") or "").strip()
    series = (r.get("Series") or "").strip()
    if "," in raw_trim:
        trim = series if series and "," not in series else raw_trim.split(",")[0].strip()
    else:
        trim = raw_trim

    return {
        "vin": vin,
        "year": year,
        "make": make.upper(),
        "model": model.upper(),
        "trim": trim,
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

    from config.settings import GA_MEASUREMENT_ID, CLARITY_PROJECT_ID
    app.config["GA_MEASUREMENT_ID"] = GA_MEASUREMENT_ID
    app.config["CLARITY_PROJECT_ID"] = CLARITY_PROJECT_ID

    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600

    @app.after_request
    def _add_cache_headers(response):
        if request.path.startswith("/static/"):
            response.cache_control.public = True
            response.cache_control.max_age = 86400
        return response

    init_db()
    init_store()
    return app
