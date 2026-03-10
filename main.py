"""Car Advisor — entry point for CLI reports and the web server."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from database.models import init_db
from scrapers.nhtsa import NHTSAScraper
from scrapers.carcomplaints import CarComplaintsScraper
from analysis.aggregator import aggregate
from analysis.mileage_model import analyze_mileage
from analysis.scorer import score_vehicle
from reports.generator import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("car_advisor")

SCRAPERS = [
    ("NHTSA", NHTSAScraper),
    ("CarComplaints", CarComplaintsScraper),
]


def run_report(make: str, model: str, year: int, mileage: int) -> dict:
    """Run the full pipeline and return the report dict."""
    init_db()

    results = []
    for name, scraper_cls in SCRAPERS:
        logger.info("Fetching data from %s ...", name)
        try:
            scraper = scraper_cls()
            data = scraper.fetch(make, model, year)
            results.append(data)
            logger.info("%s: collected %d problems, %d recalls",
                        name,
                        len(data.get("problems", [])),
                        len(data.get("recalls", [])))
        except Exception:
            logger.exception("%s scraper failed — skipping", name)

    if not results:
        logger.error("All scrapers failed. Cannot generate report.")
        return {}

    agg = aggregate(results)
    ma = analyze_mileage(agg, mileage)
    vs = score_vehicle(ma, make=make, model=model, year=year)

    bulk_stats = None
    vector_complaints = None
    try:
        from data.stats_builder import get_model_stats
        from data.vector_search import search_similar_complaints
        bulk_stats = get_model_stats(make, model, year)
        vector_complaints = search_similar_complaints(make, model, year, mileage=mileage)
        if bulk_stats:
            logger.info("Bulk stats loaded: %d complaints", bulk_stats.get("total_complaints", 0))
        if vector_complaints:
            logger.info("Retrieved %d similar complaints from vector store", len(vector_complaints))
    except Exception:
        logger.info("Bulk data not available — using scraper data only")

    report = generate_report(
        agg, ma, vs, mileage,
        vector_complaints=vector_complaints,
        bulk_stats=bulk_stats,
    )
    return report


def print_cli_report(report: dict):
    """Pretty-print the report to the terminal."""
    if not report:
        print("No report data available.")
        return

    v = report["vehicle"]
    meta = report["meta"]
    sections = report["sections"]
    summary = sections["vehicle_summary"]

    print("\n" + "=" * 70)
    print(f"  {v['year']} {v['make']} {v['model']}  —  {v['mileage']:,} miles")
    print("=" * 70)
    print(f"  Reliability Risk Score: {summary['reliability_risk_score']}/100"
          f"  (Grade: {summary['letter_grade']})")
    print(f"  Sources: {', '.join(meta['sources_used'])}")
    print(f"  Problems found: {meta['total_problems_found']}"
          f"  |  Recalls: {meta['total_recalls_found']}")
    print("-" * 70)

    # Recalls
    if summary["recalls"]:
        print(f"\n  RECALLS ({len(summary['recalls'])})")
        for r in summary["recalls"]:
            print(f"    [{r['campaign']}] {r['component']}")
            print(f"      {r['summary'][:120]}")

    # Top issues
    risk = sections["current_risk"]
    if risk["top_issues"]:
        print(f"\n  TOP ISSUES AT {risk['mileage']:,} MILES")
        for issue in risk["top_issues"]:
            print(f"    #{issue['rank']} {issue['system']}"
                  f" ({issue['probability']} probability)"
                  f" — {issue['complaint_count']} complaints")
            print(f"       {issue['description'][:100]}")

    # System risks
    if risk["system_risks"]:
        print("\n  RISK BY SYSTEM")
        for sr in risk["system_risks"]:
            bar_len = int(sr["risk_score"] / 5)
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"    {sr['system']:15s} [{bar}] {sr['risk_score']}/100")

    # Future forecast
    forecast = sections["future_forecast"]
    for window in forecast["forecast_windows"]:
        if window["predicted_issues"]:
            print(f"\n  {window['window_label'].upper()}"
                  f" (by {window['target_mileage']:,} mi)")
            if window["estimated_total_cost"]:
                print(f"    Estimated total cost: {window['estimated_total_cost']}")
            for pi in window["predicted_issues"][:5]:
                cost = pi.get("estimated_cost") or "N/A"
                print(f"    - {pi['system']}: {pi['description'][:80]} ({cost})")

    # Negotiation
    neg = sections["negotiation"]
    if neg["talking_points"]:
        print(f"\n  NEGOTIATION AMMUNITION")
        print(f"    {neg['summary']}")
        for tp in neg["talking_points"][:5]:
            cost = tp.get("estimated_cost") or ""
            print(f"    - {tp['system']}: {tp['issue'][:60]} {cost}")

    # Inspection
    cl = sections["inspection_checklist"]
    must = cl.get("must_check", [])
    if must:
        print(f"\n  MUST-CHECK ITEMS ({len(must)})")
        for item in must[:5]:
            print(f"    ! {item['system']}: {item['what_to_look_for'][:80]}")

    print("\n" + "=" * 70 + "\n")


def serve():
    """Start the Flask web server."""
    from config.settings import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
    from ui.app import create_app

    app = create_app()
    logger.info("Starting Car Advisor on http://%s:%d", FLASK_HOST, FLASK_PORT)
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=False,
    )


def cmd_load_bulk_data(filepath: str | None = None):
    """Load FLAT_CMPL.txt into nhtsa_bulk.db."""
    from data.bulk_loader import load_flat_cmpl
    total = load_flat_cmpl(filepath)
    print(f"Loaded {total:,} complaints into nhtsa_bulk.db")


def cmd_build_stats():
    """Pre-compute per-model statistics from bulk data."""
    from data.stats_builder import build_stats
    build_stats()
    print("Statistics build complete.")


def cmd_build_vectors(reset: bool = False):
    """Embed complaint narratives into ChromaDB vector store."""
    from data.embed_complaints import build_vector_store
    count = build_vector_store(reset=reset)
    print(f"Embedded {count:,} complaints into ChromaDB vector store.")


def cmd_setup_bulk(filepath: str | None = None):
    """Run the full bulk data setup: load → stats → vectors."""
    print("=" * 60)
    print("  Step 1/3: Loading bulk complaints data...")
    print("=" * 60)
    cmd_load_bulk_data(filepath)

    print("\n" + "=" * 60)
    print("  Step 2/3: Building per-model statistics...")
    print("=" * 60)
    cmd_build_stats()

    print("\n" + "=" * 60)
    print("  Step 3/3: Building vector store (this may take a while)...")
    print("=" * 60)
    cmd_build_vectors()

    print("\n" + "=" * 60)
    print("  Bulk data setup complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Car Advisor — used car buying report generator"
    )
    parser.add_argument("command", nargs="?", default=None,
                        help="'serve' | 'load-bulk-data' | 'build-stats' | 'build-vectors' | 'setup-bulk'")
    parser.add_argument("--make", type=str, help="Vehicle make (e.g. Toyota)")
    parser.add_argument("--model", type=str, help="Vehicle model (e.g. Camry)")
    parser.add_argument("--year", type=int, help="Model year (e.g. 2018)")
    parser.add_argument("--mileage", type=int, help="Current mileage (e.g. 75000)")
    parser.add_argument("--json", action="store_true",
                        help="Output report as JSON instead of formatted text")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to FLAT_CMPL.txt (for load-bulk-data)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset vector store before building (for build-vectors)")

    args = parser.parse_args()

    if args.command == "serve":
        serve()
        return

    if args.command == "load-bulk-data":
        cmd_load_bulk_data(args.file)
        return

    if args.command == "build-stats":
        cmd_build_stats()
        return

    if args.command == "build-vectors":
        cmd_build_vectors(reset=args.reset)
        return

    if args.command == "setup-bulk":
        cmd_setup_bulk(args.file)
        return

    if not all([args.make, args.model, args.year, args.mileage]):
        if args.command is None:
            parser.print_help()
            print("\nExamples:")
            print("  python main.py serve")
            print("  python main.py --make Toyota --model Camry --year 2018 --mileage 75000")
            print("\nBulk data management:")
            print("  python main.py load-bulk-data           # Load FLAT_CMPL.txt")
            print("  python main.py build-stats              # Pre-compute statistics")
            print("  python main.py build-vectors            # Build vector store")
            print("  python main.py setup-bulk               # Run all three steps")
            sys.exit(0)
        parser.error("--make, --model, --year, and --mileage are all required for CLI mode")

    report = run_report(args.make, args.model, args.year, args.mileage)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_cli_report(report)


if __name__ == "__main__":
    main()
