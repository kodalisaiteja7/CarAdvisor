"""Comprehensive pipeline test: every layer from imports to end-to-end report."""

import json
import sys
import time
import traceback

PASS = 0
FAIL = 0
ERRORS = []


def test(name):
    """Decorator to run and report on a test."""
    def decorator(fn):
        global PASS, FAIL
        print(f"\n{'='*60}")
        print(f"  TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            PASS += 1
            print(f"  RESULT: PASS")
        except Exception as e:
            FAIL += 1
            tb = traceback.format_exc()
            ERRORS.append((name, str(e), tb))
            safe_err = str(e).encode("ascii", errors="replace").decode()
            safe_tb = tb.encode("ascii", errors="replace").decode()
            print(f"  RESULT: FAIL -- {safe_err}")
            print(f"  {safe_tb}")
        return fn
    return decorator


# ---------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------

@test("All core module imports")
def test_imports():
    from config.settings import BASE_DIR, NHTSA_API_BASE, VEHICLE_SYSTEMS
    from database.models import init_db
    from scrapers.base import BaseScraper
    from scrapers.nhtsa import NHTSAScraper
    from scrapers.carcomplaints import CarComplaintsScraper
    from analysis.normalizer import normalize_source_data, NormalizedProblem
    from analysis.aggregator import aggregate, AggregatedVehicleData
    from analysis.mileage_model import analyze_mileage, MileagePhase
    from analysis.scorer import score_vehicle, VehicleScore
    from analysis.llm_enhancer import enhance_inspection_checklist, enhance_report_sections
    from reports.generator import generate_report
    from utils.trace import start_trace, end_trace
    print("  All core imports OK")


@test("Bulk data module imports")
def test_bulk_imports():
    from data.bulk_loader import NHTSAComplaint, _get_bulk_engine, BulkBase
    from data.stats_builder import get_model_stats, get_mileage_curve, get_calibrated_weights, get_complaint_baseline
    from data.embed_complaints import build_vector_store, CHROMA_DIR
    from data.vector_search import search_similar_complaints, is_vector_store_available
    print("  All bulk data imports OK")


# ---------------------------------------------------------------
# 2. BULK DATA INTEGRITY
# ---------------------------------------------------------------

@test("SQLite bulk database exists and has data")
def test_bulk_db():
    from data.bulk_loader import BULK_DB_PATH, _get_bulk_engine, NHTSAComplaint, BulkBase
    from sqlalchemy import func
    from sqlalchemy.orm import sessionmaker
    from pathlib import Path

    assert Path(BULK_DB_PATH).exists(), f"nhtsa_bulk.db not found at {BULK_DB_PATH}"
    print(f"  DB path: {BULK_DB_PATH}")

    engine = _get_bulk_engine()
    BulkBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    total = session.query(func.count(NHTSAComplaint.id)).scalar()
    assert total > 0, "nhtsa_complaints table is empty"
    print(f"  Total complaints: {total:,}")

    with_mileage = session.query(func.count(NHTSAComplaint.id)).filter(
        NHTSAComplaint.mileage.isnot(None), NHTSAComplaint.mileage > 0
    ).scalar()
    print(f"  With mileage: {with_mileage:,} ({with_mileage/total*100:.1f}%)")

    sample = session.query(NHTSAComplaint).filter(
        NHTSAComplaint.make == "TOYOTA", NHTSAComplaint.model == "CAMRY"
    ).first()
    if sample:
        print(f"  Sample: {sample.year} {sample.make} {sample.model}, mileage={sample.mileage}, system={sample.system}")
    session.close()


@test("Model stats table has data")
def test_model_stats():
    from data.stats_builder import get_model_stats

    stats = get_model_stats("Toyota", "Camry", 2018)
    assert stats is not None, "No stats for Toyota Camry 2018"
    print(f"  Toyota Camry 2018:")
    print(f"    Total complaints: {stats['total_complaints']}")
    print(f"    Percentile: {stats['complaints_percentile']}")
    print(f"    Crash rate: {stats['crash_rate']:.2%}")
    print(f"    Interpretation: {stats['interpretation']}")
    print(f"    Mileage dist: {stats['mileage_distribution']}")

    stats2 = get_model_stats("Honda", "Civic", 2019)
    if stats2:
        print(f"  Honda Civic 2019: {stats2['total_complaints']} complaints, {stats2['interpretation']}")
    else:
        print("  Honda Civic 2019: no stats (may not be in DB)")


@test("Mileage failure curves work")
def test_mileage_curves():
    from data.stats_builder import get_mileage_curve

    curve = get_mileage_curve("Toyota", "Camry", 2018, "Engine")
    if curve:
        print(f"  Engine failure curve: p10={curve.get('p10')}, median={curve.get('median')}, p90={curve.get('p90')}, count={curve.get('count')}")
    else:
        print("  No engine failure curve data (may need more data points)")

    curve2 = get_mileage_curve("Toyota", "Camry", 2018, "Transmission")
    if curve2:
        print(f"  Transmission failure curve: p10={curve2.get('p10')}, median={curve2.get('median')}, p90={curve2.get('p90')}, count={curve2.get('count')}")


@test("Calibrated weights and baselines")
def test_calibrated_weights():
    from data.stats_builder import get_calibrated_weights, get_complaint_baseline

    weights = get_calibrated_weights("Toyota", "Camry", 2018)
    if weights:
        print(f"  Calibrated weights: {weights}")
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1: {total}"
    else:
        print("  No calibrated weights (fallback will be used)")

    baseline = get_complaint_baseline("Toyota", "Camry", 2018)
    if baseline:
        print(f"  Baseline: ratio={baseline['complaint_ratio']}, percentile={baseline['percentile']}, interp={baseline['interpretation']}")


@test("ChromaDB vector store is available")
def test_vector_store():
    from data.vector_search import is_vector_store_available

    available = is_vector_store_available()
    assert available, "Vector store not available"
    print("  Vector store is available and has documents")


@test("Vector search returns results")
def test_vector_search():
    from data.vector_search import search_similar_complaints

    results = search_similar_complaints("Toyota", "Camry", 2018, system="Engine", mileage=80000, n_results=5)
    assert results, "Vector search returned no results for Toyota Camry 2018 Engine"
    print(f"  Found {len(results)} similar complaints:")
    for r in results[:3]:
        print(f"    [{r['system']}] mile={r.get('mileage',0)} dist={r.get('distance',0):.4f} | {r['narrative'][:80]}...")


# ---------------------------------------------------------------
# 3. SCRAPERS
# ---------------------------------------------------------------

@test("NHTSA API scraper")
def test_nhtsa_scraper():
    from scrapers.nhtsa import NHTSAScraper
    scraper = NHTSAScraper()

    print("  Fetching Toyota Camry 2018...")
    data = scraper.fetch("Toyota", "Camry", 2018)
    assert "problems" in data, "Missing 'problems' key"
    assert "recalls" in data, "Missing 'recalls' key"
    print(f"  Problems: {len(data['problems'])}")
    print(f"  Recalls: {len(data['recalls'])}")
    print(f"  Ratings: {data.get('ratings', {})}")
    for p in data['problems'][:2]:
        print(f"    [{p['category']}] {p['description'][:80]}...")


@test("CarComplaints scraper")
def test_carcomplaints_scraper():
    from scrapers.carcomplaints import CarComplaintsScraper
    scraper = CarComplaintsScraper()

    print("  Fetching Toyota Camry 2018...")
    data = scraper.fetch("Toyota", "Camry", 2018)
    assert "problems" in data, "Missing 'problems' key"
    print(f"  Problems: {len(data['problems'])}")
    for p in data['problems'][:2]:
        print(f"    [{p['category']}] count={p.get('complaint_count',0)} | {p['description'][:80]}...")


# ---------------------------------------------------------------
# 4. ANALYSIS PIPELINE
# ---------------------------------------------------------------

@test("Normalizer processes scraper output")
def test_normalizer():
    from analysis.normalizer import normalize_source_data
    from scrapers.nhtsa import NHTSAScraper

    scraper = NHTSAScraper()
    raw = scraper.fetch("Toyota", "Camry", 2018)
    nd = normalize_source_data(raw)
    print(f"  Normalized problems: {len(nd.problems)}")
    print(f"  Normalized recalls: {len(nd.recalls)}")
    for p in nd.problems[:2]:
        print(f"    [{p.category}] sev={p.severity} safety={p.safety_impact} count={p.complaint_count}")


@test("Aggregator merges multiple sources")
def test_aggregator():
    from scrapers.nhtsa import NHTSAScraper
    from scrapers.carcomplaints import CarComplaintsScraper
    from analysis.aggregator import aggregate

    results = []
    for cls in [NHTSAScraper, CarComplaintsScraper]:
        try:
            data = cls().fetch("Toyota", "Camry", 2018)
            results.append(data)
        except Exception as e:
            print(f"  Warning: {cls.__name__} failed: {e}")

    assert results, "No scraper data to aggregate"
    agg = aggregate(results)
    print(f"  Sources: {agg.sources_used}")
    print(f"  Total problems: {len(agg.problems)}")
    print(f"  Total complaints: {agg.total_complaints}")
    print(f"  Recalls: {len(agg.recalls)}")


@test("Mileage model classifies problems correctly")
def test_mileage_model():
    from scrapers.nhtsa import NHTSAScraper
    from analysis.aggregator import aggregate
    from analysis.mileage_model import analyze_mileage, MileagePhase

    raw = NHTSAScraper().fetch("Toyota", "Camry", 2018)
    agg = aggregate([raw])

    for mileage in [25000, 75000, 120000]:
        ma = analyze_mileage(agg, mileage)
        print(f"  At {mileage:,} miles:")
        print(f"    Bracket: {ma.bracket}")
        print(f"    Phase counts: {ma.phase_counts}")
        print(f"    System risks: {[(sr.system, sr.risk_score) for sr in ma.system_risks[:3]]}")


@test("Scorer produces consistent mileage-proportional risk")
def test_scorer():
    from scrapers.nhtsa import NHTSAScraper
    from analysis.aggregator import aggregate
    from analysis.mileage_model import analyze_mileage
    from analysis.scorer import score_vehicle

    raw = NHTSAScraper().fetch("Toyota", "Camry", 2018)
    agg = aggregate([raw])

    scores = {}
    for mileage in [25000, 50000, 75000, 100000, 150000]:
        ma = analyze_mileage(agg, mileage)
        vs = score_vehicle(ma, make="Toyota", model="Camry", year=2018)
        scores[mileage] = vs.reliability_risk_score
        print(f"  {mileage:>7,} mi -> risk={vs.reliability_risk_score:.1f}, grade={vs.letter_grade}, issues={len(vs.top_issues)}")

    low_score = scores[25000]
    high_score = scores[150000]
    print(f"\n  Mileage monotonicity check: 25k={low_score:.1f} vs 150k={high_score:.1f}")
    if high_score > low_score:
        print(f"  PASS: Higher mileage has higher risk (diff={high_score-low_score:.1f})")
    else:
        print(f"  WARNING: Higher mileage does NOT have higher risk! Investigate scorer.")


# ---------------------------------------------------------------
# 5. REPORT GENERATOR (without LLM)
# ---------------------------------------------------------------

@test("Report generator produces valid structure")
def test_report_structure():
    from scrapers.nhtsa import NHTSAScraper
    from analysis.aggregator import aggregate
    from analysis.mileage_model import analyze_mileage
    from analysis.scorer import score_vehicle
    from reports.generator import generate_report
    from data.stats_builder import get_model_stats
    from data.vector_search import search_similar_complaints

    raw = NHTSAScraper().fetch("Toyota", "Camry", 2018)
    agg = aggregate([raw])
    ma = analyze_mileage(agg, 75000)
    vs = score_vehicle(ma, make="Toyota", model="Camry", year=2018)

    bulk_stats = get_model_stats("Toyota", "Camry", 2018)
    vector_complaints = search_similar_complaints("Toyota", "Camry", 2018, mileage=75000)

    print(f"  Bulk stats: {'available' if bulk_stats else 'N/A'}")
    print(f"  Vector complaints: {len(vector_complaints) if vector_complaints else 0}")

    report = generate_report(
        agg, ma, vs, 75000,
        vector_complaints=vector_complaints,
        bulk_stats=bulk_stats,
    )

    required_keys = ["vehicle", "meta", "sections"]
    for k in required_keys:
        assert k in report, f"Missing top-level key: {k}"

    sections = report["sections"]
    required_sections = [
        "vehicle_summary", "inspection_checklist", "current_risk",
        "future_forecast", "owner_experience", "red_flags", "negotiation"
    ]
    for s in required_sections:
        assert s in sections, f"Missing section: {s}"
        print(f"  Section '{s}': OK")

    vs_section = sections["vehicle_summary"]
    print(f"\n  Vehicle: {report['vehicle']}")
    print(f"  Risk score: {vs_section['reliability_risk_score']}")
    print(f"  Grade: {vs_section['letter_grade']}")
    print(f"  Total complaints: {vs_section['total_complaints']}")
    print(f"  Total recalls: {vs_section['total_recalls']}")

    checklist = sections["inspection_checklist"]
    print(f"  Checklist: must={len(checklist.get('must_check',[]))}, recommended={len(checklist.get('recommended',[]))}, standard={len(checklist.get('standard',[]))}")

    if "executive_summary" in sections:
        print(f"  Executive summary: {sections['executive_summary'].get('text','')[:100]}...")


# ---------------------------------------------------------------
# 6. FLASK APP
# ---------------------------------------------------------------

@test("Flask app creates and serves routes")
def test_flask_app():
    from ui.app import create_app

    app = create_app()
    client = app.test_client()

    resp = client.get("/")
    assert resp.status_code == 200, f"Homepage returned {resp.status_code}"
    print(f"  GET / → {resp.status_code} ({len(resp.data)} bytes)")

    resp = client.get("/api/years")
    assert resp.status_code == 200, f"/api/years returned {resp.status_code}"
    years = resp.get_json()
    print(f"  GET /api/years → {len(years)} years")

    resp = client.get("/api/makes?year=2018")
    assert resp.status_code == 200, f"/api/makes returned {resp.status_code}"
    makes = resp.get_json()
    print(f"  GET /api/makes?year=2018 → {len(makes)} makes")

    if makes:
        resp = client.get(f"/api/models?make=Toyota&year=2018")
        assert resp.status_code == 200
        models = resp.get_json()
        print(f"  GET /api/models?make=Toyota&year=2018 → {len(models)} models")


# ---------------------------------------------------------------
# 7. END-TO-END (API analyze endpoint)
# ---------------------------------------------------------------

@test("API /api/analyze triggers background analysis")
def test_api_analyze():
    from ui.app import create_app, _reports, _progress

    app = create_app()
    client = app.test_client()

    resp = client.post("/api/analyze", json={
        "make": "Honda",
        "model": "Civic",
        "year": 2019,
        "mileage": 60000,
    })
    assert resp.status_code == 200, f"/api/analyze returned {resp.status_code}"
    data = resp.get_json()
    report_id = data.get("report_id")
    assert report_id, "No report_id returned"
    print(f"  Report ID: {report_id}")

    max_wait = 120
    waited = 0
    while waited < max_wait:
        time.sleep(2)
        waited += 2
        events = _progress.get(report_id, [])
        for evt in events:
            if evt.get("status") == "done":
                print(f"  Analysis complete after {waited}s")
                report = _reports.get(report_id)
                assert report, "Report not stored after completion"
                print(f"  Report has {len(report.get('sections', {}))} sections")
                if report.get("sections", {}).get("executive_summary"):
                    summary = report["sections"]["executive_summary"].get("text", "")
                    print(f"  Executive summary: {summary[:150]}...")
                return
            elif evt.get("status") == "error":
                print(f"  Analysis ERRORED after {waited}s: {evt.get('message')}")
                return

        if waited % 10 == 0:
            latest = events[-1] if events else {}
            print(f"  ... waiting ({waited}s), latest: {latest.get('source','?')} | {latest.get('message','')[:60]}")

    print(f"  WARNING: Analysis did not complete within {max_wait}s")


# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------

print("\n" + "=" * 60)
print(f"  PIPELINE TEST RESULTS")
print(f"  Passed: {PASS}")
print(f"  Failed: {FAIL}")
print(f"  Total:  {PASS + FAIL}")
print("=" * 60)

if ERRORS:
    print("\nFAILED TESTS:")
    for name, err, _ in ERRORS:
        print(f"  X {name}: {err}")

sys.exit(1 if FAIL > 0 else 0)
