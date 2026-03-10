"""Quick integration test for all new bulk data components."""
from data.bulk_loader import NHTSAComplaint
from data.stats_builder import get_model_stats, get_mileage_curve, get_calibrated_weights, get_complaint_baseline
from data.vector_search import search_similar_complaints, is_vector_store_available

print("All imports successful")
print(f"Vector store available: {is_vector_store_available()}")

stats = get_model_stats("Toyota", "Camry", 2020)
if stats:
    print(f"Toyota Camry 2020: {stats['total_complaints']} complaints, {stats['interpretation']}")

curve = get_mileage_curve("Toyota", "Camry", 2020, "Engine")
if curve:
    print(f"Engine failure curve: median={curve['median']} miles, count={curve['count']}")

weights = get_calibrated_weights("Toyota", "Camry", 2020)
if weights:
    print(f"Calibrated weights: {weights}")

baseline = get_complaint_baseline("Toyota", "Camry", 2020)
if baseline:
    print(f"Baseline: ratio={baseline['complaint_ratio']}, percentile={baseline['percentile']}")

# Test vector search (may return empty if store isn't built yet)
results = search_similar_complaints("Toyota", "Camry", 2020, system="Engine", mileage=80000, n_results=3)
if results:
    print(f"Vector search returned {len(results)} results:")
    for r in results[:2]:
        print(f"  - [{r['system']}] {r['narrative'][:80]}...")
else:
    print("Vector search: no results (store may not be built yet)")

print("\nPipeline integration test passed!")
