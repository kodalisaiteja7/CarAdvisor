"""
Test score parity: compare scoring with full nhtsa_bulk.db vs cache-only.

Simulates the production environment by temporarily pointing BULK_DB_PATH
to a non-existent location, forcing the cache fallback.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# --- Step 1: Score with full nhtsa_bulk.db (local behavior) ---
print("=== Test with FULL nhtsa_bulk.db ===")
from data.stats_builder import get_model_stats, get_calibrated_weights, get_complaint_baseline

stats_full = get_model_stats("TOYOTA", "CAMRY", 2018)
weights_full = get_calibrated_weights("TOYOTA", "CAMRY", 2018)
baseline_full = get_complaint_baseline("TOYOTA", "CAMRY", 2018)

print(f"  total_complaints: {stats_full['total_complaints'] if stats_full else 'None'}")
print(f"  global_median:    {stats_full['global_median_complaints'] if stats_full else 'None'}")
print(f"  global_mean:      {stats_full['global_mean_complaints'] if stats_full else 'None'}")
print(f"  crash_rate:       {stats_full['crash_rate'] if stats_full else 'None'}")
print(f"  severity_index:   {stats_full['severity_index'] if stats_full else 'None'}")
print(f"  weights:          {weights_full}")
print(f"  baseline median:  {baseline_full['median_model_complaints'] if baseline_full else 'None'}")

# --- Step 2: Score with cache-only (simulating Railway) ---
print("\n=== Test with CACHE ONLY (simulating Railway) ===")

# Force reload modules to clear caches
import importlib
import data.bulk_loader as bl
import data.stats_builder as sb

# Temporarily point BULK_DB_PATH to non-existent location
original_path = bl.BULK_DB_PATH
bl.BULK_DB_PATH = "DOES_NOT_EXIST.db"
sb.BULK_DB_PATH = "DOES_NOT_EXIST.db"

# Clear the scorer caches too
from analysis import scorer
scorer._cached_weights.clear()
scorer._cached_baseline.clear()

stats_cache = sb.get_model_stats("TOYOTA", "CAMRY", 2018)
weights_cache = sb.get_calibrated_weights("TOYOTA", "CAMRY", 2018)
baseline_cache = sb.get_complaint_baseline("TOYOTA", "CAMRY", 2018)

print(f"  total_complaints: {stats_cache['total_complaints'] if stats_cache else 'None'}")
print(f"  global_median:    {stats_cache['global_median_complaints'] if stats_cache else 'None'}")
print(f"  global_mean:      {stats_cache['global_mean_complaints'] if stats_cache else 'None'}")
print(f"  crash_rate:       {stats_cache['crash_rate'] if stats_cache else 'None'}")
print(f"  severity_index:   {stats_cache['severity_index'] if stats_cache else 'None'}")
print(f"  weights:          {weights_cache}")
print(f"  baseline median:  {baseline_cache['median_model_complaints'] if baseline_cache else 'None'}")

# Restore
bl.BULK_DB_PATH = original_path
sb.BULK_DB_PATH = original_path

# --- Step 3: Compare ---
print("\n=== COMPARISON ===")
if stats_full and stats_cache:
    match = (
        stats_full['total_complaints'] == stats_cache['total_complaints']
        and stats_full['global_median_complaints'] == stats_cache['global_median_complaints']
        and weights_full == weights_cache
        and baseline_full == baseline_cache
    )
    print(f"  Data matches: {match}")
    if not match:
        print("  MISMATCH DETAILS:")
        if stats_full['total_complaints'] != stats_cache['total_complaints']:
            print(f"    complaints: {stats_full['total_complaints']} vs {stats_cache['total_complaints']}")
        if weights_full != weights_cache:
            print(f"    weights: {weights_full} vs {weights_cache}")
        if baseline_full != baseline_cache:
            print(f"    baseline: {baseline_full} vs {baseline_cache}")
elif stats_cache is None:
    print("  CACHE RETURNED None - fallback not working!")
else:
    print(f"  full={stats_full is not None}, cache={stats_cache is not None}")
