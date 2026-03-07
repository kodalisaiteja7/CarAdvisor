import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'car_advisor.db'}")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 7 * 24 * 3600))  # 7 days

SCRAPER_DEFAULT_DELAY = float(os.getenv("SCRAPER_DELAY", 2.0))
SCRAPER_MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", 3))

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

VEHICLE_SYSTEMS = [
    "Engine",
    "Transmission",
    "Electrical",
    "Suspension",
    "Brakes",
    "Body/Paint",
    "Interior",
    "HVAC",
    "Steering",
    "Fuel System",
    "Exhaust",
    "Cooling",
]

MILEAGE_BRACKETS = [
    (0, 30_000),
    (30_000, 60_000),
    (60_000, 90_000),
    (90_000, 120_000),
    (120_000, 150_000),
    (150_000, float("inf")),
]

SEVERITY_WEIGHTS = {
    "complaint_count": 0.30,
    "severity": 0.25,
    "safety_impact": 0.25,
    "repair_cost": 0.20,
}

NHTSA_API_BASE = "https://api.nhtsa.gov"

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "CarAdvisor/1.0")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_ENABLED = bool(ANTHROPIC_API_KEY)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 8192))
