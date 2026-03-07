import logging
import re

from config.settings import NHTSA_API_BASE
from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

COMPONENT_MAP = {
    "ENGINE": "Engine",
    "ENGINE AND ENGINE COOLING": "Engine",
    "ENGINE COOLING": "Cooling",
    "POWER TRAIN": "Transmission",
    "POWERTRAIN": "Transmission",
    "AUTOMATIC TRANSMISSION": "Transmission",
    "MANUAL TRANSMISSION": "Transmission",
    "ELECTRICAL SYSTEM": "Electrical",
    "ELECTRONIC STABILITY CONTROL": "Electrical",
    "LIGHTS": "Electrical",
    "SUSPENSION": "Suspension",
    "STEERING": "Steering",
    "SERVICE BRAKES": "Brakes",
    "SERVICE BRAKES, HYDRAULIC": "Brakes",
    "SERVICE BRAKES, AIR": "Brakes",
    "PARKING BRAKE": "Brakes",
    "AIR BAGS": "Interior",
    "SEAT BELTS": "Interior",
    "SEATS": "Interior",
    "INTERIOR LIGHTING": "Interior",
    "EXTERIOR LIGHTING": "Electrical",
    "FUEL SYSTEM, GASOLINE": "Fuel System",
    "FUEL SYSTEM, DIESEL": "Fuel System",
    "FUEL SYSTEM": "Fuel System",
    "EXHAUST SYSTEM": "Exhaust",
    "BODY": "Body/Paint",
    "STRUCTURE": "Body/Paint",
    "VISIBILITY": "Body/Paint",
    "TIRES": "Suspension",
    "WHEELS": "Suspension",
    "AIR CONDITIONING": "HVAC",
    "HYBRID PROPULSION SYSTEM": "Engine",
    "VEHICLE SPEED CONTROL": "Engine",
    "FORWARD COLLISION AVOIDANCE": "Electrical",
    "BACK OVER PREVENTION": "Electrical",
    "LANE DEPARTURE": "Electrical",
    "LATCHES/LOCKS/LINKAGES": "Body/Paint",
    "EQUIPMENT": "Interior",
}

_MILEAGE_PATTERNS = [
    re.compile(r"(\d{1,3}(?:,\d{3})+)\s*(?:miles|mi\b)", re.IGNORECASE),
    re.compile(r"(\d{4,6})\s*(?:miles|mi\b)", re.IGNORECASE),
    re.compile(r"(\d{1,3})[kK]\s*(?:miles|mi\b)?", re.IGNORECASE),
]


def _extract_mileage(text: str) -> int | None:
    """Pull the first mileage mention out of free-text."""
    if not text:
        return None
    for pattern in _MILEAGE_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group(1).replace(",", "")
            value = int(raw)
            if value < 1000 and "k" in m.group(0).lower():
                value *= 1000
            if 1000 <= value <= 500000:
                return value
    return None


def _map_component(raw: str) -> str:
    if not raw:
        return "Other"
    upper = raw.upper().strip()
    for prefix, system in COMPONENT_MAP.items():
        if upper.startswith(prefix):
            return system
    return "Other"


class NHTSAScraper(BaseScraper):
    source_name = "nhtsa"
    base_url = NHTSA_API_BASE
    delay = 0.5
    respect_robots = False  # public government API

    def scrape(self, make: str, model: str, year: int) -> dict:
        result = self._empty_result(self.source_name, make, model, year)

        recalls = self._fetch_recalls(make, model, year)
        result["recalls"] = recalls

        complaints = self._fetch_complaints(make, model, year)
        result["problems"] = complaints

        ratings = self._fetch_ratings(make, model, year)
        result["ratings"] = ratings

        return result

    # ------------------------------------------------------------------
    # Recalls
    # ------------------------------------------------------------------

    def _fetch_recalls(self, make: str, model: str, year: int) -> list[dict]:
        url = (
            f"{self.base_url}/recalls/recallsByVehicle"
            f"?make={make}&model={model}&modelYear={year}"
        )
        try:
            data = self._get_json(url)
        except Exception:
            logger.exception("[nhtsa] Failed to fetch recalls")
            return []

        recalls = []
        for r in data.get("results", []):
            recalls.append({
                "campaign_number": r.get("NHTSACampaignNumber", ""),
                "component": r.get("Component", ""),
                "summary": r.get("Summary", ""),
                "consequence": r.get("Consequence", ""),
                "remedy": r.get("Remedy", ""),
                "report_date": r.get("ReportReceivedDate", ""),
            })
        return recalls

    # ------------------------------------------------------------------
    # Complaints
    # ------------------------------------------------------------------

    def _fetch_complaints(self, make: str, model: str, year: int) -> list[dict]:
        url = (
            f"{self.base_url}/complaints/complaintsByVehicle"
            f"?make={make}&model={model}&modelYear={year}"
        )
        try:
            data = self._get_json(url)
        except Exception:
            logger.exception("[nhtsa] Failed to fetch complaints")
            return []

        component_groups: dict[str, list[dict]] = {}
        for c in data.get("results", []):
            raw_component = c.get("components", "")
            category = _map_component(raw_component)
            summary = c.get("summary", "")
            crash = c.get("crash", "N") == "Y"
            fire = c.get("fire", "N") == "Y"

            mileage = None
            raw_miles = c.get("odiNumber") or c.get("mileage")
            if isinstance(raw_miles, (int, float)) and 1000 <= raw_miles <= 500000:
                mileage = int(raw_miles)
            if mileage is None:
                mileage = _extract_mileage(summary)

            if category not in component_groups:
                component_groups[category] = []
            component_groups[category].append({
                "summary": summary,
                "mileage": mileage,
                "crash": crash,
                "fire": fire,
                "date": c.get("dateComplaintFiled", ""),
                "raw_component": raw_component,
            })

        problems = []
        for category, items in component_groups.items():
            mileages = sorted([i["mileage"] for i in items if i["mileage"]])

            if mileages:
                trim = max(1, len(mileages) // 10)
                trimmed = mileages[trim:-trim] if len(mileages) > 4 else mileages
                mileage_low = trimmed[0] if trimmed else mileages[0]
                mileage_high = trimmed[-1] if trimmed else mileages[-1]
                median_mileage = mileages[len(mileages) // 2]
            else:
                mileage_low = None
                mileage_high = None
                median_mileage = None

            crash_count = sum(1 for i in items if i["crash"])
            fire_count = sum(1 for i in items if i["fire"])
            safety_score = min(10, (crash_count * 3 + fire_count * 5))

            severity = "low"
            if len(items) > 20 or safety_score >= 5:
                severity = "high"
            elif len(items) > 5:
                severity = "medium"

            top_reports = [i["summary"] for i in items[:5] if i["summary"]]
            sub_components = {}
            for i in items:
                rc = i.get("raw_component", "").strip()
                if rc:
                    sub_components[rc] = sub_components.get(rc, 0) + 1
            top_sub = sorted(sub_components.items(), key=lambda x: x[1], reverse=True)

            desc_parts = []
            if top_sub:
                sub_names = [f"{name} ({cnt})" for name, cnt in top_sub[:3]]
                desc_parts.append(f"Affected components: {', '.join(sub_names)}")
            if crash_count:
                desc_parts.append(f"{crash_count} crash report(s)")
            if fire_count:
                desc_parts.append(f"{fire_count} fire report(s)")
            desc_parts.append(f"{len(items)} total NHTSA complaint(s) filed")
            if median_mileage:
                desc_parts.append(f"median failure at {median_mileage:,} miles")

            problems.append({
                "category": category,
                "description": ". ".join(desc_parts),
                "typical_mileage_range": (
                    [mileage_low, mileage_high]
                    if mileage_low is not None
                    else None
                ),
                "severity": severity,
                "frequency": (
                    "common" if len(items) > 20
                    else "moderate" if len(items) > 5
                    else "rare"
                ),
                "estimated_repair_cost": None,
                "complaint_count": len(items),
                "safety_impact": safety_score,
                "user_reports": top_reports,
            })

        problems.sort(key=lambda p: p["complaint_count"], reverse=True)
        return problems

    # ------------------------------------------------------------------
    # Safety ratings
    # ------------------------------------------------------------------

    def _fetch_ratings(self, make: str, model: str, year: int) -> dict:
        url = (
            f"{self.base_url}/SafetyRatings/modelyear/{year}"
            f"/make/{make}/model/{model}"
        )
        try:
            data = self._get_json(url)
        except Exception:
            logger.exception("[nhtsa] Failed to fetch safety ratings")
            return {}

        results = data.get("Results", [])
        if not results:
            return {}

        vehicle_id = results[0].get("VehicleId")
        if not vehicle_id:
            return {"variants": [r.get("VehicleDescription", "") for r in results]}

        detail_url = f"{self.base_url}/SafetyRatings/VehicleId/{vehicle_id}"
        try:
            detail = self._get_json(detail_url)
        except Exception:
            return {}

        detail_results = detail.get("Results", [{}])
        if not detail_results:
            return {}

        r = detail_results[0]
        return {
            "overall": r.get("OverallRating", "N/A"),
            "frontal_crash": r.get("FrontCrashDriversideRating", "N/A"),
            "side_crash": r.get("SideCrashDriversideRating", "N/A"),
            "rollover": r.get("RolloverRating", "N/A"),
            "complaints_count": r.get("ComplaintsCount", 0),
            "recalls_count": r.get("RecallsCount", 0),
        }

    # ------------------------------------------------------------------
    # Vehicle lookup helpers (used by the UI for cascading dropdowns)
    # ------------------------------------------------------------------

    def get_makes(self, year: int | None = None) -> list[str]:
        """Return a list of all makes (optionally filtered by year)."""
        url = f"{self.base_url}/SafetyRatings"
        if year:
            url = f"{self.base_url}/SafetyRatings/modelyear/{year}"
        try:
            data = self._get_json(url)
        except Exception:
            return []
        results = data.get("Results", [])
        makes = sorted({r.get("Make", "") for r in results if r.get("Make")})
        return makes

    def get_models(self, make: str, year: int) -> list[str]:
        url = (
            f"{self.base_url}/SafetyRatings/modelyear/{year}/make/{make}"
        )
        try:
            data = self._get_json(url)
        except Exception:
            return []
        results = data.get("Results", [])
        models = sorted({r.get("Model", "") for r in results if r.get("Model")})
        return models

    def get_years(self) -> list[int]:
        """Return available model years from the NHTSA ratings database."""
        url = f"{self.base_url}/SafetyRatings"
        try:
            data = self._get_json(url)
        except Exception:
            return list(range(2000, 2027))
        results = data.get("Results", [])
        years = sorted(
            {int(r["ModelYear"]) for r in results if r.get("ModelYear")},
            reverse=True,
        )
        return years if years else list(range(2000, 2027))
