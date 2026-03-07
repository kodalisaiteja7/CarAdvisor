"""Edmunds.com scraper — consumer reviews, common problems, and TCO data."""

import logging
import re

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

_COST_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
_MILEAGE_RE = re.compile(r"(\d{1,3}(?:,\d{3})*)\s*(?:miles|mi\b)", re.IGNORECASE)
_RATING_RE = re.compile(r"(\d+\.?\d*)\s*/?\s*(?:5|10)", re.IGNORECASE)

CATEGORY_MAP = {
    "engine": "Engine",
    "transmission": "Transmission",
    "electrical": "Electrical",
    "suspension": "Suspension",
    "brakes": "Brakes",
    "body": "Body/Paint",
    "paint": "Body/Paint",
    "interior": "Interior",
    "comfort": "Interior",
    "ac": "HVAC",
    "climate": "HVAC",
    "heating": "HVAC",
    "cooling": "Cooling",
    "steering": "Steering",
    "handling": "Steering",
    "fuel": "Fuel System",
    "exhaust": "Exhaust",
    "drivetrain": "Transmission",
    "powertrain": "Transmission",
    "technology": "Electrical",
    "infotainment": "Electrical",
    "safety": "Interior",
    "lights": "Electrical",
}


def _normalize_category(raw: str) -> str:
    lower = raw.lower().strip()
    for key, system in CATEGORY_MAP.items():
        if key in lower:
            return system
    return raw.title() if raw else "Other"


def _extract_cost(text: str) -> tuple[float | None, float | None]:
    matches = _COST_RE.findall(text)
    if not matches:
        return None, None
    values = [float(v.replace(",", "")) for v in matches]
    values = [v for v in values if 10 <= v <= 50000]
    if not values:
        return None, None
    return min(values), max(values)


def _extract_mileage(text: str) -> int | None:
    m = _MILEAGE_RE.search(text)
    if m:
        val = int(m.group(1).replace(",", ""))
        if 1000 <= val <= 500000:
            return val
    return None


class EdmundsScraper(BaseScraper):
    source_name = "edmunds"
    base_url = "https://www.edmunds.com"
    delay = 3.0
    respect_robots = False

    def scrape(self, make: str, model: str, year: int) -> dict:
        result = self._empty_result(self.source_name, make, model, year)

        reviews = self._scrape_reviews(make, model, year)
        result["problems"] = reviews.get("problems", [])
        result["ratings"] = reviews.get("ratings", {})

        tco = self._scrape_tco(make, model, year)
        if tco:
            result["ratings"]["tco"] = tco

        return result

    def _scrape_reviews(self, make: str, model: str, year: int) -> dict:
        make_slug = make.lower().replace(" ", "-")
        model_slug = model.lower().replace(" ", "-")
        url = f"{self.base_url}/{make_slug}/{model_slug}/{year}/review/"

        data = {"problems": [], "ratings": {}}

        try:
            resp = self._get(url)
        except Exception:
            logger.debug("[edmunds] Could not fetch review page")
            return data

        soup = BeautifulSoup(resp.text, "lxml")

        rating_el = soup.find(class_=re.compile(r"rating|score|grade", re.IGNORECASE))
        if rating_el:
            text = rating_el.get_text(strip=True)
            m = _RATING_RE.search(text)
            if m:
                data["ratings"]["edmunds_rating"] = float(m.group(1))

        self._extract_problems_from_reviews(soup, data, make, model, year)

        consumer_url = f"{self.base_url}/{make_slug}/{model_slug}/{year}/consumer-reviews/"
        try:
            resp2 = self._get(consumer_url)
            soup2 = BeautifulSoup(resp2.text, "lxml")
            self._extract_consumer_reviews(soup2, data, make, model, year)
        except Exception:
            logger.debug("[edmunds] Could not fetch consumer reviews page")

        return data

    def _extract_problems_from_reviews(
        self, soup: BeautifulSoup, data: dict, make: str, model: str, year: int
    ):
        """Extract problem mentions from editorial review content."""
        problem_keywords = [
            "problem", "issue", "fail", "recall", "defect", "complaint",
            "unreliable", "broke", "replace", "repair", "malfunction",
            "warning light", "leak", "noise", "vibrat",
        ]

        for section in soup.find_all(["p", "li", "div"]):
            text = section.get_text(" ", strip=True)
            if len(text) < 40 or len(text) > 1000:
                continue

            lower = text.lower()
            matches = sum(1 for kw in problem_keywords if kw in lower)
            if matches < 2:
                continue

            category = "Other"
            for cat_kw, system in CATEGORY_MAP.items():
                if cat_kw in lower:
                    category = system
                    break

            mileage = _extract_mileage(text)
            cost_low, cost_high = _extract_cost(text)

            data["problems"].append({
                "category": category,
                "description": text[:200],
                "typical_mileage_range": (
                    [max(0, mileage - 15000), mileage + 15000] if mileage else None
                ),
                "severity": "medium",
                "frequency": "moderate",
                "estimated_repair_cost": (
                    f"${cost_low:.0f}-${cost_high:.0f}"
                    if cost_low is not None and cost_high is not None
                    else None
                ),
                "complaint_count": 1,
                "safety_impact": None,
                "user_reports": [text[:500]],
            })

    def _extract_consumer_reviews(
        self, soup: BeautifulSoup, data: dict, make: str, model: str, year: int
    ):
        """Extract problems from consumer review entries."""
        review_containers = soup.find_all(class_=re.compile(
            r"review|comment|entry", re.IGNORECASE
        ))

        negative_keywords = [
            "problem", "issue", "broke", "fail", "disappoint", "regret",
            "repair", "replace", "dealer", "warranty", "defect", "recall",
            "noise", "leak", "vibrat", "stall", "hesitat",
        ]

        for container in review_containers:
            text = container.get_text(" ", strip=True)
            if len(text) < 50:
                continue

            lower = text.lower()
            neg_count = sum(1 for kw in negative_keywords if kw in lower)
            if neg_count < 2:
                continue

            category = "Other"
            for cat_kw, system in CATEGORY_MAP.items():
                if cat_kw in lower:
                    category = system
                    break

            mileage = _extract_mileage(text)
            cost_low, cost_high = _extract_cost(text)

            data["problems"].append({
                "category": category,
                "description": f"Consumer-reported {category.lower()} issue on {year} {make} {model}",
                "typical_mileage_range": (
                    [max(0, mileage - 15000), mileage + 15000] if mileage else None
                ),
                "severity": "medium" if neg_count >= 3 else "low",
                "frequency": "moderate",
                "estimated_repair_cost": (
                    f"${cost_low:.0f}-${cost_high:.0f}"
                    if cost_low is not None and cost_high is not None
                    else None
                ),
                "complaint_count": 1,
                "safety_impact": None,
                "user_reports": [text[:500]],
            })

    def _scrape_tco(self, make: str, model: str, year: int) -> dict | None:
        """Scrape True Cost to Own data."""
        make_slug = make.lower().replace(" ", "-")
        model_slug = model.lower().replace(" ", "-")
        url = f"{self.base_url}/{make_slug}/{model_slug}/{year}/cost-to-own/"

        try:
            resp = self._get(url)
        except Exception:
            logger.debug("[edmunds] Could not fetch TCO page")
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        tco = {}

        cost_sections = soup.find_all(string=re.compile(
            r"(depreciation|insurance|maintenance|repairs|fuel|fees)", re.IGNORECASE
        ))
        for el in cost_sections:
            parent = el.parent if hasattr(el, "parent") else None
            if not parent:
                continue
            row_text = parent.get_text(" ", strip=True) if parent else ""
            costs = _COST_RE.findall(row_text)
            if costs:
                label = el.strip().lower().replace(" ", "_")
                tco[label] = float(costs[0].replace(",", ""))

        total_el = soup.find(string=re.compile(r"total|true cost to own", re.IGNORECASE))
        if total_el:
            parent = total_el.parent if hasattr(total_el, "parent") else None
            if parent:
                text = parent.get_text(" ", strip=True)
                costs = _COST_RE.findall(text)
                if costs:
                    tco["total_5yr"] = float(costs[-1].replace(",", ""))

        return tco if tco else None
