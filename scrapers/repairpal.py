"""RepairPal.com scraper — reliability ratings, common problems, and repair costs."""

import logging
import re

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

_COST_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
_MILEAGE_RE = re.compile(r"(\d{1,3}(?:,\d{3})*)\s*(?:miles|mi\b)", re.IGNORECASE)

CATEGORY_MAP = {
    "engine": "Engine",
    "transmission": "Transmission",
    "electrical": "Electrical",
    "suspension": "Suspension",
    "brakes": "Brakes",
    "body": "Body/Paint",
    "interior": "Interior",
    "ac": "HVAC",
    "hvac": "HVAC",
    "air conditioning": "HVAC",
    "heating": "HVAC",
    "cooling": "Cooling",
    "steering": "Steering",
    "fuel": "Fuel System",
    "exhaust": "Exhaust",
    "drivetrain": "Transmission",
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


class RepairPalScraper(BaseScraper):
    source_name = "repairpal"
    base_url = "https://repairpal.com"
    delay = 3.0
    respect_robots = False

    def scrape(self, make: str, model: str, year: int) -> dict:
        result = self._empty_result(self.source_name, make, model, year)

        reliability = self._scrape_reliability(make, model, year)
        if reliability:
            result["ratings"] = reliability.get("ratings", {})

        problems = self._scrape_problems(make, model, year)
        result["problems"] = problems

        return result

    def _scrape_reliability(self, make: str, model: str, year: int) -> dict | None:
        make_slug = make.lower().replace(" ", "-")
        model_slug = model.lower().replace(" ", "-")
        url = f"{self.base_url}/reliability/{make_slug}/{model_slug}/{year}"

        try:
            resp = self._get(url)
        except Exception:
            logger.debug("[repairpal] Could not fetch reliability page")
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        ratings = {}

        score_el = soup.find(class_=re.compile(r"reliability.*score|score.*reliability|rating", re.IGNORECASE))
        if score_el:
            text = score_el.get_text(strip=True)
            nums = re.findall(r"(\d+\.?\d*)\s*/?\s*5", text)
            if nums:
                ratings["reliability_score"] = float(nums[0])
                ratings["reliability_out_of"] = 5
            else:
                ratings["reliability_text"] = text

        rank_el = soup.find(string=re.compile(r"\d+\s*(st|nd|rd|th)\s*out of", re.IGNORECASE))
        if rank_el:
            ratings["rank_text"] = rank_el.strip()

        cost_el = soup.find(string=re.compile(r"annual.*repair|repair.*cost|maintenance.*cost", re.IGNORECASE))
        if cost_el:
            parent = cost_el.parent if hasattr(cost_el, "parent") else None
            if parent:
                cost_text = parent.get_text(" ", strip=True)
                cost_low, cost_high = _extract_cost(cost_text)
                if cost_low is not None:
                    ratings["annual_repair_cost_low"] = cost_low
                    ratings["annual_repair_cost_high"] = cost_high or cost_low

        frequency_el = soup.find(string=re.compile(r"frequency|visits?\s*per\s*year|times?\s*per\s*year", re.IGNORECASE))
        if frequency_el:
            nums = re.findall(r"(\d+\.?\d*)", frequency_el)
            if nums:
                ratings["repair_frequency_per_year"] = float(nums[0])

        return {"ratings": ratings} if ratings else None

    def _scrape_problems(self, make: str, model: str, year: int) -> list[dict]:
        make_slug = make.lower().replace(" ", "-")
        model_slug = model.lower().replace(" ", "-")
        url = f"{self.base_url}/problems/{make_slug}/{model_slug}/{year}"

        try:
            resp = self._get(url)
        except Exception:
            logger.debug("[repairpal] Could not fetch problems page")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        problems = []

        entries = soup.find_all(class_=re.compile(
            r"problem|common.*issue|repair.*item|listing", re.IGNORECASE
        ))

        for entry in entries:
            heading = entry.find(["h2", "h3", "h4", "a", "strong"])
            if not heading:
                continue

            title = heading.get_text(strip=True)
            if len(title) < 5 or len(title) > 200:
                continue

            full_text = entry.get_text(" ", strip=True)
            category = _normalize_category(title)
            cost_low, cost_high = _extract_cost(full_text)
            mileage = _extract_mileage(full_text)

            severity_text = full_text.lower()
            if any(w in severity_text for w in ("severe", "serious", "critical", "danger")):
                severity = "high"
            elif any(w in severity_text for w in ("moderate", "common", "frequent")):
                severity = "medium"
            else:
                severity = "low"

            frequency_match = re.search(r"(\d+)%", full_text)
            freq_pct = int(frequency_match.group(1)) if frequency_match else None

            problems.append({
                "category": category,
                "description": title,
                "typical_mileage_range": (
                    [max(0, mileage - 15000), mileage + 15000] if mileage else None
                ),
                "severity": severity,
                "frequency": (
                    "common" if (freq_pct and freq_pct > 30)
                    else "moderate" if (freq_pct and freq_pct > 10)
                    else "moderate"
                ),
                "estimated_repair_cost": (
                    f"${cost_low:.0f}-${cost_high:.0f}"
                    if cost_low is not None and cost_high is not None
                    else None
                ),
                "complaint_count": 1,
                "safety_impact": None,
                "user_reports": [full_text[:500]] if len(full_text) > 30 else [],
            })

        if not problems:
            problems = self._parse_fallback(soup, make, model, year)

        return problems

    def _parse_fallback(
        self, soup: BeautifulSoup, make: str, model: str, year: int
    ) -> list[dict]:
        """Try to extract any useful data from less-structured pages."""
        problems = []
        for section in soup.find_all(["article", "section", "div"]):
            text = section.get_text(" ", strip=True)
            if len(text) < 50 or len(text) > 2000:
                continue
            if not any(kw in text.lower() for kw in ("problem", "issue", "repair", "replace", "fail")):
                continue

            cost_low, cost_high = _extract_cost(text)
            mileage = _extract_mileage(text)

            heading = section.find(["h2", "h3", "h4"])
            title = heading.get_text(strip=True) if heading else "Reported issue"
            category = _normalize_category(title)

            problems.append({
                "category": category,
                "description": title[:200],
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
                "user_reports": [],
            })
            if len(problems) >= 10:
                break

        return problems
