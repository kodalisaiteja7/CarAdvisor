"""CarComplaints.com scraper — overview page plus problem sub-pages."""

import logging
import re

from bs4 import BeautifulSoup, Tag

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

_MILEAGE_PATTERNS = [
    re.compile(r"(\d{1,3}(?:,\d{3})+)\s*(?:miles|mi\b)", re.IGNORECASE),
    re.compile(r"(\d{4,6})\s*(?:miles|mi\b)", re.IGNORECASE),
    re.compile(r"(\d{1,3})[kK]\s*(?:miles|mi\b)?"),
]
_COST_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")


def _extract_mileage(text: str) -> int | None:
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


def _extract_all_mileages(text: str) -> list[int]:
    results = []
    if not text:
        return results
    for pattern in _MILEAGE_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(1).replace(",", "")
            value = int(raw)
            if value < 1000 and "k" in m.group(0).lower():
                value *= 1000
            if 1000 <= value <= 500000:
                results.append(value)
    return results


def _extract_cost(text: str) -> tuple[float | None, float | None]:
    matches = _COST_RE.findall(text)
    if not matches:
        return None, None
    values = [float(v.replace(",", "")) for v in matches]
    values = [v for v in values if 10 <= v <= 50000]
    if not values:
        return None, None
    return min(values), max(values)


CATEGORY_MAP = {
    "engine": "Engine",
    "transmission": "Transmission",
    "electrical": "Electrical",
    "suspension": "Suspension",
    "brakes": "Brakes",
    "body": "Body/Paint",
    "paint": "Body/Paint",
    "interior": "Interior",
    "ac": "HVAC",
    "heating": "HVAC",
    "hvac": "HVAC",
    "air conditioning": "HVAC",
    "steering": "Steering",
    "fuel": "Fuel System",
    "exhaust": "Exhaust",
    "cooling": "Cooling",
    "drivetrain": "Transmission",
    "powertrain": "Transmission",
    "accessories": "Interior",
    "windows": "Body/Paint",
    "windshield": "Body/Paint",
    "seats": "Interior",
    "lights": "Electrical",
    "electronics": "Electrical",
}


def _normalize_category(raw: str) -> str:
    lower = raw.lower().strip()
    for key, system in CATEGORY_MAP.items():
        if key in lower:
            return system
    return raw.title() if raw else "Other"


class CarComplaintsScraper(BaseScraper):
    source_name = "carcomplaints"
    base_url = "https://www.carcomplaints.com"
    delay = 2.5

    def scrape(self, make: str, model: str, year: int) -> dict:
        result = self._empty_result(self.source_name, make, model, year)

        overview = self._scrape_overview(make, model, year)
        if overview is None:
            return result

        result["problems"] = overview.get("problems", [])
        result["ratings"] = overview.get("ratings", {})
        return result

    def _make_url(self, make: str, model: str, year: int) -> str:
        make_slug = make.title().replace(" ", "_")
        model_slug = model.title().replace(" ", "_")
        return f"{self.base_url}/{make_slug}/{model_slug}/{year}/"

    def _scrape_overview(self, make: str, model: str, year: int) -> dict | None:
        url = self._make_url(make, model, year)

        try:
            resp = self._get(url)
        except PermissionError:
            logger.info("[carcomplaints] Blocked by robots.txt")
            return None
        except Exception:
            logger.exception("[carcomplaints] Failed to fetch overview page")
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        data: dict = {"problems": [], "ratings": {}}

        self._parse_ratings(soup, data)
        self._parse_problems(soup, data, make, model, year)

        problem_links = self._find_problem_links(soup, make, model, year)
        if problem_links:
            self._scrape_problem_pages(problem_links, data, make, model, year)

        return data

    def _parse_ratings(self, soup: BeautifulSoup, data: dict):
        worst_badge = soup.find(class_=re.compile(r"worst", re.IGNORECASE))
        if worst_badge:
            data["ratings"]["worst_model_year"] = True
            badge_text = worst_badge.get_text(strip=True)
            if badge_text:
                data["ratings"]["badge"] = badge_text

        awesome_badge = soup.find(class_=re.compile(r"awesome|seal", re.IGNORECASE))
        if awesome_badge:
            data["ratings"]["seal_of_awesome"] = True

        overall_el = soup.find(class_=re.compile(r"overall|rating", re.IGNORECASE))
        if overall_el:
            text = overall_el.get_text(strip=True)
            data["ratings"]["overall_text"] = text

    def _find_problem_links(
        self, soup: BeautifulSoup, make: str, model: str, year: int
    ) -> list[tuple[str, str]]:
        """Find links to problem-specific sub-pages like /Toyota/Camry/2018/engine/."""
        links = []
        make_slug = make.title().replace(" ", "_")
        model_slug = model.title().replace(" ", "_")
        base_path = f"/{make_slug}/{model_slug}/{year}/".lower()

        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            if href.startswith(base_path) and href != base_path:
                segment = href.replace(base_path, "").strip("/").split("/")[0]
                if segment and len(segment) > 1 and not segment.isdigit():
                    category = _normalize_category(segment)
                    full_url = f"{self.base_url}{a['href']}"
                    if (full_url, category) not in links:
                        links.append((full_url, category))

        return links[:8]

    def _scrape_problem_pages(
        self,
        links: list[tuple[str, str]],
        data: dict,
        make: str,
        model: str,
        year: int,
    ):
        """Scrape individual problem category pages for detailed complaint data."""
        existing_cats = {p["category"] for p in data["problems"]}

        for url, category in links:
            try:
                resp = self._get(url)
            except Exception:
                logger.debug("[carcomplaints] Could not fetch sub-page: %s", url)
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            complaints = self._parse_complaint_entries(soup)

            if not complaints:
                continue

            all_mileages = []
            all_costs_low = []
            all_costs_high = []
            reports = []

            for c in complaints:
                if c["mileage"]:
                    all_mileages.append(c["mileage"])
                if c["cost_low"] is not None:
                    all_costs_low.append(c["cost_low"])
                if c["cost_high"] is not None:
                    all_costs_high.append(c["cost_high"])
                if c["text"]:
                    reports.append(c["text"][:500])

            all_mileages.sort()
            if all_mileages:
                trim = max(1, len(all_mileages) // 10)
                if len(all_mileages) > 4:
                    mileage_low = all_mileages[trim]
                    mileage_high = all_mileages[-(trim + 1)]
                else:
                    mileage_low = all_mileages[0]
                    mileage_high = all_mileages[-1]
            else:
                mileage_low = None
                mileage_high = None

            cost_low = min(all_costs_low) if all_costs_low else None
            cost_high = max(all_costs_high) if all_costs_high else None

            count = len(complaints)
            severity = "high" if count > 50 else "medium" if count > 10 else "low"

            if category in existing_cats:
                for p in data["problems"]:
                    if p["category"] == category:
                        if mileage_low is not None and p.get("typical_mileage_range") is None:
                            p["typical_mileage_range"] = [mileage_low, mileage_high]
                        if cost_low is not None and p.get("estimated_repair_cost") is None:
                            p["estimated_repair_cost"] = f"${cost_low:.0f}-${cost_high:.0f}"
                        p["complaint_count"] = max(p.get("complaint_count", 0), count)
                        p["user_reports"].extend(reports[:5])
                        break
            else:
                data["problems"].append({
                    "category": category,
                    "description": f"{category} problems reported on {year} {make} {model} ({count} complaints)",
                    "typical_mileage_range": (
                        [mileage_low, mileage_high]
                        if mileage_low is not None
                        else None
                    ),
                    "severity": severity,
                    "frequency": "common" if count > 50 else "moderate" if count > 10 else "rare",
                    "estimated_repair_cost": (
                        f"${cost_low:.0f}-${cost_high:.0f}"
                        if cost_low is not None and cost_high is not None
                        else None
                    ),
                    "complaint_count": count,
                    "safety_impact": None,
                    "user_reports": reports[:10],
                })

    def _parse_complaint_entries(self, soup: BeautifulSoup) -> list[dict]:
        """Parse individual complaint entries from a problem sub-page."""
        complaints = []

        entries = soup.find_all(class_=re.compile(
            r"complaint|entry|report|comment", re.IGNORECASE
        ))

        for entry in entries:
            text = entry.get_text(" ", strip=True)
            if len(text) < 30:
                continue

            mileage = _extract_mileage(text)
            cost_low, cost_high = _extract_cost(text)

            complaints.append({
                "text": text[:500],
                "mileage": mileage,
                "cost_low": cost_low,
                "cost_high": cost_high,
            })

        if not complaints:
            full_text = soup.get_text(" ", strip=True)
            mileages = _extract_all_mileages(full_text)
            cost_low, cost_high = _extract_cost(full_text)

            paragraphs = soup.find_all(["p", "div", "li"])
            for p in paragraphs:
                pt = p.get_text(" ", strip=True)
                if len(pt) > 50 and any(
                    kw in pt.lower()
                    for kw in ("problem", "issue", "fail", "broke", "replace", "repair", "dealer")
                ):
                    m = _extract_mileage(pt)
                    cl, ch = _extract_cost(pt)
                    complaints.append({
                        "text": pt[:500],
                        "mileage": m,
                        "cost_low": cl,
                        "cost_high": ch,
                    })

        return complaints

    def _parse_problems(
        self, soup: BeautifulSoup, data: dict, make: str, model: str, year: int
    ):
        problem_containers = soup.find_all(
            class_=re.compile(r"problem|complaint|category", re.IGNORECASE)
        )

        seen_categories: dict[str, dict] = {}

        for container in problem_containers:
            heading = container.find(["h2", "h3", "h4", "a", "strong"])
            if not heading:
                continue

            raw_category = heading.get_text(strip=True)
            if len(raw_category) > 100 or len(raw_category) < 2:
                continue

            category = _normalize_category(raw_category)

            count_el = container.find(
                string=re.compile(r"\d+\s*(complaints?|problems?|reports?)", re.IGNORECASE)
            )
            complaint_count = 0
            if count_el:
                nums = re.findall(r"\d+", count_el)
                if nums:
                    complaint_count = int(nums[0])

            full_text = container.get_text(" ", strip=True)

            mileage = _extract_mileage(full_text)
            cost_low, cost_high = _extract_cost(full_text)

            severity = "low"
            severity_el = container.find(
                class_=re.compile(r"sever|danger|warn|critical", re.IGNORECASE)
            )
            if severity_el:
                sev_text = severity_el.get_text(strip=True).lower()
                if any(w in sev_text for w in ("really awful", "danger", "critical")):
                    severity = "high"
                elif any(w in sev_text for w in ("pretty bad", "warning", "moderate")):
                    severity = "medium"
            elif complaint_count > 100:
                severity = "high"
            elif complaint_count > 20:
                severity = "medium"

            if category in seen_categories:
                existing = seen_categories[category]
                existing["complaint_count"] = max(
                    existing["complaint_count"], complaint_count
                )
                if full_text and len(full_text) > 20:
                    existing["user_reports"].append(full_text[:500])
                continue

            problem = {
                "category": category,
                "description": f"{raw_category} issues reported on {year} {make} {model}",
                "typical_mileage_range": (
                    [max(0, mileage - 15000), mileage + 15000] if mileage else None
                ),
                "severity": severity,
                "frequency": (
                    "common" if complaint_count > 50
                    else "moderate" if complaint_count > 10
                    else "rare"
                ),
                "estimated_repair_cost": (
                    f"${cost_low:.0f}-${cost_high:.0f}"
                    if cost_low is not None and cost_high is not None
                    else None
                ),
                "complaint_count": complaint_count,
                "safety_impact": None,
                "user_reports": (
                    [full_text[:500]] if full_text and len(full_text) > 20 else []
                ),
            }
            seen_categories[category] = problem
            data["problems"].append(problem)

        data["problems"].sort(
            key=lambda p: p.get("complaint_count", 0), reverse=True
        )

        if not data["problems"]:
            self._parse_fallback(soup, data, make, model, year)

    def _parse_fallback(
        self, soup: BeautifulSoup, data: dict, make: str, model: str, year: int
    ):
        links = soup.find_all("a", href=re.compile(
            r"/(engine|transmission|electrical|brakes|steering|suspension|interior|body|exhaust|cooling|fuel|hvac|air)/",
            re.IGNORECASE,
        ))

        for link in links:
            raw = link.get_text(strip=True)
            category = _normalize_category(raw)
            count_match = re.search(r"(\d+)", raw)
            count = int(count_match.group(1)) if count_match else 0

            data["problems"].append({
                "category": category,
                "description": f"{raw} — {year} {make} {model}",
                "typical_mileage_range": None,
                "severity": "medium" if count > 10 else "low",
                "frequency": "moderate" if count > 10 else "rare",
                "estimated_repair_cost": None,
                "complaint_count": count,
                "safety_impact": None,
                "user_reports": [],
            })
