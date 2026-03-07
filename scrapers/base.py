import logging
import time
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from fake_useragent import UserAgent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import SCRAPER_DEFAULT_DELAY, SCRAPER_MAX_RETRIES
from database.cache import get_cached, set_cached

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base class for all data-source scrapers.

    Provides rate limiting, retry logic, User-Agent rotation, caching,
    and robots.txt checking.  Subclasses implement ``scrape`` to return
    data in the project's standardized JSON schema.
    """

    source_name: str = "base"
    base_url: str = ""
    delay: float = SCRAPER_DEFAULT_DELAY
    respect_robots: bool = True

    def __init__(self):
        self.session = requests.Session()
        self._ua = UserAgent(fallback="Mozilla/5.0")
        self._last_request_time: float = 0
        self._robots_parsers: dict[str, RobotFileParser] = {}
        self._rotate_ua()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, make: str, model: str, year: int) -> dict:
        """Return data for a vehicle, using cache when possible."""
        cached = get_cached(self.source_name, make, model, year)
        if cached is not None:
            return cached

        logger.info(
            "[%s] Scraping %s %s %d", self.source_name, make, model, year
        )
        data = self.scrape(make, model, year)
        set_cached(self.source_name, make, model, year, data)
        return data

    @abstractmethod
    def scrape(self, make: str, model: str, year: int) -> dict:
        """Scrape the source and return standardized data.

        Must return a dict matching the project schema::

            {
                "source": "<source_name>",
                "make": "...",
                "model": "...",
                "year": 2018,
                "problems": [...],
                "recalls": [...],
                "ratings": {}
            }
        """

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(SCRAPER_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(self, url: str, **kwargs) -> requests.Response:
        """GET with rate-limiting, UA rotation, retries, and robots check."""
        self._respect_rate_limit()
        self._rotate_ua()

        if self.respect_robots and not self._robots_allowed(url):
            logger.warning("[%s] Blocked by robots.txt: %s", self.source_name, url)
            raise PermissionError(f"robots.txt disallows: {url}")

        logger.debug("[%s] GET %s", self.source_name, url)
        response = self.session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(SCRAPER_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get_json(self, url: str, **kwargs) -> dict:
        """GET and parse JSON response."""
        resp = self._get(url, **kwargs)
        return resp.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _respect_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            sleep_for = self.delay - elapsed
            logger.debug("[%s] Rate-limit sleep %.2fs", self.source_name, sleep_for)
            time.sleep(sleep_for)
        self._last_request_time = time.time()

    def _rotate_ua(self):
        self.session.headers["User-Agent"] = self._ua.random

    def _robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots_parsers:
            rp = RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                logger.debug("[%s] Could not fetch robots.txt for %s", self.source_name, origin)
                return True
            self._robots_parsers[origin] = rp
        return self._robots_parsers[origin].can_fetch("*", url)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(source: str, make: str, model: str, year: int) -> dict:
        return {
            "source": source,
            "make": make,
            "model": model,
            "year": year,
            "problems": [],
            "recalls": [],
            "ratings": {},
        }
