import json
import logging
from datetime import datetime, timezone

from database.models import ScrapedResult, get_session
from config.settings import CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


def get_cached(source: str, make: str, model: str, year: int) -> dict | None:
    """Return cached scraper result if it exists and hasn't expired."""
    session = get_session()
    try:
        result = (
            session.query(ScrapedResult)
            .filter_by(
                source=source,
                make=make.lower(),
                model=model.lower(),
                year=year,
            )
            .order_by(ScrapedResult.scraped_at.desc())
            .first()
        )
        if result is None:
            return None

        scraped_at = result.scraped_at
        if scraped_at.tzinfo is not None:
            scraped_at = scraped_at.replace(tzinfo=None)
        age = (datetime.utcnow() - scraped_at).total_seconds()
        if age > CACHE_TTL_SECONDS:
            logger.info(
                "Cache expired for %s %s %s %d (age: %.0fs)",
                source, make, model, year, age,
            )
            return None

        logger.info("Cache hit for %s %s %s %d", source, make, model, year)
        return json.loads(result.raw_json)
    finally:
        session.close()


def set_cached(source: str, make: str, model: str, year: int, data: dict) -> None:
    """Store a scraper result in the cache."""
    session = get_session()
    try:
        record = ScrapedResult(
            source=source,
            make=make.lower(),
            model=model.lower(),
            year=year,
            raw_json=json.dumps(data),
            scraped_at=datetime.utcnow(),
        )
        session.add(record)
        session.commit()
        logger.info("Cached result for %s %s %s %d", source, make, model, year)
    except Exception:
        session.rollback()
        logger.exception("Failed to cache result for %s %s %s %d", source, make, model, year)
    finally:
        session.close()
