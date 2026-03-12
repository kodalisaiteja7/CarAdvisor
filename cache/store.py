"""Unified state store with Redis (production) and in-memory (dev) backends.

All report state (_reports, _progress, _traces, _vehicle_cache) goes through
this module so the Flask app works identically in both environments.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_backend: _StoreBackend | None = None


class _StoreBackend:
    """Interface for state storage."""

    def get_report(self, report_id: str) -> dict | None:
        raise NotImplementedError

    def set_report(self, report_id: str, report: dict) -> None:
        raise NotImplementedError

    def get_progress(self, report_id: str) -> list[dict]:
        raise NotImplementedError

    def push_progress(self, report_id: str, event: dict) -> None:
        raise NotImplementedError

    def init_progress(self, report_id: str) -> None:
        raise NotImplementedError

    def get_trace(self, report_id: str) -> dict | None:
        raise NotImplementedError

    def set_trace(self, report_id: str, trace: dict) -> None:
        raise NotImplementedError

    def get_cached_report_id(self, cache_key: str) -> str | None:
        raise NotImplementedError

    def set_cached_report_id(self, cache_key: str, report_id: str) -> None:
        raise NotImplementedError

    def health_check(self) -> bool:
        raise NotImplementedError


class _MemoryBackend(_StoreBackend):
    """In-memory fallback for local development."""

    def __init__(self) -> None:
        self._reports: dict[str, dict] = {}
        self._progress: dict[str, list[dict]] = {}
        self._traces: dict[str, dict] = {}
        self._vehicle_cache: dict[str, str] = {}

    def get_report(self, report_id: str) -> dict | None:
        return self._reports.get(report_id)

    def set_report(self, report_id: str, report: dict) -> None:
        self._reports[report_id] = report

    def get_progress(self, report_id: str) -> list[dict]:
        return self._progress.get(report_id, [])

    def push_progress(self, report_id: str, event: dict) -> None:
        self._progress.setdefault(report_id, []).append(event)

    def init_progress(self, report_id: str) -> None:
        self._progress[report_id] = []

    def get_trace(self, report_id: str) -> dict | None:
        return self._traces.get(report_id)

    def set_trace(self, report_id: str, trace: dict) -> None:
        self._traces[report_id] = trace

    def get_cached_report_id(self, cache_key: str) -> str | None:
        return self._vehicle_cache.get(cache_key)

    def set_cached_report_id(self, cache_key: str, report_id: str) -> None:
        self._vehicle_cache[cache_key] = report_id

    def health_check(self) -> bool:
        return True


class _RedisBackend(_StoreBackend):
    """Redis-backed state for production (multi-worker safe)."""

    _REPORT_TTL = 24 * 3600       # 24 hours
    _PROGRESS_TTL = 3600          # 1 hour
    _TRACE_TTL = 3600             # 1 hour
    _VEHICLE_CACHE_TTL = 7 * 86400  # 7 days

    def __init__(self, redis_url: str) -> None:
        import redis
        self._r = redis.from_url(redis_url, decode_responses=True)
        self._r.ping()
        logger.info("Redis backend connected: %s", redis_url.split("@")[-1])

    def get_report(self, report_id: str) -> dict | None:
        data = self._r.get(f"report:{report_id}")
        return json.loads(data) if data else None

    def set_report(self, report_id: str, report: dict) -> None:
        self._r.setex(f"report:{report_id}", self._REPORT_TTL, json.dumps(report, default=str))

    def get_progress(self, report_id: str) -> list[dict]:
        items = self._r.lrange(f"progress:{report_id}", 0, -1)
        return [json.loads(i) for i in items]

    def push_progress(self, report_id: str, event: dict) -> None:
        key = f"progress:{report_id}"
        self._r.rpush(key, json.dumps(event))
        self._r.expire(key, self._PROGRESS_TTL)

    def init_progress(self, report_id: str) -> None:
        key = f"progress:{report_id}"
        self._r.delete(key)
        self._r.expire(key, self._PROGRESS_TTL)

    def get_trace(self, report_id: str) -> dict | None:
        data = self._r.get(f"trace:{report_id}")
        return json.loads(data) if data else None

    def set_trace(self, report_id: str, trace: dict) -> None:
        self._r.setex(f"trace:{report_id}", self._TRACE_TTL, json.dumps(trace, default=str))

    def get_cached_report_id(self, cache_key: str) -> str | None:
        return self._r.get(f"vcache:{cache_key}")

    def set_cached_report_id(self, cache_key: str, report_id: str) -> None:
        self._r.setex(f"vcache:{cache_key}", self._VEHICLE_CACHE_TTL, report_id)

    def health_check(self) -> bool:
        try:
            return self._r.ping()
        except Exception:
            return False


def init_store() -> None:
    """Initialize the store backend. Call once at app startup."""
    global _backend
    if _backend is not None:
        return

    from config.settings import REDIS_URL
    if REDIS_URL:
        try:
            _backend = _RedisBackend(REDIS_URL)
            return
        except Exception:
            logger.warning("Redis unavailable, falling back to in-memory store", exc_info=True)

    _backend = _MemoryBackend()
    logger.info("Using in-memory store (set REDIS_URL for production)")


def _store() -> _StoreBackend:
    if _backend is None:
        init_store()
    return _backend  # type: ignore[return-value]


def get_report(report_id: str) -> dict | None:
    return _store().get_report(report_id)

def set_report(report_id: str, report: dict) -> None:
    _store().set_report(report_id, report)

def get_progress(report_id: str) -> list[dict]:
    return _store().get_progress(report_id)

def push_progress(report_id: str, event: dict) -> None:
    _store().push_progress(report_id, event)

def init_progress(report_id: str) -> None:
    _store().init_progress(report_id)

def get_trace(report_id: str) -> dict | None:
    return _store().get_trace(report_id)

def set_trace(report_id: str, trace: dict) -> None:
    _store().set_trace(report_id, trace)

def get_cached_report_id(cache_key: str) -> str | None:
    return _store().get_cached_report_id(cache_key)

def set_cached_report_id(cache_key: str, report_id: str) -> None:
    _store().set_cached_report_id(cache_key, report_id)

def health_check() -> bool:
    return _store().health_check()
