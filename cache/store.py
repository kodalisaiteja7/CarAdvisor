"""Unified state store with file-based (production), Redis, and in-memory backends.

All report state (reports, progress, traces, vehicle_cache) goes through
this module so the Flask app works identically in all environments.

The file-based backend is multi-worker safe and used automatically when
RAILWAY_VOLUME_MOUNT_PATH is set, eliminating the need for Redis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

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


class _FileBackend(_StoreBackend):
    """File-based storage for multi-worker production (no Redis needed).

    Uses the Railway volume or any shared filesystem. All workers can
    read/write to the same directory, solving the state-sharing problem.
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir) / "cache_store"
        (self._base / "reports").mkdir(parents=True, exist_ok=True)
        (self._base / "progress").mkdir(parents=True, exist_ok=True)
        (self._base / "traces").mkdir(parents=True, exist_ok=True)
        (self._base / "vcache").mkdir(parents=True, exist_ok=True)
        logger.info("File-based store at %s", self._base)

    def _read_json(self, path: Path) -> dict | list | None:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _write_json(self, path: Path, data) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _safe_name(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get_report(self, report_id: str) -> dict | None:
        return self._read_json(self._base / "reports" / f"{report_id}.json")

    def set_report(self, report_id: str, report: dict) -> None:
        self._write_json(self._base / "reports" / f"{report_id}.json", report)

    def get_progress(self, report_id: str) -> list[dict]:
        data = self._read_json(self._base / "progress" / f"{report_id}.json")
        return data if isinstance(data, list) else []

    def push_progress(self, report_id: str, event: dict) -> None:
        path = self._base / "progress" / f"{report_id}.json"
        events = self.get_progress(report_id)
        events.append(event)
        self._write_json(path, events)

    def init_progress(self, report_id: str) -> None:
        self._write_json(self._base / "progress" / f"{report_id}.json", [])

    def get_trace(self, report_id: str) -> dict | None:
        return self._read_json(self._base / "traces" / f"{report_id}.json")

    def set_trace(self, report_id: str, trace: dict) -> None:
        self._write_json(self._base / "traces" / f"{report_id}.json", trace)

    def get_cached_report_id(self, cache_key: str) -> str | None:
        path = self._base / "vcache" / f"{self._safe_name(cache_key)}.txt"
        try:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        return None

    def set_cached_report_id(self, cache_key: str, report_id: str) -> None:
        path = self._base / "vcache" / f"{self._safe_name(cache_key)}.txt"
        path.write_text(report_id, encoding="utf-8")

    def health_check(self) -> bool:
        return self._base.exists()


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
    """Initialize the store backend. Call once at app startup.

    Priority: Redis > File-based (Railway volume) > In-memory.
    """
    global _backend
    if _backend is not None:
        return

    from config.settings import REDIS_URL
    if REDIS_URL:
        try:
            _backend = _RedisBackend(REDIS_URL)
            return
        except Exception:
            logger.warning("Redis unavailable, trying file backend", exc_info=True)

    vol_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if vol_path and Path(vol_path).is_dir():
        _backend = _FileBackend(vol_path)
        return

    _backend = _MemoryBackend()
    logger.info("Using in-memory store (single-worker only)")


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
