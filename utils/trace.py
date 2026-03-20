"""Debug tracing for the analysis pipeline.

Captures every step of a report generation: user query, scraper outputs,
LLM prompts/responses, and section data, for testing and debugging.

Uses thread-local storage so traces are automatically scoped to the
background analysis thread without passing objects through the call stack.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_thread_local = threading.local()

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class DebugTrace:
    """Accumulates debug data throughout a single report generation run."""

    def __init__(self, report_id: str):
        self.report_id = report_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.user_query: dict = {}
        self.scrapers: dict[str, dict] = {}
        self.analysis: dict = {}
        self.llm_calls: list[dict] = []
        self.sections_pre_llm: dict = {}
        self.sections_post_llm: dict = {}

    # -- Logging helpers --------------------------------------------------

    def log_user_query(self, **kwargs):
        self.user_query = kwargs

    def log_scraper(self, name: str, status: str, data=None, error=None):
        summary = None
        if data and isinstance(data, dict):
            summary = {
                "source": data.get("source"),
                "problems_count": len(data.get("problems", [])),
                "recalls_count": len(data.get("recalls", [])),
                "has_ratings": bool(data.get("ratings")),
                "problems_preview": [
                    {
                        "category": p.get("category"),
                        "description": (p.get("description") or "")[:200],
                        "severity": p.get("severity"),
                        "complaint_count": p.get("complaint_count"),
                    }
                    for p in (data.get("problems") or [])[:8]
                ],
                "recalls_preview": [
                    {
                        "component": r.get("component"),
                        "summary": (r.get("summary") or "")[:200],
                    }
                    for r in (data.get("recalls") or [])[:5]
                ],
            }

        self.scrapers[name] = {
            "status": status,
            "summary": summary,
            "raw_data": data,
            "error": str(error) if error else None,
        }

    def log_analysis(self, **kwargs):
        self.analysis = kwargs

    def log_llm_call(
        self,
        purpose: str,
        prompt: str,
        response_raw: str | None,
        response_parsed=None,
        status: str = "success",
    ):
        self.llm_calls.append({
            "call_number": len(self.llm_calls) + 1,
            "purpose": purpose,
            "prompt": prompt,
            "prompt_char_count": len(prompt),
            "response_raw": response_raw,
            "response_char_count": len(response_raw) if response_raw else 0,
            "response_parsed": response_parsed,
            "status": status,
        })

    def log_sections(self, phase: str, sections: dict):
        snapshot = _safe_deepcopy(sections)
        if phase == "pre_llm":
            self.sections_pre_llm = snapshot
        else:
            self.sections_post_llm = snapshot

    # -- Persistence ------------------------------------------------------

    def save(self) -> Path:
        LOGS_DIR.mkdir(exist_ok=True)
        filepath = LOGS_DIR / f"{self.report_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str, ensure_ascii=False)
        logger.info("Debug trace saved to %s", filepath)
        return filepath

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "scrapers": {
                name: {
                    "status": s["status"],
                    "summary": s["summary"],
                    "error": s["error"],
                    "raw_data": s["raw_data"],
                }
                for name, s in self.scrapers.items()
            },
            "analysis": self.analysis,
            "llm_calls": self.llm_calls,
            "sections_pre_llm": self.sections_pre_llm,
            "sections_post_llm": self.sections_post_llm,
        }


# -- Thread-local access -------------------------------------------------


def start_trace(report_id: str) -> DebugTrace:
    trace = DebugTrace(report_id)
    _thread_local.trace = trace
    return trace


def get_trace() -> DebugTrace | None:
    return getattr(_thread_local, "trace", None)


def end_trace() -> DebugTrace | None:
    trace = getattr(_thread_local, "trace", None)
    if trace:
        trace.save()
    _thread_local.trace = None
    return trace


# -- Helpers --------------------------------------------------------------


def _safe_deepcopy(obj):
    """Best-effort deep copy that falls back to the original on failure."""
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj
