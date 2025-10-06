# progress.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import time
import threading

@dataclass
class ProgressRecord:
    task_id: str
    status: str            # "queued" | "running" | "error" | "done"
    percent: float         # 0.0 - 100.0
    message: str           # human readable note
    started_at: float      # epoch seconds
    updated_at: float      # epoch seconds
    result: Optional[Dict[str, Any]] = None  # e.g., {"download_url": "..."}

class ProgressStore:
    """
    Thread-safe in-memory progress store.
    Import and use the singleton `progress_store` from anywhere.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, ProgressRecord] = {}

    def init(self, task_id: str, message: str = "queued") -> None:
        now = time.time()
        rec = ProgressRecord(
            task_id=task_id,
            status="queued",
            percent=0.0,
            message=message,
            started_at=now,
            updated_at=now,
        )
        with self._lock:
            self._data[task_id] = rec

    def update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        percent: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            rec = self._data.get(task_id)
            if not rec:
                # If a worker updates before init() (rare), create a record on the fly
                now = time.time()
                rec = ProgressRecord(
                    task_id=task_id,
                    status=status or "running",
                    percent=float(percent or 0.0),
                    message=message or "",
                    started_at=now,
                    updated_at=now,
                    result=result,
                )
                self._data[task_id] = rec
            else:
                if status is not None:
                    rec.status = status
                if percent is not None:
                    # keep it bounded
                    rec.percent = max(0.0, min(100.0, float(percent)))
                if message is not None:
                    rec.message = message
                if result is not None:
                    rec.result = result
                rec.updated_at = time.time()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._data.get(task_id)
            return asdict(rec) if rec else None

    def delete(self, task_id: str) -> None:
        with self._lock:
            self._data.pop(task_id, None)

    def gc(self, older_than_seconds: int = 24 * 3600) -> int:
        """Optional: prune entries older than a day (or custom)."""
        cutoff = time.time() - older_than_seconds
        removed = 0
        with self._lock:
            for k in list(self._data.keys()):
                if self._data[k].updated_at < cutoff:
                    del self._data[k]
                    removed += 1
        return removed

# Singleton
progress_store = ProgressStore()
