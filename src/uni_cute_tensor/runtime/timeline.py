"""JSONL timeline events for H2D / kernel / D2H / wait phases."""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class TimelineEvent:
    ts: float
    name: str
    phase: str  # h2d | kernel | d2h | wait | host | other
    device: str = ""
    duration_sec: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    job_id: str = ""


class Timeline:
    def __init__(self, *, job_id: Optional[str] = None):
        self.job_id = job_id or uuid.uuid4().hex[:12]
        self._events: list[TimelineEvent] = []
        self._lock = threading.Lock()
        self.t0 = time.time()

    def emit(
        self,
        name: str,
        phase: str,
        *,
        duration_sec: float = 0.0,
        device: str = "",
        **meta: Any,
    ) -> None:
        ev = TimelineEvent(
            ts=time.time() - self.t0,
            name=name,
            phase=phase,
            device=device,
            duration_sec=duration_sec,
            meta=meta,
            job_id=self.job_id,
        )
        with self._lock:
            self._events.append(ev)

    @contextmanager
    def span(self, name: str, phase: str, *, device: str = "", **meta: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.emit(
                name,
                phase,
                duration_sec=time.perf_counter() - t0,
                device=device,
                **meta,
            )

    @property
    def events(self) -> list[TimelineEvent]:
        with self._lock:
            return list(self._events)

    def to_jsonl(self) -> str:
        lines = []
        for e in self.events:
            d = asdict(e)
            lines.append(json.dumps(d, ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")

    def write_jsonl(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl(), encoding="utf-8")

    def summary(self) -> dict[str, float]:
        by_phase: dict[str, float] = {}
        for e in self.events:
            by_phase[e.phase] = by_phase.get(e.phase, 0.0) + e.duration_sec
        return by_phase


_tls = threading.local()


def set_timeline(tl: Optional[Timeline]) -> None:
    _tls.timeline = tl


def get_timeline() -> Optional[Timeline]:
    return getattr(_tls, "timeline", None)


@contextmanager
def timeline_scope(job_id: Optional[str] = None) -> Iterator[Timeline]:
    tl = Timeline(job_id=job_id)
    prev = get_timeline()
    set_timeline(tl)
    try:
        yield tl
    finally:
        set_timeline(prev)
