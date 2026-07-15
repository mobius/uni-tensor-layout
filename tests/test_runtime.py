"""Unit tests for runtime DataPlane + Timeline (no accelerator required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uni_cute_tensor.runtime.dataplane import GemmRequest, create_dataplane
from uni_cute_tensor.runtime.timeline import Timeline, timeline_scope


def test_timeline_span_and_jsonl(tmp_path: Path):
    with timeline_scope(job_id="t1") as tl:
        with tl.span("work", "kernel", device="host"):
            pass
        tl.emit("mark", "other", duration_sec=0.001, device="host", note="x")
    assert len(tl.events) == 2
    assert tl.events[0].phase == "kernel"
    assert tl.events[0].duration_sec >= 0.0
    summary = tl.summary()
    assert "kernel" in summary
    out = tmp_path / "tl.jsonl"
    tl.write_jsonl(out)
    text = out.read_text(encoding="utf-8")
    assert "kernel" in text
    assert "t1" in text


def test_host_dataplane_gemm():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((48, 40))
    b = rng.standard_normal((40, 32))
    with create_dataplane("host") as plane:
        r = plane.gemm(GemmRequest(a=a, b=b, name="host-smoke"))
    assert r.status == "pass"
    assert r.backend == "host"
    assert r.max_abs_err < 1e-9
    assert r.c.shape == (48, 32)


def test_create_dataplane_unknown():
    with pytest.raises(ValueError, match="unknown"):
        create_dataplane("nope")  # type: ignore[arg-type]


def test_timeline_thread_local_restore():
    outer = Timeline(job_id="outer")
    from uni_cute_tensor.runtime.timeline import get_timeline, set_timeline

    set_timeline(outer)
    with timeline_scope(job_id="inner") as inner:
        assert get_timeline() is inner
    assert get_timeline() is outer
    set_timeline(None)
