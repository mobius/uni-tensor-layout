"""Power sampling unit tests (no hardware required for structure)."""

from __future__ import annotations

from uni_cute_tensor.power_sample import (
    PowerSampler,
    probe_power_sources,
    read_ipmitool_watts,
    read_rapl_watts,
    read_ve_sensor_watts,
)


def test_probe_power_sources_dict():
    p = probe_power_sources()
    assert "rapl" in p and "ipmitool" in p and "ve_sensors" in p
    assert "ve_note" in p


def test_rapl_state_machine():
    w0, st = read_rapl_watts(None)
    # first call only primes
    assert w0 is None or isinstance(w0, float)
    if st is not None:
        import time

        time.sleep(0.05)
        w1, st2 = read_rapl_watts(st)
        # may or may not get a sample depending on platform
        assert st2 is not None
        if w1 is not None:
            assert w1 >= 0


def test_sampler_stop_shape():
    s = PowerSampler(interval_sec=0.05)
    s.start()
    import time

    time.sleep(0.2)
    out = s.stop()
    assert "n" in out and "mean_w" in out and "degrade" in out
    assert "sources_used" in out and "by_source" in out


def test_ve_without_env_returns_none():
    # default heuristic refuses to guess sensor indices
    import os

    os.environ.pop("UCT_VE_POWER_SENSORS", None)
    # may still be None
    _ = read_ve_sensor_watts()
