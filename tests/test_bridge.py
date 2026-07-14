"""Bridge / discovery tests (hardware optional)."""

from uni_cute_tensor.bridge.uni_adapter import (
    discover_devices,
    plan_to_task_specs,
    resolve_uni_root,
    ve_device_names,
)
from uni_cute_tensor.partition.multi_device import partition_matrix_rows


def test_discover_includes_host():
    devs = discover_devices(prefer_uni=False)
    assert any(d.name == "host" and d.kind == "host" for d in devs)


def test_plan_to_task_specs_parallel():
    plan = partition_matrix_rows(64, 32, ["ve1", "ve2"])
    specs = plan_to_task_specs(plan)
    assert len(specs) == 2
    assert all(s["depends_on"] == ["prepare"] for s in specs)


def test_resolve_uni_root_type():
    root = resolve_uni_root()
    # May be None in CI without uni checkout
    assert root is None or root.is_dir()


def test_ve_names_filter():
    devs = discover_devices(prefer_uni=False)
    names = ve_device_names(devs, online_only=False)
    assert all(n.startswith("ve") for n in names)
