"""Backends: host reference, real VE NLC DGEMM, Phi smoke."""

from cpu_cute_tensor.backends.host_dgemm import host_blocked_dgemm, host_numpy_dgemm
from cpu_cute_tensor.backends.host_ref import host_dgemm_reference, host_sharded_dgemm
from cpu_cute_tensor.backends.phi_smoke import find_phi_peak_binary, run_phi_peak_smoke
from cpu_cute_tensor.backends.ve_dgemm import (
    compile_ve_dgemm,
    multi_ve_layout_dgemm,
    run_ve_dgemm_shard,
    ve_toolchain_available,
)

__all__ = [
    "host_dgemm_reference",
    "host_sharded_dgemm",
    "host_numpy_dgemm",
    "host_blocked_dgemm",
    "compile_ve_dgemm",
    "run_ve_dgemm_shard",
    "multi_ve_layout_dgemm",
    "ve_toolchain_available",
    "run_phi_peak_smoke",
    "find_phi_peak_binary",
]
