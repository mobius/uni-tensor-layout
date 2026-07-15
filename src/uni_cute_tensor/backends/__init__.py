"""Backends: host reference, real VE NLC DGEMM, Phi dgemm/smoke."""

from uni_cute_tensor.backends.host_dgemm import (
    host_avx512_dgemm,
    host_blocked_dgemm,
    host_numpy_dgemm,
)
from uni_cute_tensor.backends.host_ref import host_dgemm_reference, host_sharded_dgemm
from uni_cute_tensor.backends.phi_dgemm import (
    compile_phi_dgemm,
    run_phi_dgemm,
    try_icc_license,
)
from uni_cute_tensor.backends.phi_smoke import find_phi_peak_binary, run_phi_peak_smoke
from uni_cute_tensor.backends.ve_dgemm import (
    compile_ve_dgemm,
    multi_ve_layout_dgemm,
    run_ve_dgemm_shard,
    ve_toolchain_available,
)
from uni_cute_tensor.backends.ve_worker import (
    VeWorkerPool,
    compile_ve_worker,
    multi_ve_layout_dgemm_pooled,
)
from uni_cute_tensor.backends.phi_prep import run_phi_prep_scale

__all__ = [
    "host_dgemm_reference",
    "host_sharded_dgemm",
    "host_numpy_dgemm",
    "host_blocked_dgemm",
    "host_avx512_dgemm",
    "compile_ve_dgemm",
    "run_ve_dgemm_shard",
    "multi_ve_layout_dgemm",
    "ve_toolchain_available",
    "VeWorkerPool",
    "compile_ve_worker",
    "multi_ve_layout_dgemm_pooled",
    "run_phi_prep_scale",
    "run_phi_peak_smoke",
    "find_phi_peak_binary",
    "compile_phi_dgemm",
    "run_phi_dgemm",
    "try_icc_license",
]
