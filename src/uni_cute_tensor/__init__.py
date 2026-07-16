"""uni-cute-tensor: CuTe layout algebra for Host / Phi / VE heterogeneous systems.

Public API (v1.0 stable) — see ``docs/architecture/*_api_v1.md``.
"""

from __future__ import annotations

__version__ = "1.5.0"

from uni_cute_tensor.atoms import (
    HOST_AVX512_8x8x8_F64,
    PHI_KNC_8x8x8_F64,
    VE_NLC_DGEMM_64x64x64_F64,
    list_atoms,
)
from uni_cute_tensor.backends.host_dgemm import host_dgemm
from uni_cute_tensor.backends.host_ref import host_dgemm_reference, host_sharded_dgemm
from uni_cute_tensor.partition.cost_model import choose_best_placement, estimate_gemm_placement
from uni_cute_tensor.partition.dispatch import DispatchRecommendation, recommend_backend
from uni_cute_tensor.partition.multi_device import (
    PlacementPlan,
    Shard,
    partition_matrix_cols,
    partition_matrix_k,
    partition_matrix_rows,
    partition_to_devices,
)
from uni_cute_tensor.partition.runner import execute_auto, execute_plan
from uni_cute_tensor.power import PowerCap
from uni_cute_tensor.runtime.dataplane import GemmRequest, GemmResult, create_dataplane
from uni_cute_tensor.runtime.job_runner import JobResult, run_job
from uni_cute_tensor.runtime.timeline import Timeline, get_timeline, set_timeline, timeline_scope

__all__ = [
    "__version__",
    # atoms
    "HOST_AVX512_8x8x8_F64",
    "PHI_KNC_8x8x8_F64",
    "VE_NLC_DGEMM_64x64x64_F64",
    "list_atoms",
    # layout / placement
    "PlacementPlan",
    "Shard",
    "partition_matrix_rows",
    "partition_matrix_cols",
    "partition_matrix_k",
    "partition_to_devices",
    "choose_best_placement",
    "estimate_gemm_placement",
    "execute_plan",
    "execute_auto",
    "recommend_backend",
    "DispatchRecommendation",
    # host gemm
    "host_dgemm",
    "host_dgemm_reference",
    "host_sharded_dgemm",
    # power / runtime
    "PowerCap",
    "create_dataplane",
    "GemmRequest",
    "GemmResult",
    "run_job",
    "JobResult",
    "Timeline",
    "timeline_scope",
    "get_timeline",
    "set_timeline",
]
