"""uni-cute-tensor: CuTe layout algebra for Host / Phi / VE heterogeneous systems."""

from __future__ import annotations

__version__ = "0.7.0"

from uni_cute_tensor.atoms import (
    HOST_AVX512_8x8x8_F64,
    PHI_KNC_8x8x8_F64,
    VE_NLC_DGEMM_64x64x64_F64,
    list_atoms,
)
from uni_cute_tensor.partition.multi_device import (
    PlacementPlan,
    Shard,
    partition_matrix_rows,
    partition_to_devices,
)

__all__ = [
    "__version__",
    "HOST_AVX512_8x8x8_F64",
    "PHI_KNC_8x8x8_F64",
    "VE_NLC_DGEMM_64x64x64_F64",
    "list_atoms",
    "PlacementPlan",
    "Shard",
    "partition_matrix_rows",
    "partition_to_devices",
]
