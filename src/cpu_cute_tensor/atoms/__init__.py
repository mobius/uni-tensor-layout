"""Hardware-oriented MMA-style atoms for this machine (not NVIDIA Tensor Cores)."""

from __future__ import annotations

from cpu_cute_tensor.atoms.host_avx512 import HOST_ATOMS, HOST_AVX512_8x8x8_F64
from cpu_cute_tensor.atoms.phi_knc import PHI_ATOMS, PHI_KNC_8x8x8_F64
from cpu_cute_tensor.atoms.ve import VE_ATOMS, VE_NLC_DGEMM_64x64x64_F64, VE_VECTOR_AXPY_F64

_ALL = {**HOST_ATOMS, **PHI_ATOMS, **VE_ATOMS}


def list_atoms() -> dict[str, object]:
    """Return all registered custom atoms keyed by name."""
    return dict(_ALL)


__all__ = [
    "HOST_AVX512_8x8x8_F64",
    "HOST_ATOMS",
    "PHI_KNC_8x8x8_F64",
    "PHI_ATOMS",
    "VE_NLC_DGEMM_64x64x64_F64",
    "VE_VECTOR_AXPY_F64",
    "VE_ATOMS",
    "list_atoms",
]
