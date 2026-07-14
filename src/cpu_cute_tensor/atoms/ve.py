"""NEC Vector Engine 1.0 atoms.

VE high performance for DGEMM comes from NLC BLAS. The atom describes the
*outer* tile used when calling cblas_dgemm, not the internal micro-kernel.
"""

from __future__ import annotations

from tensor_layouts import Layout
from tensor_layouts.atoms import MMAAtom

# 64x64x64 is a convenient planning tile; real NLC accepts general sizes.
VE_NLC_DGEMM_64x64x64_F64 = MMAAtom(
    name="VE_NLC_DGEMM_64x64x64_F64",
    ptx="nlc_cblas_dgemm",
    shape_mnk=(64, 64, 64),
    thr_id=Layout(1, 1),  # whole OpenMP team / NLC handles threads
    a_layout=Layout((1, 64 * 64), (0, 1)),
    b_layout=Layout((1, 64 * 64), (0, 1)),
    c_layout=Layout((1, 64 * 64), (0, 1)),
)

# 1-D vector style "atom" for axpy / SpMV value streams (shape as M x 1 x 1).
VE_VECTOR_AXPY_F64 = MMAAtom(
    name="VE_VECTOR_AXPY_F64",
    ptx="ncc_vector_axpy",
    shape_mnk=(256, 1, 1),
    thr_id=Layout(1, 1),
    a_layout=Layout((1, 256), (0, 1)),
    b_layout=Layout((1, 1), (0, 0)),
    c_layout=Layout((1, 256), (0, 1)),
)

VE_ATOMS = {
    VE_NLC_DGEMM_64x64x64_F64.name: VE_NLC_DGEMM_64x64x64_F64,
    VE_VECTOR_AXPY_F64.name: VE_VECTOR_AXPY_F64,
}
