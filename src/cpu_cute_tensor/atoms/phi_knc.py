"""Xeon Phi 7120P (KNC / IMCI) tile atoms.

KNC has 512-bit vector units (8x FP64 FMA per instruction family). Atoms here
mirror the Host AVX-512 tile shape for portable planning; real kernels use
ICC -mmic / IMCI intrinsics via uni-framework backends.
"""

from __future__ import annotations

from tensor_layouts import Layout
from tensor_layouts.atoms import MMAAtom

PHI_KNC_8x8x8_F64 = MMAAtom(
    name="PHI_KNC_8x8x8_F64",
    ptx="imci_fma_pd",
    shape_mnk=(8, 8, 8),
    thr_id=Layout(1, 1),
    a_layout=Layout((1, 64), (0, 1)),
    b_layout=Layout((1, 64), (0, 1)),
    c_layout=Layout((1, 64), (0, 1)),
)

PHI_ATOMS = {
    PHI_KNC_8x8x8_F64.name: PHI_KNC_8x8x8_F64,
}
