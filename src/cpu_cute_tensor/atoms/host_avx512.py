"""Host AVX-512 tile atoms (Cascade Lake / Xeon Gold 6252).

These describe logical (M,N,K) micro-tiles aligned to 512-bit FP64 vectors
(8 doubles per ZMM). They are educational / planning layouts, not vendor ISA
encodings.
"""

from __future__ import annotations

from tensor_layouts import Layout
from tensor_layouts.atoms import MMAAtom

# One OpenMP "thread" holds a full 8x8 C tile; A is 8xK strip, B is Kx8 strip.
# Layouts use column-major element offsets in the atom tile, matching CuTe MMA
# atom convention (T, V) -> matrix offset.

HOST_AVX512_8x8x8_F64 = MMAAtom(
    name="HOST_AVX512_8x8x8_F64",
    ptx="zmm_fma_pd",  # mnemonic label, not real PTX
    shape_mnk=(8, 8, 8),
    thr_id=Layout(1, 1),  # single logical worker for the micro-kernel
    # A (M,K)=(8,8): column-major offsets, thr 0 owns all values
    a_layout=Layout((1, 64), (0, 1)),
    b_layout=Layout((1, 64), (0, 1)),
    c_layout=Layout((1, 64), (0, 1)),
)

HOST_ATOMS = {
    HOST_AVX512_8x8x8_F64.name: HOST_AVX512_8x8x8_F64,
}
