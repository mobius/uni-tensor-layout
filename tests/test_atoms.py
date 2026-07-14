"""Custom atoms smoke tests."""

from tensor_layouts import size

from uni_cute_tensor.atoms import (
    HOST_AVX512_8x8x8_F64,
    PHI_KNC_8x8x8_F64,
    VE_NLC_DGEMM_64x64x64_F64,
    list_atoms,
)


def test_host_atom_shape():
    assert HOST_AVX512_8x8x8_F64.shape_mnk == (8, 8, 8)
    assert "AVX512" in HOST_AVX512_8x8x8_F64.name


def test_phi_atom_shape():
    assert PHI_KNC_8x8x8_F64.shape_mnk == (8, 8, 8)


def test_ve_atom_shape():
    m, n, k = VE_NLC_DGEMM_64x64x64_F64.shape_mnk
    assert (m, n, k) == (64, 64, 64)
    assert size(VE_NLC_DGEMM_64x64x64_F64.c_layout) == m * n


def test_list_atoms():
    atoms = list_atoms()
    assert HOST_AVX512_8x8x8_F64.name in atoms
    assert PHI_KNC_8x8x8_F64.name in atoms
    assert VE_NLC_DGEMM_64x64x64_F64.name in atoms
