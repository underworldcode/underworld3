"""tensor / matrix operations on meshes"""

import sympy
from sympy import sympify
from typing import Optional, Callable

idxmap = [
    (0, []),
    (0, []),
    (3, [(0, 0), (1, 1), (0, 1)]),
    (6, [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]),
]

_P_mandel_2 = sympy.Matrix.diag([1, 1, sympy.sqrt(2)])
_P_mandel_3 = sympy.Matrix.diag([1, 1, 1, sympy.sqrt(2), sympy.sqrt(2), sympy.sqrt(2)])

_P_voigt_2 = sympy.Matrix.diag([1, 1, 1 / 2])
_P_voigt_3 = sympy.Matrix.diag([1, 1, 1, 1 / 2, 1 / 2, 1 / 2])

P_mandel = [None, None, _P_mandel_2, _P_mandel_3]
P_voigt = [None, None, _P_voigt_2, _P_voigt_3]


def rank2_symmetric_sym(name, dim):
    r"""Rank 2 symmetric tensor (as symbolic matrix), name is a sympy latex expression"""

    d = dim
    T = sympy.Matrix(sympy.symarray(name, (d, d)))

    for i in range(d):
        for j in range(i, d):
            T[j, i] = T[i, j]

    return T


def rank4_symmetric_sym(name, dim):
    r"""Rank 4 symmetric tensor (as symbolic matrix), name is a sympy latex expression"""

    d = dim
    dd = idxmap[d][0]

    T = sympy.MutableDenseNDimArray(sympy.symarray(name, (d, d, d, d)))
    TT = rank4_to_mandel(T, dim)

    for I in range(dd):
        for J in range(I, dd):
            TT[J, I] = TT[I, J]

    return mandel_to_rank4(TT, dim)


def tensor_rotation(R, T):
    r"""Rotate tensor of any rank using matrix R"""

    rank = T.rank()

    for i in range(rank):
        T = sympy.tensorcontraction(sympy.tensorproduct(R, T), (1, rank + 1))

    return T


def _rank2_to_unscaled_matrix(v_ij, dim, covariant=True):
    r"""Convert rank 2 tensor (v_ij) to voigt (vector) form (V_I)"""

    imapping = idxmap

    vdim = imapping[dim][0]

    V_I = sympy.Matrix.zeros(1, vdim)

    for I in range(imapping[dim][0]):
        indices = imapping[dim][1]
        i, j = indices[I]
        V_I[I] = v_ij[i, j]

    return V_I


def _rank4_to_unscaled_matrix(c_ijkl, dim):
    r"""Convert rank 4 tensor (c_ijkl) to matrix form (C_IJ)"""

    imapping = idxmap

    vdim = imapping[dim][0]

    C_IJ = sympy.Matrix.zeros(vdim, vdim)

    for I in range(vdim):
        indices = imapping[dim][1]
        i, j = indices[I]
        for J in range(vdim):
            k, l = indices[J]
            C_IJ[I, J] = c_ijkl[i, j, k, l]

    return C_IJ


def _unscaled_matrix_to_rank2(V_I, dim):
    r"""Convert to rank 2 tensor (v_ij) from voigt (vector) form (V_I)"""

    # convert Voight form V_I to v_ij (matrix)
    v_ij = sympy.Matrix.zeros(dim, dim)

    imapping = idxmap
    vdim = imapping[dim][0]

    for I in range(imapping[dim][0]):
        indices = imapping[dim][1]
        i, j = indices[I]

    return v_ij


def _unscaled_matrix_to_rank4(C_IJ, dim):
    r"""Convert to rank 4 tensor (c_ijkl) from matrix form (C_IJ)"""

    imapping = idxmap

    vdim = imapping[dim][0]

    c_ijkl = sympy.MutableDenseNDimArray(sympy.symarray("c", (dim, dim, dim, dim)))

    for I in range(imapping[dim][0]):
        indices = imapping[dim][1]
        i, j = indices[I]
        for J in range(imapping[dim][0]):
            k, l = indices[J]
            # C_IJ -> C_ijkl -> C_jilk -> C_ij_lk -> C_jikl (Symmetry)
            c_ijkl[i, j, k, l] = C_IJ[I, J]
            c_ijkl[j, i, k, l] = C_IJ[I, J]
            c_ijkl[i, j, l, k] = C_IJ[I, J]
            c_ijkl[j, i, l, k] = C_IJ[I, J]

    return c_ijkl


def rank2_to_voigt(v_ij, dim, covariant=True):
    r"""Convert rank 2 tensor (v_ij) to voigt (vector) form (V_I)"""

    imapping = idxmap
    vdim = imapping[dim][0]

    V_I = _rank2_to_unscaled_matrix(v_ij, dim)

    if not covariant:
        V_I = V_I * P_voigt[dim].inv()

    return V_I


def voigt_to_rank2(V_I, dim, covariant=True):
    r"""Convert to rank 2 tensor (v_ij) from voigt (vector) form (V_I)"""

    imapping = idxmap
    vdim = imapping[dim][0]

    if not covariant:
        V_I *= P_voigt[dim]

    v_ij = _unscaled_matrix_to_rank2(V_I, dim)

    return v_ij


def rank4_to_voigt(c_ijkl, dim):
    r"""Convert rank 4 tensor (c_ijkl) to voigt (matrix) form (C_IJ)"""

    imapping = idxmap

    vdim = imapping[dim][0]

    C_IJ = _rank4_to_unscaled_matrix(c_ijkl, dim)

    return C_IJ


def voigt_to_rank4(C_IJ, dim):
    r"""Convert to rank 4 tensor (c_ijkl) from voigt (matrix) form (C_IJ)"""

    imapping = idxmap
    vdim = imapping[dim][0]

    return _unscaled_matrix_to_rank4(C_IJ, dim)


def rank2_to_mandel(v_ij, dim):

    P = P_mandel[dim]

    v_I = P * _rank2_to_unscaled_matrix(v_ij, dim, covariant=True).T

    return v_I


def rank4_to_mandel(c_ijkl, dim):

    P = P_mandel[dim]
    c_IJ = P * _rank4_to_unscaled_matrix(c_ijkl, dim) * P

    return c_IJ


def mandel_to_rank2(v_I, dim):

    P = P_mandel[dim]

    return _unscaled_matrix_to_rank2(P.inv() * v_I, dim)


def mandel_to_rank4(c_IJ, dim):

    P = P_mandel[dim]

    return _unscaled_matrix_to_rank4(P.inv() * c_IJ * P.inv(), dim)


def rank4_identity(dim):
    r"""I_ijkl = (\delta_ij \delta_kl + \delta_il\delta_jk)/2"""

    I = sympy.MutableDenseNDimArray.zeros(dim, dim, dim, dim)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    I[i, j, k, l] = (
                        sympy.sympify((i == k) * (j == l) + (i == l) * (j == k)) / 2
                    )

    return I
