"""#629 items 1+2: exact nested transfers for native refine() pairs, and the
FAC patch/background split that drives patch-restricted smoothing.

The transfer test discriminates by DEGREE-k reproduction: the nested embedding
must reproduce a full degree-k polynomial to machine precision, which the
point-located (linear-exact) builders cannot do at k >= 2. The split test uses
an NVB band hierarchy — bounded closure, so the background genuinely falls
through — and checks classification, the halo, and the end-to-end solve with
the ASM patch smoother live.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _box(dim, cell):
    lo, hi = (0,) * dim, (1,) * dim
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=lo, maxCoords=hi, cellSize=cell, regular=False,
        qdegree=2, refinement=1)


@pytest.mark.parametrize("dim,degree,cell", [(2, 1, 0.2), (2, 2, 0.2),
                                             (3, 2, 0.4)])
def test_nested_refine_pair_is_exact_at_degree(dim, degree, cell):
    mesh = _box(dim, cell)
    L0, L1 = mesh._coarse_level_meshes()[:2]
    assert custom_mg._is_native_refine_pair(L0, L1)
    # the mesh itself is the finest hierarchy level: (L0, mesh) pairs too
    assert custom_mg._is_native_refine_pair(L0, mesh)
    assert not custom_mg._is_native_refine_pair(L1, mesh)   # same slot

    Pn = custom_mg.nested_refine_pair_prolongation(L0, L1, degree, True)
    assert Pn is not None
    c0 = np.asarray(L0._get_coords_for_basis(degree, True))
    c1 = np.asarray(L1._get_coords_for_basis(degree, True))
    assert Pn.shape == (c1.shape[0], c0.shape[0])

    def f(c):
        if degree == 1:
            return c.sum(axis=1) + 0.5
        return (c[:, 0] ** 2 + 2 * c[:, 0] * c[:, 1] - c[:, -1] ** 2
                + c.sum(axis=1))

    assert np.abs(Pn @ f(c0) - f(c1)).max() < 1e-11
    # partition of unity, and the embedding is structurally full rank
    assert np.abs(np.asarray(Pn.sum(axis=1)).ravel() - 1.0).max() < 1e-11
    assert int((np.asarray((Pn != 0).sum(axis=0)).ravel() == 0).sum()) == 0


def test_untagged_pair_declines_to_geometric():
    mesh = _box(2, 0.2)
    L0, L1 = mesh._coarse_level_meshes()[:2]
    plain = uw.discretisation.Mesh(
        L1.dm.clone(), simplex=True,
        coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
        qdegree=2, boundaries=mesh.boundaries)
    assert not custom_mg._is_native_refine_pair(L0, plain)


# --------------------------------------------------------------------------- #
#  FAC split
# --------------------------------------------------------------------------- #
def _nvb_band(dm, m0, band):
    cs, ce = dm.getHeightStratum(0)
    vs, ve = dm.getDepthStratum(0)
    X = dm.getCoordinatesLocal().array.reshape(-1, 2)
    cells = []
    for c in range(cs, ce):
        vv = [q - vs for q in dm.getTransitiveClosure(c)[0] if vs <= q < ve]
        if abs(X[vv, 0].mean() - 0.5) < band:
            cells.append(c)
    carry = [(b.name, b.value) for b in m0.boundaries
             if b.name not in ("Null_Boundary", "All_Boundaries")]
    return custom_mg.nvb_refine(dm, cells, boundaries=carry)


def _wrap(dm, m0):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type,
        qdegree=2, boundaries=m0.boundaries)


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    x, y = mesh.X
    p.f = 8 * sympy.pi**2 * sympy.sin(2 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
    for b in ("Bottom", "Top", "Left", "Right"):
        p.add_dirichlet_bc(0.0, b)
    p.petsc_options["ksp_rtol"] = 1e-8
    return p


def _band_hierarchy():
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=True,
        qdegree=2)
    dm1 = m0.dm.refine()
    dm2 = _nvb_band(dm1, m0, 0.15)
    return m0, [_wrap(m0.dm, m0), _wrap(dm1, m0)], _wrap(dm2, m0)


def test_fac_split_classifies_band_and_declines_uniform():
    m0, coarse, fine = _band_hierarchy()
    s = _poisson(fine)
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric")
    s.solve()
    h = s._custom_mg["hierarchy"]
    pr = h.level_patch_rows
    # level 1 is the uniform refine pair: whole-level smoothing (declined);
    # level 2 is the band: a genuine split with a nonempty halo
    assert pr[1] is None
    assert pr[2] is not None
    owned, sub = pr[2]
    n2 = h.transfers[1].getSize()[0]
    assert 0 < len(owned) < 0.75 * n2
    assert len(owned) < len(sub) <= n2
    # patch DOFs sit in the band, background outside it: check geometrically
    var = s.Unknowns.u
    cf = np.asarray(fine._get_coords_for_basis(var.degree, var.continuous))
    rmap, _ = custom_mg._reduced_map(s.dm, None)
    xs = cf[np.asarray(rmap)[owned]][:, 0]     # nc == 1: full index == node
    assert np.all(np.abs(xs - 0.5) < 0.3)


def test_fac_smoother_is_asm_and_converges():
    m0, coarse, fine = _band_hierarchy()
    s = _poisson(fine)
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric")
    s.solve()
    assert s.snes.getConvergedReason() > 0
    pc = s.snes.getKSP().getPC()
    assert pc.getType() == "mg"
    spc = pc.getMGSmoother(2).getPC()
    assert spc.getType() == "asm"
    sub = spc.getASMSubKSP()
    n_sub = sub[0].getOperators()[0].getSize()[0]
    n_lev = pc.getMGSmoother(2).getOperators()[0].getSize()[0]
    assert n_sub == len(s._custom_mg["hierarchy"].level_patch_rows[2][1])
    assert n_sub < n_lev
    # the patch-smoothed answer is the same answer
    g = _poisson(_wrap(fine.dm, m0))
    g.preconditioner = "gamg"
    g.solve()
    rel = (np.linalg.norm(s.Unknowns.u.data - g.Unknowns.u.data)
           / np.linalg.norm(g.Unknowns.u.data))
    assert rel < 1e-4
