"""Layer-2 SBR adapt-on-top: mesh.adapt(adapter='sbr') returns a refined CHILD.

The new nested adapter (no MMG, on-rank, no redistribute) refines the static base
finest where a metric demands resolution and returns a child mesh that owns a
custom-P geometric-MG hierarchy. Validated here:

  - adapt() returns a refinement child (parent link, finer than the base);
  - a solver built on the child auto-picks-up the mesh-owned custom-P hierarchy
    (pc=mg, no per-solver set_custom_fmg) and matches GAMG;
  - copy_into prolongates parent->child (FE-exact) and restricts child->parent;
  - re-adapt is non-cumulative (the base finest is unchanged);
  - node_budget localises refinement;
  - mesh.remesh is the renamed MMG path and adapt(adapter='mmg') is a deprecated
    shim that warns.

SBR uses only PETSc's refine_sbr transform + scipy custom-P, so (unlike the MMG
remesh tests) these do NOT require an mmg/pragmatic PETSc build.
"""
import warnings
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _ev(fn, coords):
    return np.asarray(uw.function.evaluate(fn, np.asarray(coords))).reshape(-1)


def _ncell(mesh):
    cs, ce = mesh.dm.getHeightStratum(0)
    return ce - cs


def _base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.3, regular=True,
        refinement=2, qdegree=3)


def _metric(base, center, h_coarse=0.0625, h_fine=0.025, width=0.15):
    """M = 1/h^2 with a fine band around x=center (anchored at the base size)."""
    M = uw.discretisation.MeshVariable(f"M_{int(center*100)}", base, 1, degree=1)
    band = sympy.exp(-(((base.N.x - center) / width) ** 2))
    M.data[:, 0] = _ev(1.0 / (h_coarse + (h_fine - h_coarse) * band) ** 2, M.coords)
    return M


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1
    p.f = 0.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    p.petsc_options["ksp_rtol"] = 1e-8
    p.petsc_options["ksp_type"] = "cg"
    return p


def test_adapt_returns_refinement_child():
    base = _base()
    n0 = _ncell(base)
    child = base.adapt(_metric(base, 0.7), max_levels=1)

    assert child is not base
    assert child.parent is base
    assert child._relationship_kind == "refinement"
    assert _ncell(child) > n0
    # base is untouched (adapt is not in-place)
    assert _ncell(base) == n0


def test_child_solver_auto_picks_up_custom_mg():
    base = _base()
    child = base.adapt(_metric(base, 0.7), max_levels=1)

    s = _poisson(child)
    s.solve()                                   # NO set_custom_fmg
    assert s.snes.getKSP().getPC().getType() == "mg"
    assert s.snes.getConvergedReason() > 0

    g = _poisson(child)
    g.preconditioner = "gamg"
    g.solve()
    rel = np.linalg.norm(s.Unknowns.u.data - g.Unknowns.u.data) / (
        np.linalg.norm(g.Unknowns.u.data) + 1e-30)
    assert rel < 1e-4


def test_copy_into_prolongate_and_restrict():
    base = _base()
    child = base.adapt(_metric(base, 0.7), max_levels=1)
    fn = sympy.sin(2 * sympy.pi * base.N.x) * sympy.cos(sympy.pi * base.N.y)

    Tb = uw.discretisation.MeshVariable("Tb", base, 1, degree=2)
    Tb.data[:, 0] = _ev(fn, Tb.coords)
    Tc = uw.discretisation.MeshVariable("Tc", child, 1, degree=2)

    Tb.copy_into(Tc)                            # prolongate parent -> child
    exact_c = _ev(fn, Tc.coords)
    assert np.linalg.norm(Tc.data[:, 0] - exact_c) / (
        np.linalg.norm(exact_c) + 1e-30) < 1e-2

    Tb2 = uw.discretisation.MeshVariable("Tb2", base, 1, degree=2)
    Tc.copy_into(Tb2)                           # restrict child -> parent (injection)
    exact_b = _ev(fn, Tb2.coords)
    assert np.linalg.norm(Tb2.data[:, 0] - exact_b) / (
        np.linalg.norm(exact_b) + 1e-30) < 1e-2


def test_readapt_is_non_cumulative():
    base = _base()
    n0 = _ncell(base)
    c1 = base.adapt(_metric(base, 0.7), max_levels=1)
    n1 = _ncell(c1)
    c2 = base.adapt(_metric(base, 0.35), max_levels=1)
    # each adapt re-marks from the SAME static base finest
    assert _ncell(base) == n0
    assert _ncell(c2) == n1            # symmetric band -> same count, different place
    assert c2 is not c1


def test_node_budget_localises_refinement():
    base = _base()
    sharp = _metric(base, 0.7, width=0.06)
    full = base.adapt(sharp, max_levels=1)
    budgeted = base.adapt(sharp, max_levels=1, node_budget=40)
    assert _ncell(budgeted) < _ncell(full)


def test_adapt_mmg_shim_warns():
    base = _base()
    H = uw.discretisation.MeshVariable("Hshim", base, 1, degree=1)
    H.data[:, 0] = 1.0 / 0.1**2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            base.adapt(H, adapter="mmg")        # forwards to remesh (MMG)
        except Exception:
            pass                                # MMG backend may be absent
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_adapt_requires_base_hierarchy():
    # a mesh without a refinement hierarchy cannot supply the MG coarse tail
    flat = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.3, regular=True, qdegree=3)
    H = uw.discretisation.MeshVariable("Hflat", flat, 1, degree=1)
    H.data[:, 0] = 1.0 / 0.02**2
    with pytest.raises(RuntimeError):
        flat.adapt(H, max_levels=1)
