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
from underworld3.function import analytic as A

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


def test_stokes_velocity_block_auto_picks_up_fmg():
    """Canonical case: SolCx Stokes (eta jump 1e6) on an adapt() child, refined
    near the viscosity jump. The velocity block must auto-pick-up the mesh-owned
    custom-P FMG (field_id=0, fgmres outer, full cycle) with NO set_custom_fmg,
    converge, and match a GAMG reference. The refinement must be genuinely local."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=True,
        refinement=2, qdegree=3)
    n_uniform = _ncell(base) * 4          # one global SBR level = x4

    # metric: refine only near the x=0.5 jump (h_coarse just above the base size)
    M = uw.discretisation.MeshVariable("Mjump", base, 1, degree=1)
    band = sympy.exp(-(((base.N.x - 0.5) / 0.08) ** 2))
    M.data[:, 0] = _ev(1.0 / (0.07 + (1.0 / 80 - 0.07) * band) ** 2, M.coords)
    child = base.adapt(M, max_levels=1)
    assert _ncell(child) < n_uniform      # local, not a uniform refine

    def _solcx(mesh):
        sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1)
        s = uw.systems.Stokes(mesh)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
        s.saddle_preconditioner = 1.0 / sol.fn_viscosity
        s.bodyforce = sol.fn_bodyforce
        s.add_dirichlet_bc((0.0, None), "Left")
        s.add_dirichlet_bc((0.0, None), "Right")
        s.add_dirichlet_bc((None, 0.0), "Bottom")
        s.add_dirichlet_bc((None, 0.0), "Top")
        s.petsc_use_pressure_nullspace = True
        s.petsc_options["snes_type"] = "ksponly"
        s.tolerance = 1e-8
        return s, sol

    sg, solg = _solcx(child)
    sg.preconditioner = "gamg"
    sg.solve()
    it_g = sg.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getIterationNumber()

    s, sol = _solcx(child)
    assert s._custom_mg is None           # nothing registered on the solver
    s.solve()                             # velocity block auto-picks-up FMG
    vksp = s.snes.getKSP().getPC().getFieldSplitSubKSP()[0]

    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGType() == 2          # PETSc PC.MGType.FULL == FMG
    assert vksp.getPC().getMGLevels() == len(base.dm_hierarchy) + 1
    assert vksp.getType() == "fgmres"
    assert vksp.getIterationNumber() <= it_g      # FMG matches/beats GAMG
    rel = np.linalg.norm(s.u.data - sg.u.data) / (np.linalg.norm(sg.u.data) + 1e-30)
    assert rel < 1e-4
    assert sol.velocity_error(s.u) < 2.0 * solg.velocity_error(sg.u) + 1e-6


def test_each_sbr_level_is_its_own_mg_level():
    """Every SBR refinement step must add its OWN custom-P MG level (not collapse
    into a single base-finest -> child jump): MG levels = base levels + n_sbr."""
    base = _base()                       # refinement=2 -> 3 base hierarchy levels
    n_base = len(base.dm_hierarchy)
    sharp = _metric(base, 0.5, h_fine=1.0 / 150, width=0.06)
    for ml, want_levels in [(1, n_base + 1), (2, n_base + 2)]:
        child = base.adapt(sharp, max_levels=ml)
        # coarse tail + the child itself
        assert len(child._custom_mg_coarse_meshes) + 1 == want_levels

    # a Poisson solve on a 2-level child must actually drive all n_base+2 levels
    child2 = base.adapt(sharp, max_levels=2)
    s = _poisson(child2)
    s.solve()
    assert s.snes.getKSP().getPC().getType() == "mg"
    assert s.snes.getKSP().getPC().getMGLevels() == n_base + 2
    assert s.snes.getConvergedReason() > 0


def test_adapt_requires_base_hierarchy():
    # a mesh without a refinement hierarchy cannot supply the MG coarse tail
    flat = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.3, regular=True, qdegree=3)
    H = uw.discretisation.MeshVariable("Hflat", flat, 1, degree=1)
    H.data[:, 0] = 1.0 / 0.02**2
    with pytest.raises(RuntimeError):
        flat.adapt(H, max_levels=1)
