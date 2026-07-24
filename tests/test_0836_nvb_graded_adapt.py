"""Layer-2 NVB graded adapt-on-top: ``mesh.adapt(engine="nvb")``.

Newest-vertex bisection is a *graded* simplicial refinement engine with a
*bounded* conforming closure — the property PETSc's longest-edge ``refine_sbr``
lacks (it can only build a uniform-finest patch). These tests pin both the engine
(``underworld3.utilities.nvb.NVBMesh``) and the integrated path:

  - conformity: 0 hanging nodes / 0 over-shared edges at every generation;
  - bounded closure: one cell deep in a uniform patch adds O(1) cells locally
    (not a drain-to-edge cascade), and #added ≤ C·#marked for a marked region;
  - shape-regularity: the number of triangle similarity classes stays bounded
    under deep refinement (no slivers);
  - grading: a bullseye and a fault funnel confine the FINEST generation near the
    feature (NVB grades where SBR refills) — and the NVB child has fewer DOFs than
    the SBR uniform patch for the same metric;
  - the graded NVB child drives the custom-P geometric-MG FMG: Poisson and SolCx
    Stokes (η jump 1e6) match a GAMG reference.

These tests exercise the serial NVBMesh reference engine (Route A) and the
integrated path; the 3D guard is asserted to raise. Parallel NVB (the native
uwnvb transform, Route B) and its confluence + FMG acceptance live in
``test_0839_nvb_parallel_adapt.py``.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw
from underworld3.function import analytic as A
from underworld3.utilities.nvb import NVBMesh

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

BOUNDS = [("Bottom", 11), ("Top", 12), ("Right", 13), ("Left", 14)]


def _ev(fn, coords):
    return np.asarray(uw.function.evaluate(fn, np.asarray(coords))).reshape(-1)


def _ncell(mesh):
    cs, ce = mesh.dm.getHeightStratum(0)
    return ce - cs


def _base(cellSize=0.25, refinement=2):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cellSize, regular=True,
        refinement=refinement, qdegree=3)


def _nvb_from_base(base):
    return NVBMesh.from_dm(base.dm_hierarchy[-1], boundaries=BOUNDS)


# --------------------------------------------------------------------------- #
#  Engine-level properties (NVBMesh in isolation — fast, no solves)
# --------------------------------------------------------------------------- #
def test_conformity_every_generation():
    """No hanging nodes / over-shared edges after each refinement generation."""
    nvb = _nvb_from_base(_base())
    assert nvb.check_conforming() == (0, 0)
    for _gen in range(5):
        cen, h, cids = nvb.centroids_h()
        r = np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5)
        marked = cids[r < 0.25]
        if marked.size == 0:
            break
        nvb.refine(set(int(c) for c in marked))
        assert nvb.check_conforming() == (0, 0), f"hanging nodes at gen {_gen}"


def test_bounded_closure_single_cell_is_local():
    """One marked cell deep in a uniform patch adds O(1) cells (bounded local
    closure) — the SBR drain-to-interface would add hundreds."""
    nvb = _nvb_from_base(_base())
    cen, h, cids = nvb.centroids_h()
    # a cell near the centre, far from any boundary
    r = np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5)
    target = int(cids[np.argmin(r)])
    n_before = len(nvb.cells)
    moved = cen[np.argmin(r)]
    nvb.refine({target})
    added = len(nvb.cells) - n_before
    assert added <= 8, f"closure not local: +{added} cells from 1 mark"
    # the added cells are confined near the marked cell, not drained to the edge
    cen2, _, _ = nvb.centroids_h()
    d = np.hypot(cen2[:, 0] - moved[0], cen2[:, 1] - moved[1])
    # everything that changed lives within a couple of base cells of the seed
    assert d[d < 0.2].size >= added


def test_bounded_closure_proportional_to_marked():
    """#added ≤ C·#marked for a marked region (the BDD/Stevenson bound)."""
    nvb = _nvb_from_base(_base())
    cen, h, cids = nvb.centroids_h()
    r = np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5)
    marked = cids[r < 0.3]
    n_before = len(nvb.cells)
    nvb.refine(set(int(c) for c in marked))
    added = len(nvb.cells) - n_before
    assert added <= 6 * marked.size, f"closure {added} >> 6·{marked.size} marked"


def test_shape_regularity_bounded_similarity_classes():
    """Deep refinement does not degenerate elements: the number of triangle
    similarity classes stays small (NVB ≤ a few per base triangle)."""
    nvb = _nvb_from_base(_base(cellSize=0.5, refinement=1))
    base_classes = nvb.similarity_classes()
    for _gen in range(8):
        cen, h, cids = nvb.centroids_h()
        r = np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5)
        marked = cids[r < 0.3]
        if marked.size == 0:
            break
        nvb.refine(set(int(c) for c in marked))
    # NVB's hallmark: a small constant number of shapes, regardless of depth
    assert nvb.similarity_classes() <= max(8, 4 * base_classes)


def test_bullseye_finest_generation_confined():
    """A bullseye (nested shrinking disks) confines the FINEST generation near the
    centre — a graded staircase, not a uniform-finest core."""
    nvb = _nvb_from_base(_base())
    radii = [0.30, 0.22, 0.15, 0.09, 0.05]
    for rad in radii:
        cen, h, cids = nvb.centroids_h()
        r = np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5)
        marked = cids[r < rad]
        if marked.size:
            nvb.refine(set(int(c) for c in marked))

    cen, h, cids = nvb.centroids_h()
    depths = np.array([nvb.depth[int(c)] for c in cids])
    r = np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5)
    dmax = depths.max()
    # finest cells hug the centre; coarse (unrefined) cells survive at the edge
    assert r[depths == dmax].max() < 0.20, "finest generation not confined"
    assert depths.min() == 0, "no coarse cells survive — refilled, not graded"
    # a genuine staircase: several distinct depth levels coexist
    assert np.unique(depths).size >= 4


def test_fault_funnel_finest_hugs_the_line():
    """A dipping-line feature funnels the finest generation to within a thin band
    of the line (perpendicular grading)."""
    nvb = _nvb_from_base(_base(cellSize=0.2, refinement=2))
    # line through (0.2,0.85)->(0.85,0.2): n·x = c with unit normal
    n = np.array([1.0, 1.0]) / np.sqrt(2)
    c0 = n @ np.array([0.2, 0.85])

    def dist(cen):
        return np.abs(cen @ n - c0)

    for band in [0.18, 0.12, 0.08, 0.05, 0.03]:
        cen, h, cids = nvb.centroids_h()
        marked = cids[dist(cen) < band]
        if marked.size:
            nvb.refine(set(int(x) for x in marked))

    cen, h, cids = nvb.centroids_h()
    depths = np.array([nvb.depth[int(x)] for x in cids])
    dmax = depths.max()
    assert dist(cen)[depths == dmax].max() < 0.08, "finest band not hugging the line"
    assert depths.min() == 0
    assert nvb.check_conforming() == (0, 0)


# --------------------------------------------------------------------------- #
#  Integrated path: mesh.adapt(engine="nvb")
# --------------------------------------------------------------------------- #
def _band_metric(base, center=0.5, h_fine=1 / 100, width=0.08):
    M = uw.discretisation.MeshVariable("Mnvb", base, 1, degree=1)
    band = sympy.exp(-(((base.N.x - center) / width) ** 2))
    M.data[:, 0] = _ev(1.0 / (0.07 + (h_fine - 0.07) * band) ** 2, M.coords)
    return M


def test_adapt_nvb_returns_graded_child():
    base = _base()
    n0 = _ncell(base)
    M = _band_metric(base)
    child = base.adapt(M, max_levels=1, engine="nvb")

    assert child is not base
    assert child.parent is base
    assert child._relationship_kind == "refinement"
    assert child._adapt_engine == "nvb"
    assert _ncell(child) > n0
    assert _ncell(base) == n0                         # base untouched
    # 2·max_levels NVB generations -> base levels + (generations-1) intermediate
    assert len(child._custom_mg_coarse_meshes) + 1 == len(base.dm_hierarchy) + 2


def test_nvb_child_fewer_dofs_than_sbr_patch():
    """For the same metric, NVB grades (fewer DOFs) where SBR refills a uniform
    patch. Compare at matched isotropic-equivalent resolution (NVB 2 gens ≈ SBR 1
    pass)."""
    base = _base()
    M = _band_metric(base, h_fine=1 / 90, width=0.06)
    sbr = base.adapt(M, max_levels=1, engine="sbr")
    nvb = base.adapt(M, max_levels=1, engine="nvb")
    assert _ncell(nvb) < _ncell(sbr)


def test_nvb_readapt_is_non_cumulative():
    base = _base()
    n0 = _ncell(base)
    c1 = base.adapt(_band_metric(base, center=0.7), max_levels=1, engine="nvb")
    c2 = base.adapt(_band_metric(base, center=0.3), max_levels=1, engine="nvb")
    assert _ncell(base) == n0
    assert c2 is not c1


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


def test_poisson_fmg_on_nvb_child_matches_gamg():
    base = _base()
    child = base.adapt(_band_metric(base), max_levels=1, engine="nvb")

    s = _poisson(child)
    s.solve()                                         # NO set_custom_fmg
    assert s.snes.getKSP().getPC().getType() == "mg"
    assert s.snes.getKSP().getPC().getMGLevels() == len(base.dm_hierarchy) + 2
    assert s.snes.getConvergedReason() > 0

    g = _poisson(child)
    g.preconditioner = "gamg"
    g.solve()
    rel = np.linalg.norm(s.Unknowns.u.data - g.Unknowns.u.data) / (
        np.linalg.norm(g.Unknowns.u.data) + 1e-30)
    assert rel < 1e-4
    # exact linear field T = y
    err = np.linalg.norm(s.Unknowns.u.data[:, 0] - s.Unknowns.u.coords[:, 1]) / (
        np.linalg.norm(s.Unknowns.u.coords[:, 1]) + 1e-30)
    assert err < 1e-8, f"Dirichlet labels wrong on NVB child: err {err}"


def test_solcx_stokes_velocity_fmg_on_nvb_child():
    """SolCx Stokes (η jump 1e6) on a graded NVB child: the velocity block must
    auto-pick-up the mesh-owned custom-P FMG and match a GAMG reference."""
    base = _base()
    M = uw.discretisation.MeshVariable("Mjump", base, 1, degree=1)
    band = sympy.exp(-(((base.N.x - 0.5) / 0.08) ** 2))
    M.data[:, 0] = _ev(1.0 / (0.07 + (1.0 / 80 - 0.07) * band) ** 2, M.coords)
    child = base.adapt(M, max_levels=1, engine="nvb")

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
    assert s._custom_mg is None
    s.solve()
    vksp = s.snes.getKSP().getPC().getFieldSplitSubKSP()[0]
    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGType() == 2              # PC.MGType.FULL == FMG
    assert vksp.getType() == "fgmres"
    assert vksp.getIterationNumber() <= it_g          # FMG matches/beats GAMG
    rel = np.linalg.norm(s.u.data - sg.u.data) / (np.linalg.norm(sg.u.data) + 1e-30)
    assert rel < 1e-4
    assert sol.velocity_error(s.u) < 2.0 * solg.velocity_error(sg.u) + 1e-6


def test_nvb_3d_serial_returns_child():
    """3D NVB adapt is served by the serial tagged-simplex engine at np=1
    (the full 3D gates live in test_0840_nvb_3d_serial_adapt.py)."""
    base3 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.4, regular=False,
        refinement=1, qdegree=2)
    H = uw.discretisation.MeshVariable("H3", base3, 1, degree=1)
    H.data[:, 0] = 1.0 / 0.2**2
    child = base3.adapt(H, max_levels=1, engine="nvb")
    assert child.parent is base3
    assert _ncell(child) > _ncell(base3)


def test_curved_boundary_snaps_every_generation():
    """Round-3b ruling (2026-07-24): new boundary vertices on curved
    domains snap onto the registered analytic surfaces at EVERY
    generation, so each intermediate level is a valid mesh in its own
    right and boundary geometry converges with refinement (chords froze
    it at base resolution: radius error ~h_base^2/8R)."""
    import underworld3 as uw

    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                              cellSize=0.25, refinement=1, qdegree=2)

    def metric(centroids):
        r = np.linalg.norm(np.asarray(centroids)[:, :2], axis=1)
        return 1.0 / np.minimum(0.03 + 0.5 * np.abs(1.0 - r), 0.3) ** 2

    child = mesh.adapt(metric, max_levels=2)
    from underworld3.meshing.smoothing import _pinned_mask

    X = np.asarray(child.X.coords)
    r = np.linalg.norm(X, axis=1)
    for label, R in (("Lower", 0.5), ("Upper", 1.0)):
        mask = _pinned_mask(child.dm, (label,))
        assert mask.any()
        assert np.abs(r[mask] - R).max() < 1.0e-12, (
            f"{label} boundary not snapped: "
            f"max radius error {np.abs(r[mask]-R).max():.2e}")
    # ... and the intermediate MG levels are snapped too (the ruling's
    # point: every level is a valid mesh)
    for i, lvl in enumerate(child._custom_mg_coarse_meshes[-2:]):
        dm = lvl.dm if hasattr(lvl, "dm") else lvl
        Xl = dm.getCoordinatesLocal().array.reshape(-1, 2)
        rl = np.linalg.norm(Xl, axis=1)
        on_out = _pinned_mask(dm, ("Upper",))
        if on_out.any():
            assert np.abs(rl[on_out] - 1.0).max() < 1.0e-12
