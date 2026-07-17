"""3D (tetrahedral) NVB graded adapt-on-top — serial engine + MG gate.

Stage 1b of the adaptivity capstone
(``docs/developer/design/ADAPTIVITY_3D_SPHERICAL_2026-07.md``): the
dimension-general tagged-simplex engine
(:class:`underworld3.utilities.nvb.TaggedBisectionMesh` — Maubach's bisection
rule with the Diening--Gehring--Storn coloring initialization) behind
``mesh.adapt(...)`` on tetrahedral meshes at np=1.

Engine-level properties (fast, no solves):
  - conformity: 0 hanging edges / 0 over-shared faces at every generation;
  - bounded closure: one cell deep in a uniformly refined patch adds O(1)
    cells locally (longest-edge would drain the patch);
  - shape regularity: at most 36 similarity classes per base tet (the
    Maubach/DGS theorem bound) under deep uniform refinement.

The MG-viability gate (the capstone's per-stage acceptance):
  - engine-less ``mesh.adapt(metric, max_levels=...)`` on a 3D mesh returns a
    graded child carrying the ``[base ... child]`` custom-P hierarchy;
  - a Poisson solve on the child drives custom-P geometric FMG with one MG
    level per refinement generation, matches a GAMG reference, and reproduces
    the exact linear solution (which also validates the 3D facet-label
    transfer, since the Dirichlet boundaries came through ``to_dm``).

The np>1 3D path raises ``NotImplementedError`` (the parallel tetrahedral
transform is stage 1c); 2D behaviour is untouched and covered by
``test_0836`` / ``test_0839``.
"""
import numpy as np
import pytest
import underworld3 as uw
from underworld3.utilities.nvb import TaggedBisectionMesh

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

BOUNDS_3D = [("Bottom", 11), ("Top", 12), ("Right", 13), ("Left", 14),
             ("Front", 15), ("Back", 16)]


def _ncell(mesh):
    cs, ce = mesh.dm.getHeightStratum(0)
    return ce - cs


def _base3(cellSize=0.4, refinement=1):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=cellSize,
        refinement=refinement, qdegree=2)


def _ball_metric(h_fine=0.06, h_coarse=0.4, r_core=0.18, width=0.25):
    """Callable metric M = 1/h(r)^2, fine inside a ball around the centre —
    the exact-geometry metric kind ``adapt`` resolves per level."""
    def metric(centroids):
        r = np.linalg.norm(np.asarray(centroids) - 0.5, axis=1)
        h = np.where(r < r_core, h_fine,
                     np.minimum(h_fine + (h_coarse - h_fine)
                                * (r - r_core) / width, h_coarse))
        return 1.0 / h**2
    return metric


# --------------------------------------------------------------------------- #
#  Engine-level properties (TaggedBisectionMesh in isolation)
# --------------------------------------------------------------------------- #
def test_conformity_every_generation():
    base = _base3()
    eng = TaggedBisectionMesh.from_dm(base.dm_hierarchy[-1],
                                      boundaries=BOUNDS_3D)
    metric = _ball_metric()
    for gen in range(6):
        cen, h, cids = eng.centroids_h()
        sel = np.where(h > 1.0 / np.sqrt(metric(cen)))[0]
        if sel.size == 0:
            break
        eng.refine(set(int(cids[j]) for j in sel))
        hanging, overshared = eng.check_conforming()
        assert hanging == 0 and overshared == 0, (
            f"gen {gen}: hanging={hanging} overshared={overshared}")


def test_bounded_closure_single_cell_is_local():
    base = _base3()
    eng = TaggedBisectionMesh.from_dm(base.dm_hierarchy[-1])
    for _ in range(6):                      # two full isotropic levels
        eng.refine(list(eng.cells))
    n0 = len(eng.cells)
    cen, _, cids = eng.centroids_h()
    deep = int(cids[np.argmin(np.linalg.norm(cen - 0.5, axis=1))])
    eng.bisect(deep)
    added = len(eng.cells) - n0
    hanging, overshared = eng.check_conforming()
    assert hanging == 0 and overshared == 0
    assert added < 200, f"closure drained: +{added} cells for one mark"


def test_shape_regularity_bounded_similarity_classes():
    base = _base3(cellSize=0.6)
    eng = TaggedBisectionMesh.from_dm(base.dm_hierarchy[-1])
    for _ in range(9):                      # three full isotropic levels
        eng.refine(list(eng.cells))
    assert eng.similarity_classes() <= 36   # dim!·dim·2^(dim-2), dim=3


# --------------------------------------------------------------------------- #
#  Integrated path + the MG gate
# --------------------------------------------------------------------------- #
def test_adapt_engineless_3d_returns_graded_child():
    base = _base3()
    n0 = _ncell(base)
    child = base.adapt(_ball_metric(), max_levels=1)     # no engine keyword
    assert child is not base
    assert child.parent is base
    assert child._adapt_engine == "nvb"
    assert _ncell(child) > n0
    assert _ncell(base) == n0                            # base untouched
    # every refinement generation is its own MG level:
    # coarse levels = base hierarchy + (generations - 1) intermediates
    n_gens = len(child._adapt_markers)
    assert 1 <= n_gens <= 3                              # dim * max_levels
    assert (len(child._custom_mg_coarse_meshes)
            == len(base.dm_hierarchy) + n_gens - 1)
    # the finest cells concentrate in the marked core (grading, not a patch);
    # geometry is read from a clone (the live mesh DM raises err73 from
    # computeCellGeometryFVM — see custom_mg)
    d = child.dm.clone()
    cs, ce = d.getHeightStratum(0)
    vols = np.empty(ce - cs)
    cens = np.empty((ce - cs, 3))
    for i, c in enumerate(range(cs, ce)):
        vol, cen = d.computeCellGeometryFVM(c)[0:2]
        vols[i] = abs(float(vol))
        cens[i] = np.asarray(cen)[:3]
    finest = vols < np.quantile(vols, 0.10)
    assert np.median(np.linalg.norm(cens[finest] - 0.5, axis=1)) < 0.35, (
        "finest decile of cells is not concentrated in the marked core")


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1
    p.f = 0.0
    p.add_dirichlet_bc(0.0, "Bottom")                    # z = 0
    p.add_dirichlet_bc(1.0, "Top")                       # z = 1
    p.petsc_options["ksp_rtol"] = 1e-8
    p.petsc_options["ksp_type"] = "cg"
    return p


def test_poisson_fmg_on_3d_nvb_child_matches_gamg():
    base = _base3()
    child = base.adapt(_ball_metric(), max_levels=1)

    s = _poisson(child)
    s.solve()                                            # NO set_custom_fmg
    assert s.snes.getKSP().getPC().getType() == "mg"
    assert (s.snes.getKSP().getPC().getMGLevels()
            == len(child._custom_mg_coarse_meshes) + 1)
    assert s.snes.getConvergedReason() > 0

    g = _poisson(child)
    g.preconditioner = "gamg"
    g.solve()
    rel = np.linalg.norm(s.Unknowns.u.data - g.Unknowns.u.data) / (
        np.linalg.norm(g.Unknowns.u.data) + 1e-30)
    assert rel < 1e-4
    # exact linear field T = z — also proves the Dirichlet facet labels
    # survived the engine's to_dm transfer
    err = np.linalg.norm(s.Unknowns.u.data[:, 0] - s.Unknowns.u.coords[:, 2]) / (
        np.linalg.norm(s.Unknowns.u.coords[:, 2]) + 1e-30)
    assert err < 1e-8, f"Dirichlet labels wrong on 3D NVB child: err {err}"
