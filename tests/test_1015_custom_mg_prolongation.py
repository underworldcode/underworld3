"""
Custom multigrid prolongation hook (set_custom_mg / utilities.custom_mg).

Geometric multigrid driven by a prolongation we build ourselves (barycentric or
RBF) from independent, non-nested coarse meshes — decoupling geometric MG from a
nested refine() hierarchy. See docs/developer/design and the study notes.
"""
import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1
    p.f = 0.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    return p


def _box(cellSize, refinement=None):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cellSize, refinement=refinement, qdegree=2)


def test_custom_mg_barycentric_matches_fmg():
    """Custom barycentric P (independent coarse meshes) must converge and agree
    with the nested-FMG solution on the same fine problem."""
    fmg = _poisson(_box(0.25, refinement=2))
    fmg.solve()
    assert fmg.snes.getConvergedReason() > 0

    cust = _poisson(_box(0.25, refinement=2))
    cust.set_custom_mg([_box(0.285), _box(0.142)], kind="barycentric")
    cust.solve()
    assert cust.snes.getKSP().getPC().getType() == "mg"
    assert cust.snes.getConvergedReason() > 0
    # same fine problem -> same solution (Galerkin RAP from our P)
    rel = (np.linalg.norm(cust.Unknowns.u.data - fmg.Unknowns.u.data)
           / (np.linalg.norm(fmg.Unknowns.u.data) + 1e-30))
    assert rel < 1e-4
    # competitive with nested FMG (not pathologically worse)
    assert cust.snes.getKSP().getIterationNumber() <= fmg.snes.getKSP().getIterationNumber() + 5


def test_custom_mg_rbf_drives_solver():
    """The RBF builder must also drive a converging geometric MG solve."""
    cust = _poisson(_box(0.25, refinement=2))
    cust.set_custom_mg([_box(0.285), _box(0.142)], kind="rbf")
    cust.solve()
    assert cust.snes.getKSP().getPC().getType() == "mg"
    assert cust.snes.getConvergedReason() > 0


def test_custom_mg_without_refine_hierarchy():
    """Key capability: geometric MG on a mesh with NO refine() hierarchy
    (FMG unavailable -> normally GAMG). Custom P supplies the hierarchy."""
    mesh = _box(0.08)                 # no refinement -> single-level
    assert len(mesh.dm_hierarchy) == 1
    cust = _poisson(mesh)
    cust.set_custom_mg([_box(0.3), _box(0.16)], kind="barycentric")
    cust.solve()
    assert cust.snes.getKSP().getPC().getType() == "mg"
    assert cust.snes.getConvergedReason() > 0


def test_set_custom_mg_validation():
    p = _poisson(_box(0.5))
    with pytest.raises(ValueError):
        p.set_custom_mg([_box(0.8)], kind="not-a-builder")
    with pytest.raises(ValueError):
        p.set_custom_mg([], kind="barycentric")
