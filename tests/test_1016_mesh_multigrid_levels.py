"""``Mesh.multigrid_levels``: a mesh's multigrid hierarchy as meshes, coarsest first,
with the mesh itself last: the coarse tail it owns (an adapt child's), or else its
refinement or coarsening hierarchy."""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.custom_mg import _is_native_refine_pair

pytestmark = pytest.mark.level_2


def _box(**hierarchy):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4, regular=False, qdegree=2,
        **hierarchy)


def _n_cells(mesh):
    start, end = mesh.dm.getHeightStratum(0)
    return end - start


def _poisson(mesh):
    poisson = uw.systems.Poisson(mesh)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.f = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        poisson.add_dirichlet_bc(0, wall)
    return poisson


@pytest.mark.tier_b
def test_levels_are_the_refinement_hierarchy_coarsest_first():
    mesh = _box(refinement=2)
    levels = mesh.multigrid_levels
    assert isinstance(levels, tuple)
    assert len(levels) == len(mesh.dm_hierarchy) == 3
    assert levels[-1] is mesh
    # one uniform refinement of a triangle mesh splits every cell into four
    cells = [_n_cells(level) for level in levels]
    assert cells[1] == 4 * cells[0] and cells[2] == 4 * cells[1], cells
    for level, dm in zip(levels, mesh.dm_hierarchy):
        start, end = dm.getHeightStratum(0)
        assert _n_cells(level) == end - start
    # consecutive levels are a refine() pair, so custom_mg transfers between them
    # by the exact nested prolongation
    assert all(_is_native_refine_pair(a, b) for a, b in zip(levels, levels[1:]))


@pytest.mark.tier_b
def test_levels_are_built_once_and_reading_them_changes_nothing():
    mesh = _box(refinement=2)
    levels = mesh.multigrid_levels
    assert all(a is b for a, b in zip(levels, mesh.multigrid_levels))
    with pytest.raises(AttributeError):
        mesh.multigrid_levels = ()
    # a coarse level is a single-level mesh: it is its own hierarchy, and reading
    # it must not re-tag it out of the family it belongs to
    for level in levels[:-1]:
        assert level.multigrid_levels == (level,)
    assert all(_is_native_refine_pair(a, b) for a, b in zip(levels, levels[1:]))


@pytest.mark.tier_b
def test_a_mesh_without_a_hierarchy_is_its_own_level():
    mesh = _box()
    assert mesh.multigrid_levels == (mesh,)
    # and no coarse wrap of it was built
    assert getattr(mesh, "_coarse_level_meshes_cache", None) is None


@pytest.mark.tier_b
@pytest.mark.skipif(uw.mpi.size > 1, reason="PETSc coarsens a distributed 2-D mesh "
                    "through ParMmg, which is 3-D only")
def test_a_coarsening_hierarchy():
    source = _box()
    mesh = uw.discretisation.Mesh(source.dm.clone(), simplex=True, qdegree=2,
                                  boundaries=source.boundaries, coarsening=1)
    levels = mesh.multigrid_levels
    assert len(levels) == len(mesh.dm_hierarchy) == 2
    assert levels[-1] is mesh
    assert _n_cells(levels[0]) < _n_cells(mesh)


@pytest.mark.tier_b
def test_an_adapt_child_reports_the_tail_its_solvers_use():
    base = _box(refinement=1)
    child = base.adapt(lambda c: np.where(
        np.hypot(c[:, 0] - 0.5, c[:, 1] - 0.5) < 0.25, 64.0**2, 4.0**2), max_levels=2)
    levels = child.multigrid_levels
    tail = child._custom_mg_coarse_meshes
    assert len(levels) == len(tail) + 1 >= 3
    assert all(a is b for a, b in zip(levels, tail)) and levels[-1] is child
    # a solver on the child picks the same hierarchy up with no set_custom_fmg call
    poisson = _poisson(child)
    poisson.solve()
    pc = poisson.snes.getKSP().getPC()
    assert pc.getType() == "mg" and pc.getMGLevels() == len(levels)


@pytest.mark.tier_b
def test_the_levels_drive_a_custom_fmg_solve():
    mesh = _box(refinement=2)
    poisson = _poisson(mesh)
    poisson.set_custom_fmg(mesh.multigrid_levels[:-1])
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    # without the registration this solver runs GAMG
    assert poisson.snes.getKSP().getPC().getType() == "mg"
