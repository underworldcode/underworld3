"""``Mesh.multigrid_levels``: a mesh's refinement hierarchy as meshes, coarsest first,
with the mesh itself last, the coarse tail a custom multigrid hierarchy is built from."""
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _box(refinement):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4, regular=False, qdegree=2,
        refinement=refinement)


def _n_cells(mesh):
    start, end = mesh.dm.getHeightStratum(0)
    return end - start


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


def test_levels_are_built_once_and_read_only():
    mesh = _box(refinement=1)
    first, second = mesh.multigrid_levels, mesh.multigrid_levels
    assert all(a is b for a, b in zip(first, second))
    with pytest.raises(AttributeError):
        mesh.multigrid_levels = ()


def test_a_mesh_without_refinement_is_its_own_level():
    mesh = _box(refinement=None)
    assert mesh.multigrid_levels == (mesh,)


def test_the_levels_drive_a_custom_fmg_solve():
    mesh = _box(refinement=2)
    poisson = uw.systems.Poisson(mesh)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.f = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        poisson.add_dirichlet_bc(0, wall)
    poisson.set_custom_fmg(mesh.multigrid_levels[:-1])
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    pc = poisson.snes.getKSP().getPC()
    assert pc.getType() == "mg"
    assert pc.getMGLevels() == len(mesh.multigrid_levels)
