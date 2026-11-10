"""A solve after a mesh move runs on the moved geometry, including when the mesh
was built with refinement and the solver's multigrid uses its hierarchy (#1037).

The same Poisson problem is solved on a box built on [0, 1.5] x [0, 1] and on a
[0, 1]^2 box stretched there with ``mesh.deform``. Before the fix, a refined
mesh's solver assembled on the unmoved finest hierarchy level: 51 % off.
Runs serially here and at np 2 from tests/parallel.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(240)]


def _poisson(mesh, name, mg):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=2)
    solver = uw.systems.Poisson(mesh, u_Field=T)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0
    x, y = mesh.X
    solver.f = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)
    for wall in ("Bottom", "Top", "Left", "Right"):
        solver.add_dirichlet_bc(0.0, wall)
    solver.tolerance = 1.0e-10
    if mg:
        solver.petsc_options["pc_type"] = "mg"
        solver.petsc_options["pc_mg_type"] = "full"
        solver.petsc_options["pc_mg_galerkin"] = "both"
    return T, solver


def _integral(mesh, T):
    return float(uw.maths.Integral(mesh, T.sym[0]).evaluate())


def _stretched_box(refinement):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, refinement=refinement, qdegree=3)
    return mesh


def _stretch(mesh):
    X = np.asarray(mesh.X.coords).copy()
    X[:, 0] *= 1.5
    mesh.deform(X)


def _solver_xmax(solver):
    c = solver.snes.getDM().getCoordinates().getArray().reshape(-1, 2)
    local = float(c[:, 0].max()) if len(c) else -np.inf
    return uw.mpi.comm.allreduce(local, op=uw.MPI.MAX)


@pytest.mark.parametrize("refinement", [0, 1, 2])
@pytest.mark.parametrize("mg", [False, True], ids=["default_pc", "fmg"])
@pytest.mark.parametrize("when", ["solver_after_move", "solver_before_move"])
def test_a_solve_after_a_move_uses_the_moved_geometry(refinement, mg, when):
    if mg and refinement == 0:
        pytest.skip("geometric multigrid needs a hierarchy")
    built = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1.5, 1), cellSize=1 / 8, refinement=refinement, qdegree=3)
    Tb, sb = _poisson(built, "Tb", mg)
    sb.solve()
    expected = _integral(built, Tb)

    moved = _stretched_box(refinement)
    if when == "solver_before_move":
        Tm, sm = _poisson(moved, "Tm", mg)
        sm.solve()
        _stretch(moved)
    else:
        _stretch(moved)
        Tm, sm = _poisson(moved, "Tm", mg)
    sm.solve()

    assert _solver_xmax(sm) == pytest.approx(1.5, abs=1e-12)
    # Built and moved meshes differ only in triangulation: 7e-5 at refinement 0.
    assert _integral(moved, Tm) == pytest.approx(expected, rel=1e-3)


def test_the_coarse_levels_keep_their_geometry_and_the_finest_level_moves():
    mesh = _stretched_box(2)
    coarse_before = mesh.dm_hierarchy[0].getCoordinates().getArray().copy()
    _stretch(mesh)
    finest = mesh.dm_hierarchy[-1].getCoordinatesLocal().array
    assert np.array_equal(finest, mesh.dm.getCoordinatesLocal().array)
    assert np.array_equal(mesh.dm_hierarchy[0].getCoordinates().getArray(), coarse_before)
    # Its own vector: a later write through mesh.dm doesn't move the hierarchy by itself.
    assert (mesh.dm_hierarchy[-1].getCoordinatesLocal().handle
            != mesh.dm.getCoordinatesLocal().handle)


def test_a_solve_after_relax_on_a_refined_annulus_uses_the_relaxed_geometry():
    # relax() moves nodes through the MMPDE mover; the FMG notes invite a solve after it.
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 4,
                              refinement=1, qdegree=3)
    T = uw.discretisation.MeshVariable("Tr", mesh, 1, degree=2)
    solver = uw.systems.Poisson(mesh, u_Field=T)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0
    x, y = mesh.X
    solver.f = x * y
    solver.add_dirichlet_bc(0.0, "Lower")
    solver.add_dirichlet_bc(0.0, "Upper")
    solver.petsc_options["pc_type"] = "mg"
    solver.petsc_options["pc_mg_type"] = "full"
    solver.petsc_options["pc_mg_galerkin"] = "both"
    solver.solve()
    before = mesh.dm.getCoordinatesLocal().array.copy()
    mesh.relax()
    after = mesh.dm.getCoordinatesLocal().array
    moved = uw.mpi.comm.allreduce(float(np.abs(after - before).max()) if after.size else 0.0,
                                  op=uw.MPI.MAX)
    assert moved > 0.0, "relax() did not move this mesh, so the test shows nothing"
    solver.solve()
    solver_coords = solver.snes.getDM().getCoordinatesLocal().array
    assert np.array_equal(solver_coords, after)
