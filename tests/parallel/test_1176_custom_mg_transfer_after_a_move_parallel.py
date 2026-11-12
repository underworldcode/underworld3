"""A move that keeps the topology keeps the custom multigrid transfers, on
several ranks.

The serial half of the record is tests/test_1176_custom_mg_transfer_after_a_move.py.
At more than one rank a native refine pair goes to the node-cloud builder and
a split level to the partitioned cell-located transfer, so the routes here are
not those of the serial file. A moved level is handed to each of them on the
nodes it was built with, bit for bit, and the transfers after a move are those
before it entry for entry.

The node-cloud builder is not continuous in its nodes. With the built nodes
recovered only to round-off (2e-16), one weight of the tetrahedral pair below
changed by 1.4e-4 at 2 and 4 ranks and by 1.1e-4 at 8.

The iteration counts were measured on 2, 3, 4, 5 and 8 ranks.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1176_custom_mg_transfer_after_a_move_parallel.py
"""
import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg, fault_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900)]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])


def _transfers(solver):
    """Copies of the hierarchy's transfers as built, coarsest pair first."""
    return [P.copy() for P in solver._custom_mg["hierarchy"].transfers]


def _largest_change(before, after):
    """The largest row sum of ``|after - before|`` over the pairs."""
    assert len(after) == len(before)
    worst = 0.0
    for P0, P1 in zip(before, after):
        assert P1.getSize() == P0.getSize()
        D = P1.copy()
        D.axpy(-1.0, P0)
        worst = max(worst, D.norm(PETSc.NormType.INFINITY))
        D.destroy()
    return worst


def _deform(mesh, amplitude=0.02):
    c = np.asarray(mesh.X.coords).copy()
    c[:, 0] += amplitude * np.prod(np.sin(np.pi * c), axis=1)   # zero on the walls
    mesh.deform(c)


def _raise_the_top_wall(mesh):
    c = np.asarray(mesh.X.coords).copy()
    c[:, 1] *= 1 + 0.05 * np.sin(np.pi * c[:, 0])
    mesh.deform(c)


def _wall_box():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 4, regular=False,
        qdegree=2, refinement=2)


def _wall_stokes(mesh):
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Left", "Right"):
        s.add_dirichlet_bc((0, 0), wall)
    s.bodyforce = sympy.Matrix([0, sympy.sin(sympy.pi * x)])
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, list(mesh.multigrid_levels[:-1]), field_id=0)
    return s


def _velocity_iterations(stokes):
    return stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getIterationNumber()


# Velocity iterations at rest and after each of three rises of the top wall,
# by rank count; a solver built on the moved mesh takes 4 at each of them.
# With the transfers rebuilt from the moved coordinates the steps take 6, 7
# and 8 at 5 and 8 ranks.
WALL_COUNTS = {2: [3, 3, 4, 4], 3: [4, 4, 4, 4], 4: [3, 3, 4, 4],
               5: [4, 4, 4, 4], 8: [4, 4, 4, 4]}


def test_a_moving_wall_keeps_the_transfers_and_the_velocity_iterations():
    mesh = _wall_box()
    s = _wall_stokes(mesh)
    s.solve()
    before = _transfers(s)
    counts = [_velocity_iterations(s)]
    for _ in range(3):
        _raise_the_top_wall(mesh)
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(_velocity_iterations(s))
    assert _largest_change(before, _transfers(s)) == 0
    if uw.mpi.size in WALL_COUNTS:
        assert counts == WALL_COUNTS[uw.mpi.size]
        built_after = _wall_stokes(mesh)
        built_after.solve()
        assert _velocity_iterations(built_after) == 4


def _fault_stokes(mesh):
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    s.tolerance = 1.0e-6
    s.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    s.add_fault_bc(0, "Flt")
    return s


def _split_level_mesh():
    """A fault on an adapt child: the add-on levels are split."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=2)
    pts = np.column_stack([np.linspace(*TRACE[:, 0], 9), np.linspace(*TRACE[:, 1], 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface("guide_1176_parallel", base, pts)
    guide.discretize()
    child = base.adapt(guide.refinement_metric_function(
        h_near=0.05 / 4, h_far=0.05, width=0.08, profile="linear"), max_levels=2)
    mesh = child.add_fault([("Flt", TRACE)])
    assert fault_mg.is_split_level(mesh._custom_mg_coarse_meshes[-1])
    return mesh


# Velocity iterations at rest and after each of three moves, and those of a
# solver built on the moved mesh, by rank count. At 8 ranks the finest pair
# carries a FAC patch at rest and a move drops it (the rule of
# tests/parallel/test_1152_fac_zone_survives_a_mover_parallel.py), so the
# count rises there with the transfers unchanged.
SPLIT_COUNTS = {2: ([6, 7, 7, 7], 7), 3: ([6, 7, 7, 7], 7), 4: ([6, 7, 7, 7], 7),
                5: ([6, 7, 7, 7], 7), 8: ([3, 15, 15, 15], 15)}


def test_a_split_level_keeps_the_transfers_and_the_velocity_iterations():
    mesh = _split_level_mesh()
    s = _fault_stokes(mesh)
    s.solve()
    before = _transfers(s)
    counts = [max(s._rotated_freeslip_info["vel_its_last"])]
    for _ in range(3):
        _deform(mesh)
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(max(s._rotated_freeslip_info["vel_its_last"]))
    assert _largest_change(before, _transfers(s)) == 0
    if uw.mpi.size in SPLIT_COUNTS:
        moved, built_on_it = SPLIT_COUNTS[uw.mpi.size]
        assert counts == moved
        built_after = _fault_stokes(mesh)
        built_after.solve()
        assert max(built_after._rotated_freeslip_info["vel_its_last"]) == built_on_it


def test_a_moved_tetrahedral_level_keeps_its_transfers():
    """3 iterations at rest and after each of two moves, which the pair
    rebuilt from the moved coordinates takes as well; the transfers tell the
    two apart."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=1 / 3, regular=False,
        qdegree=2, refinement=1)
    x, y, z = mesh.X
    s = uw.systems.Poisson(mesh, degree=2)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1
    s.f = sympy.sin(3 * x) * sympy.cos(2 * y)
    for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, list(mesh.multigrid_levels[:-1]))
    s.solve()
    before = _transfers(s)
    counts = [s.snes.getLinearSolveIterations()]
    for _ in range(2):
        _deform(mesh, amplitude=0.03)
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(s.snes.getLinearSolveIterations())
    assert _largest_change(before, _transfers(s)) == 0
    assert counts == [3, 3, 3]
