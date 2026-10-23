"""Custom-P multigrid on more ranks than its coarsest level has cells.

A coarse level is small by construction, so on a large run most ranks own none
of its cells, and the fine mesh of a placed fault is gathered onto one rank, so
most ranks own none of the finest cells either. Found on kaiju at np 104 (a
124-cell coarsest level, the faulted annulus): ``_colocate_level`` shaped its
centroid arrays from the cells it found, an empty rank raised IndexError, and
the ranks that had cells waited in the next all-gather for ever.

Run with at least four ranks; the 1 x 1 box at cell size 1 has four cells.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=4), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


def _box(cellSize, refinement=None):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cellSize,
        regular=True, refinement=refinement, qdegree=2)


def _cells(mesh):
    start, end = mesh.dm.getHeightStratum(0)
    return end - start


def _wrap(dm, like):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=like.CoordinateSystem.coordinate_type,
        qdegree=2, boundaries=like.boundaries)


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


def test_colocating_a_level_with_empty_ranks():
    """Every rank returns, and the coarse level keeps all its cells."""
    base = _box(1.0, refinement=3)
    coarsest = base._coarse_level_meshes()[0]
    total = uw.mpi.comm.allreduce(_cells(coarsest))
    empty = uw.mpi.comm.allreduce(int(_cells(coarsest) == 0 and _cells(base) == 0))
    assert total < uw.mpi.size and empty > 0, (total, empty)    # the case under test

    moved = custom_mg._colocate_level(coarsest, base)
    assert uw.mpi.comm.allreduce(_cells(moved)) == total


def test_custom_fmg_with_a_coarsest_level_smaller_than_the_ranks():
    """The custom-P solve installs its multigrid and agrees with GAMG."""
    m0 = _box(1.0)
    dm1 = m0.dm.refine()
    dm2 = dm1.refine()
    dm3 = dm2.refine()
    assert uw.mpi.comm.allreduce(_cells(m0)) < uw.mpi.size      # the case under test

    reference = _poisson(_wrap(dm3, m0))
    reference.preconditioner = "gamg"
    reference.solve()

    solver = _poisson(_wrap(dm3, m0))
    custom_mg.set_custom_fmg(solver, [_wrap(m0.dm, m0), _wrap(dm1, m0), _wrap(dm2, m0)],
                             builder="barycentric")
    solver.solve()

    assert solver.snes.getKSP().getPC().getType() == "mg"
    assert solver.snes.getConvergedReason() > 0
    diff = np.asarray(solver.Unknowns.u.data) - np.asarray(reference.Unknowns.u.data)
    num = uw.mpi.comm.allreduce(float(np.sum(diff ** 2)))
    den = uw.mpi.comm.allreduce(float(np.sum(np.asarray(reference.Unknowns.u.data) ** 2)))
    assert (num / den) ** 0.5 < 1.0e-6



def test_cutting_a_line_with_empty_ranks():
    """The fault line cut runs on every rank and splits the edges it crosses."""
    from underworld3.utilities.line_cut import cut_along_lines
    mesh = _box(1.0)
    empty = uw.mpi.comm.allreduce(int(_cells(mesh) == 0))
    assert empty > 0, empty                                     # the case under test
    line = np.array([[-0.1, 0.55], [1.1, 0.55]])               # wall to wall
    cut, _info = cut_along_lines(mesh.dm, [line])
    start, end = cut.getHeightStratum(0)
    assert uw.mpi.comm.allreduce(end - start) > uw.mpi.comm.allreduce(_cells(mesh))

