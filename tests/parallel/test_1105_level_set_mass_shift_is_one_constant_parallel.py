"""The level set's mass correction shifts the field by one constant on every rank.

The correction finds the uniform shift that restores the enclosed volume, by a
secant iteration started from an estimate of the transition band's area. The
volumes are global integrals, but the band was estimated from each rank's own
nodes, so each rank started from its own guess and ended at its own shift: at
np 2 the far-field walls held 2.75e-4 on one rank and 3.33e-4 on the other,
against 3.13e-4 serially (measured in the review of #1211).

One SLCN step is taken without the correction, and the correction is then
applied to that field. Where neither field is clipped, the change is the
shift, and it must be the same number on every node of every rank.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import level_set

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]


def test_the_mass_correction_is_one_shift_on_every_rank():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 16, qdegree=3)
    x, y = mesh.X
    psi = uw.discretisation.MeshVariable("psi_shift", mesh, 1, degree=2)
    eps = level_set.interface_thickness(mesh, psi, scale=2.0)
    X = np.asarray(psi.coords)
    level_set.initialise_psi(psi, eps,
                             signed_distance=0.2 - np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.7))
    solver = uw.systems.LevelSetSolver(psi, velocity=sympy.Matrix([[-(y - 0.5), x - 0.5]]),
                                       epsilon=eps, advection="slcn", far_field=0.0,
                                       reini_steps=1, reini_frequency=5)
    solver.conserve_mass = False
    solver.solve(0.1)
    before = np.array(psi.array[:, 0, 0])
    solver._correct_mass(solver._target_volume)
    after = np.asarray(psi.array[:, 0, 0])

    unclipped = (before > 0.0) & (before < 1.0) & (after > 0.0) & (after < 1.0)
    shift = after[unclipped] - before[unclipped]
    comm = uw.mpi.comm
    lowest = comm.allreduce(float(shift.min()) if shift.size else np.inf, op=uw.MPI.MIN)
    highest = comm.allreduce(float(shift.max()) if shift.size else -np.inf, op=uw.MPI.MAX)
    assert abs(lowest) > 1e-8                       # the correction did something
    assert highest - lowest < 1e-14 * max(abs(lowest), 1.0)


def test_slcn_steps_with_the_clip_and_the_wall_correction_complete():
    # The clip's early return was decided per rank and skipped the collective
    # volume integral on a rank with nothing to clip, and the wall correction
    # skipped its collective write on a rank with no wall node: at np 2 and 3
    # the step hung (review of #1234). The timeout turns a hang into a failure.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 16, qdegree=3)
    x, y = mesh.X
    psi = uw.discretisation.MeshVariable("psi_steps", mesh, 1, degree=2)
    eps = level_set.interface_thickness(mesh, psi, scale=2.0)
    X = np.asarray(psi.coords)
    level_set.initialise_psi(psi, eps,
                             signed_distance=0.15 - np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.75))
    solver = uw.systems.LevelSetSolver(psi, velocity=sympy.Matrix([[-(y - 0.5), x - 0.5]]),
                                       epsilon=eps, advection="slcn",
                                       adv_solver_bc=("Left", "Right", "Top", "Bottom"),
                                       reini_steps=1, reini_frequency=2)
    for _ in range(4):
        solver.solve(0.05)
    # The correction stops once the volume is within its absolute tolerance.
    assert abs(solver.volume_drift) * solver._target_volume <= solver._mass_correction_tol
