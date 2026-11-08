"""Re-solving a rotated free-slip model whose body force is reassigned does not grow memory, in parallel (#1007).

A reassigned expression re-JITs the solver and rebuilds its DM, and ``dm.handle`` is in
the rotated workspace's geometry signature, so every such solve rebuilds the workspace:
the rotation, the operator and the custom-FMG prolongation. The old workspace's
prolongation list and the builders' intermediate matrices were dropped, not destroyed.
In parallel petsc4py defers each destroy to a collective ``PETSc.garbage_cleanup``,
which only a new mesh ran (test_0078), so each rebuild kept them: measured on
development 15f13d0c at np 2, peak RSS rose 805, 819, 832, 847, 860, 874 MB over six
solves, about 7 MB per rank per solve. The rebuild now empties the deferred list.

Run with at least two ranks.
"""
import gc
import sys

import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities import custom_mg, memprobe

# As in test_0078: the resident set tracks freed memory on Linux only.
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900),
              pytest.mark.skipif(sys.platform != "linux",
                                 reason="the resident set tracks freed memory on Linux only")]

WARMUP, MEASURED = 2, 5
BOUND_MB = 2.0               # per solve and rank; the leak was about 7


def _resident_mb():
    return memprobe.snapshot()["rss_mb"]


def test_reassigning_the_body_force_does_not_grow_memory():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2,
                              refinement=2, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    unit_r = sympy.Matrix([[x / r, y / r]])
    v = uw.discretisation.MeshVariable("V1144", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1144", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.add_rotated_freeslip_bc(0.0, "Lower")
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    stokes.tolerance = 1.0e-6

    def solve(k):
        stokes.bodyforce = (-1) ** k * sympy.cos(2 * th) * unit_r   # rebuilds the DM
        stokes.solve(zero_init_guess=True)
        info = stokes._rotated_freeslip_info
        assert info["converged"], info["rnorm_history"]
        if k:
            # the path under test: the workspace was rebuilt, not reused
            assert info["rotation_reused"] is False, k

    for k in range(WARMUP):
        solve(k)
    gc.collect()
    start = _resident_mb()
    for k in range(WARMUP, WARMUP + MEASURED):
        solve(k)
    gc.collect()
    growth = (_resident_mb() - start) / MEASURED
    worst = MPI.COMM_WORLD.allreduce(growth, op=MPI.MAX)
    assert worst < BOUND_MB, f"resident set grew {worst:.2f} MB per solve on the worst rank"
