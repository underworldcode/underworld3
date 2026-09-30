"""MPI test for the wall-clock guard (``systems/solver_health.py``).

Run via mpi_runner.sh (mpirun -np N python ptest_0203_*.py).

The parallel hazard is specific and severe: a PETSc convergence test that returns
different verdicts on different ranks leaves the ranks on different code paths, and the
next collective inside PETSc deadlocks. Wall clocks are not synchronised across ranks,
so "has the budget run out" is exactly the kind of question that would split them —
which is why the guard reduces the answer over the solver's communicator.

**This script completing at all is the main assertion.** A split verdict hangs rather
than fails; the runner's timeout is what catches it. The explicit assertions then check
that the ranks agree about what happened, and that a guarded solve with a generous
budget still produces the same answer as an unguarded one.
"""
import numpy as np
import sympy

import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.05, qdegree=3
)
v = uw.discretisation.MeshVariable("Up", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("Pp", mesh, 1, degree=1, continuous=True)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
x, y = mesh.X
stokes.bodyforce = sympy.Matrix(
    [0.0, -sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]
)
for wall in ("Top", "Bottom", "Left", "Right"):
    stokes.add_dirichlet_bc((0.0, 0.0), wall)
stokes.petsc_options["snes_error_if_not_converged"] = "false"

# Unguarded reference.
stokes.solve()
stokes.solve(zero_init_guess=True)
reference = np.asarray(v.array).copy()
assert stokes.solve_report.converged, f"rank {uw.mpi.rank}: reference solve failed"

# A generous budget must not disturb the answer on any rank.
stokes.guard(wall_per_step=600.0)
stokes.solve(zero_init_guess=True)
report = stokes.solve_report
assert report.converged, f"rank {uw.mpi.rank}: generous budget broke a healthy solve"
assert not report.deadline_expired
drift = float(np.abs(np.asarray(v.array) - reference).max())
assert drift < 1.0e-6, f"rank {uw.mpi.rank}: guarded answer drifted by {drift:.3e}"

# A budget that cannot be met. Every rank must reach the same verdict — if the reduction
# in deadline_passed() were dropped, the ranks whose clock had not yet expired would
# keep iterating while the others reported failure, and PETSc would deadlock here.
stokes.guard(wall_per_step=1.0e-6)
stokes.solve(zero_init_guess=True)
cut = stokes.solve_report
stokes.unguard()

assert cut.deadline_expired, f"rank {uw.mpi.rank}: the deadline never fired"
assert not cut.converged, f"rank {uw.mpi.rank}: a cut solve reported success"

verdicts = uw.mpi.comm.allgather((cut.deadline_expired, cut.converged, cut.reason))
assert len(set(verdicts)) == 1, f"ranks disagree about the guarded solve: {verdicts}"

# And the guard must come off cleanly on every rank.
stokes.solve(zero_init_guess=True)
assert stokes.solve_report.converged, f"rank {uw.mpi.rank}: unguard did not restore"


# --- the hazard, provoked deliberately ------------------------------------------------
# The checks above cannot show the reduction is doing anything: with a 1e-6 s budget
# every rank's clock expires at once, so a per-rank decision would agree by luck. Skew
# the clocks instead — rank 0's runs fast, the others' barely move — so the ranks WOULD
# reach opposite verdicts if each decided alone. This relies on every rank calling the
# convergence tests the same number of times, which holds because Krylov iterations are
# globally synchronised; that is what makes the reduction well-posed rather than a
# guess. Without the reduction this section deadlocks.
from underworld3.systems import solver_health


class _RankSkewedClock:
    def __init__(self, rank):
        self.now = 0.0
        self.tick = 10.0 if rank == 0 else 1.0e-9


    def monotonic(self):
        self.now += self.tick
        return self.now


solver_health.time = _RankSkewedClock(uw.mpi.rank)
stokes.guard(wall_per_step=50.0)          # only rank 0's clock can reach this
stokes.solve(zero_init_guess=True)
skewed = stokes.solve_report
stokes.unguard()

assert skewed.deadline_expired, (
    f"rank {uw.mpi.rank}: one rank's clock expired but this rank did not follow — "
    "the expiry decision was not reduced across the communicator"
)
skewed_verdicts = uw.mpi.comm.allgather((skewed.deadline_expired, skewed.reason))
assert len(set(skewed_verdicts)) == 1, (
    f"ranks disagree under skewed clocks: {skewed_verdicts}"
)

uw.pprint(f"ptest_0203: wall-clock guard consistent across {uw.mpi.size} ranks")
