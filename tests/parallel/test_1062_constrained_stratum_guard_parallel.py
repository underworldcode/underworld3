"""Parallel regression for the empty-stratum guard in
``SNES_Stokes_Constrained._constrain_interior_multipliers_in_section``
(issue #291, PR #318).

Before #318, ``Stokes_Constrained`` segfaulted at np>1 in the interior-
multiplier section reduction: on a rank owning zero points with a given
boundary label value, ``dm.getLabel("UW_Boundaries").getStratumIS(bvalue)``
returned a valid non-None PETSc IS with ``getSize() == 0``, the existing
``if bd_is is not None`` guard passed through, and the following
``bd_is.getIndices()`` crashed with SIGSEGV.

The canonical ``tests/test_1062_constrained_solcx.py`` uses the SolCx
eta_B=1e6 benchmark which is far too slow to serve as a fast np≥2
regression check (Constrained is ~34× slower than Nitsche per #244).
This test uses the same solver family but at **unit viscosity** and a
16×16 grid, so both the guard and a bit-identical parallel-vs-serial
diagnostic land in ~10 seconds at np=2.

Run:
    mpirun -n 2 python -m pytest --with-mpi \\
        tests/parallel/test_1062_constrained_stratum_guard_parallel.py
    mpirun -n 4 python -m pytest --with-mpi \\
        tests/parallel/test_1062_constrained_stratum_guard_parallel.py
"""
import numpy as np
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]

# Serial reference (unit viscosity, sinusoidal body force, 16×16 quad box, free-
# slip on all four walls, tol 1e-8). Re-derive with `python <thisfile>`.
GOLDEN_U_INF = 2.533050e-2


def _solve():
    """Build + solve the constrained SolCx-lite problem; return |u|_inf."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3)
    s = uw.systems.Stokes_Constrained(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    s.bodyforce = sympy.Matrix(
        [[0.0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]])
    # Free-slip on all four walls via in-saddle multipliers. On a 2-way axis-
    # aligned split at np=2 this puts one axis-aligned pair (Left/Right or
    # Bottom/Top) with all points on one rank and zero on the other — the
    # exact configuration that triggered #291's SIGSEGV.
    s.add_constraint_bc(0.0, "Left",   normal=sympy.Matrix([[-1.0, 0.0]]))
    s.add_constraint_bc(0.0, "Right",  normal=sympy.Matrix([[ 1.0, 0.0]]))
    s.add_constraint_bc(0.0, "Bottom", normal=sympy.Matrix([[ 0.0,-1.0]]))
    s.add_constraint_bc(0.0, "Top",    normal=sympy.Matrix([[ 0.0, 1.0]]))
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-8
    s.solve()
    return float(np.max(np.abs(s.Unknowns.u.data)))


def test_constrained_parallel_no_segfault():
    """Pre-#318 this test would SIGSEGV during solve() at np>1. Post-#318 it
    solves and reproduces the serial |u|_inf to a tight tolerance."""
    u_inf = _solve()
    assert np.isclose(u_inf, GOLDEN_U_INF, rtol=1e-3, atol=0), (
        f"|u|_inf differs serial vs np={uw.mpi.size}: "
        f"golden={GOLDEN_U_INF:.6e} got={u_inf:.6e}")


if __name__ == "__main__":
    # Recompute the serial GOLDEN: `python <thisfile>`.
    u_inf = _solve()
    if uw.mpi.rank == 0:
        print(f"DIAG |u|_inf = {u_inf:.6e}")
