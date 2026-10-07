"""How much of an assembly is the pointwise callbacks (#823, tier 2).

Times the residual and Jacobian assembly of a fixture at a saved state, then swaps
its constitutive law for a constant viscosity on the same mesh, fields and boundary
conditions and times the same assemblies again. The constant law's callbacks cost a
few nanoseconds, so its time is the assembly machinery (element loop, tabulation,
quadrature, insertion); the difference is the fixture's callbacks. Divided by the
number of quadrature points, it is the callbacks' cost per point. The route is chosen
by ``UW_JIT_GRAPH``.
"""
import os
import sys
import time

import numpy as np
import underworld3 as uw
from underworld3.utilities import _jit_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures   # noqa: E402

params = uw.Params(
    uw_fixture=uw.Param("notch", description="fixture"),
    uw_state=uw.Param("~/+Simulations/jit_graph/tier2/notch_newton_tree.npz",
                      description="route_ab .npz whose X is the state"),
    uw_repeat=uw.Param(20, description="assemblies timed"),
)
route = "graph" if _jit_graph.enabled() else "tree"
fixture = str(params.uw_fixture)
stokes, _ = fixtures.build(fixture)
kwargs = fixtures.prepare_solve(stokes, fixture)
stokes.petsc_options["snes_max_it"] = 0
X_state = np.load(os.path.expanduser(str(params.uw_state)))["X"]


def timed_assembly(label):
    stokes.solve(**kwargs)              # (re)builds the kernels; no iteration
    snes = stokes.snes
    X = snes.getSolution()
    X.array[:] = X_state
    F = X.duplicate()
    J, P = snes.getJacobian()[0], snes.getJacobian()[1]
    snes.computeFunction(X, F)
    snes.computeJacobian(X, J, P)
    n = int(params.uw_repeat)
    # the minimum of single assemblies: the least disturbed by other load
    t_res = t_jac = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        snes.computeFunction(X, F)
        t_res = min(t_res, time.perf_counter() - t)
    for _ in range(n):
        t = time.perf_counter()
        snes.computeJacobian(X, J, P)
        t_jac = min(t_jac, time.perf_counter() - t)
    print(f"[{fixture} {route} {label}] residual {1e3 * t_res:.2f} ms, "
          f"Jacobian {1e3 * t_jac:.2f} ms (minimum of {n}); Jacobian and "
          f"preconditioner {'one matrix' if J.handle == P.handle else 'two matrices'}")
    return t_res, t_jac


law = timed_assembly("law")

# quadrature points of the velocity block
dm = stokes.mesh.dm
c0, c1 = dm.getHeightStratum(0)
nq = len(stokes.dm.getField(0)[0].getQuadrature().getData()[1])
points = (c1 - c0) * nq

cm = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model = cm
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.is_setup = False
floor = timed_assembly("constant viscosity")

print(f"[{fixture} {route}] {c1 - c0} cells x {nq} quadrature points = {points}")
for name, a, b in (("residual", law[0], floor[0]), ("Jacobian", law[1], floor[1])):
    print(f"[{fixture} {route}] {name}: callbacks {1e3 * (a - b):.2f} ms of {1e3 * a:.2f} ms"
          f" ({100 * (a - b) / a:.0f}%), {1e9 * (a - b) / max(points, 1):.0f} ns per point per assembly")
