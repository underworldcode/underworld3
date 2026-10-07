"""Assemble the residual and the Jacobian of one fixture at a given state, on the route
the environment selects (``UW_JIT_GRAPH=1`` or unset), and save them (#823, tier 2).

Two runs at the same state, one per route, are compared by
``route_assemble_compare.py``. The state is the SNES solution vector: ``zero`` (rest,
with the boundary values), or the ``X`` saved by ``route_ab.py``.
"""
import os
import sys

import numpy as np
import underworld3 as uw
from underworld3.utilities import _jit_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures   # noqa: E402

params = uw.Params(
    uw_fixture=uw.Param("box", description="box | powerlaw | linear | vep | ti | notch"),
    uw_tangent=uw.Param("newton", description="newton | picard | continuation"),
    uw_state=uw.Param("zero", description="zero, or a route_ab .npz whose X is the state"),
    uw_tag=uw.Param("zero", description="label for the output file"),
    uw_out=uw.Param("~/+Simulations/jit_graph/tier2/assemble", description="output directory"),
)
route = "graph" if _jit_graph.enabled() else "tree"
fixture, tangent = str(params.uw_fixture), str(params.uw_tangent)

stokes, _ = fixtures.build(fixture)
stokes.consistent_jacobian = {"newton": True, "picard": False,
                              "continuation": "continuation"}[tangent]
kwargs = fixtures.prepare_solve(stokes, fixture)
stokes.petsc_options["snes_max_it"] = 0
if "ksp_monitor" in stokes.petsc_options:
    stokes.petsc_options.delValue("ksp_monitor")
stokes.solve(**kwargs)

snes = stokes.snes
X = snes.getSolution()
state = str(params.uw_state)
if state == "zero":
    X.set(0.0)
else:
    X.array[:] = np.load(os.path.expanduser(state))["X"]
F = X.duplicate()
snes.computeFunction(X, F)
J, P = snes.getJacobian()[0], snes.getJacobian()[1]
snes.computeJacobian(X, J, P)
ai, aj, av = J.getValuesCSR()

out = os.path.expanduser(str(params.uw_out))
os.makedirs(out, exist_ok=True)
path = os.path.join(out, f"{fixture}_{tangent}_{params.uw_tag}_{route}.npz")
np.savez(path, F=np.asarray(F.array).copy(), ai=ai, aj=aj, av=av)
print(f"[{fixture} {tangent} {params.uw_tag} {route}] |F| {F.norm():.6e}  "
      f"nnz {len(av)}  saved {path}")
