"""End-to-end A/B of the two JIT routes on one fixture (#823, tier 2).

The route is chosen by the environment, so one build serves both: ``UW_JIT_GRAPH=1`` is
the graph route, unset is the expanded tree (tier 1). Run once per route, then
``route_ab_compare.py`` on the two ``.npz`` files.

Records the whole pointwise setup (generation and compile separately), the size of the
generated header, the Newton solve (SNES reason, nonlinear and linear iterations, the
residual history), and the time of repeated residual and Jacobian assemblies at the
converged state. Run with ``UW_JIT_CACHE=0 UW_NO_USAGE_METRICS=1``.
"""
import hashlib
import os
import sys
import time

import numpy as np
import underworld3 as uw
import underworld3.utilities._jitextension as jx
from underworld3.utilities import _jit_graph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures   # noqa: E402

params = uw.Params(
    uw_fixture=uw.Param("box", description="box | powerlaw | linear | vep | ti | notch"),
    uw_tangent=uw.Param("newton", description="newton | picard | continuation"),
    uw_maxit=uw.Param(30, description="SNES iteration limit"),
    uw_tol=uw.Param(1.0e-8, description="SNES relative tolerance"),
    uw_assemble=uw.Param(20, description="residual and Jacobian assemblies to time"),
    uw_out=uw.Param("~/+Simulations/jit_graph/tier2", description="output directory"),
)
route = "graph" if _jit_graph.enabled() else "tree"
fixture = str(params.uw_fixture)
tangent = str(params.uw_tangent)

stokes, _ = fixtures.build(fixture)
stokes.consistent_jacobian = {"newton": True, "picard": False,
                              "continuation": "continuation"}[tangent]
solve_kwargs = fixtures.prepare_solve(stokes, fixture)
stokes.tolerance = float(params.uw_tol)
stokes.petsc_options["snes_max_it"] = int(params.uw_maxit)
if "ksp_monitor" in stokes.petsc_options:
    stokes.petsc_options.delValue("ksp_monitor")

seen = {"generate": 0.0, "compile": 0.0, "pointwise": 0.0, "n_pointwise": 0,
        "header bytes": 0}
generate, compile_ = jx.generate_c_source, jx.compile_and_load


def timed_generate(*a, **k):
    t = time.perf_counter()
    modname, codeguys, diag = generate(*a, **k)
    seen["generate"] += time.perf_counter() - t
    header = dict(codeguys)["cy_ext.h"]
    header = header.replace(modname, "__MOD__").replace(diag["randstr"], "__RS__")
    seen["header md5"] = hashlib.md5(header.encode()).hexdigest()[:10]
    seen["header bytes"] += len(header)
    return modname, codeguys, diag


def timed_compile(*a, **k):
    t = time.perf_counter()
    out = compile_(*a, **k)
    seen["compile"] += time.perf_counter() - t
    return out


jx.generate_c_source, jx.compile_and_load = timed_generate, timed_compile
pointwise = type(stokes)._setup_pointwise_functions


def timed_pointwise(self, *a, **k):
    t = time.perf_counter()
    out = pointwise(self, *a, **k)
    seen["pointwise"] += time.perf_counter() - t
    seen["n_pointwise"] += 1
    return out


type(stokes)._setup_pointwise_functions = timed_pointwise

t = time.perf_counter()
stokes.solve(**solve_kwargs)
wall = time.perf_counter() - t
r = stokes.solve_report

# repeated assemblies at the converged state
snes = stokes.snes
X = snes.getSolution()
F = X.duplicate()
J, P = snes.getJacobian()[0], snes.getJacobian()[1]
n = int(params.uw_assemble)
snes.computeFunction(X, F)
t = time.perf_counter()
for _ in range(n):
    snes.computeFunction(X, F)
t_res = (time.perf_counter() - t) / n
snes.computeJacobian(X, J, P)
t = time.perf_counter()
for _ in range(n):
    snes.computeJacobian(X, J, P)
t_jac = (time.perf_counter() - t) / n

print(f"[{fixture} {tangent} {route}] pointwise setup {seen['pointwise']:.2f} s "
      f"({seen['n_pointwise']} call) = generate {seen['generate']:.2f} s + compile "
      f"{seen['compile']:.2f} s + other; header {seen['header bytes']:,} B "
      f"md5 {seen.get('header md5')}")
print(f"[{fixture} {tangent} {route}] solve {wall:.2f} s: {r.reason_str} nl={r.nl_its} "
      f"ksp={r.ksp_its} fnorm={r.fnorm:.3e}")
print(f"[{fixture} {tangent} {route}] assembly at the solution: residual {1e3 * t_res:.2f} ms, "
      f"Jacobian {1e3 * t_jac:.2f} ms (mean of {n})")

out = os.path.expanduser(str(params.uw_out))
os.makedirs(out, exist_ok=True)
np.savez(os.path.join(out, f"{fixture}_{tangent}_{route}.npz"),
         v=np.asarray(stokes.u.array), p=np.asarray(stokes.p.array),
         X=np.asarray(X.array).copy(),
         history=np.asarray(r.history, dtype=float),
         reason=r.reason_str, nl=r.nl_its, ksp=r.ksp_its,
         pointwise=seen["pointwise"], generate=seen["generate"], compile=seen["compile"],
         header_bytes=seen["header bytes"], solve=wall, t_res=t_res, t_jac=t_jac)
