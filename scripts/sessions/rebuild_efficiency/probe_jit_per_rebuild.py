"""Does each faulted-mesh rebuild compile new JIT modules, and what does it cost?

The moving-fault fixture (as scripts/sessions/rebuild_leak), six rebuilds.
After each, the number of compiled modules held by the in-process JIT cache,
the time to build the solver (``_build``) and to solve, and whether the
generated sources differ between rebuilds (their hashes).

    python probe_jit_per_rebuild.py
"""
import gc
import time

import numpy as np

import underworld3 as uw
from underworld3.utilities import _jitextension as jit

params = uw.Params(uw_rebuilds=6, uw_depth=3)

# time every getext call (code generation plus cache lookup or compile)
_getext = jit.getext
getext_seconds = []


def _timed_getext(*args, **kwargs):
    start = time.perf_counter()
    result = _getext(*args, **kwargs)
    getext_seconds.append(time.perf_counter() - start)
    return result


jit.getext = _timed_getext
import underworld3.cython.generic_solvers as _solvers   # noqa: E402
_solvers.getext = _timed_getext                 # the solvers import it by name

HALF = np.array([0.20, 0.05])
base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
    refinement=1, qdegree=2)
seen = set()
for rebuild in range(int(params.uw_rebuilds)):
    centre = 0.35 + 0.05 * rebuild
    tip_a, tip_b = np.array([centre, 0.5]) - HALF, np.array([centre, 0.5]) + HALF
    samples = np.column_stack([np.linspace(tip_a[0], tip_b[0], 9),
                               np.linspace(tip_a[1], tip_b[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface(f"guide{rebuild}", base, samples)
    guide.discretize()
    metric = guide.refinement_metric_function(
        h_near=0.05 / 2 ** int(params.uw_depth), h_far=0.05, width=0.08,
        profile="linear")
    child = base.adapt(metric, max_levels=int(params.uw_depth))
    mesh = child.add_fault([("Flt", np.vstack([tip_a, tip_b]))])
    _x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, "Flt")
    start = time.perf_counter()
    stokes._build()
    build = time.perf_counter() - start
    start = time.perf_counter()
    stokes.solve()
    solve = time.perf_counter() - start
    modules = set(jit._ext_dict)
    new = modules - seen
    seen |= modules
    uw.pprint(f"REBUILD {rebuild}: build {build:.2f} s, solve {solve:.2f} s, "
              f"modules held {len(modules)}, new this rebuild {len(new)}, "
              f"getext calls {len(getext_seconds)} taking {sum(getext_seconds):.2f} s")
    getext_seconds.clear()
    del stokes, mesh, child, guide, metric, y
    gc.collect()
