"""Who keeps a dropped faulted mesh alive?

One rebuild of the moving-fault fixture (as in probe_rebuild_memory.py), a
solve, then every reference dropped. Weak references to the finest mesh, its
add-on child and the solver say what survived; for each survivor the chain of
referrers is walked back (breadth first, skipping frames and this script's
own containers) until it reaches a module, a class or a type, and the chain
is printed: the owner of the leak is the first library object on it.

    python probe_who_holds.py
"""
import gc
import sys
import types
import weakref

import numpy as np

import underworld3 as uw

params = uw.Params(uw_depth=3, uw_solve=1, uw_max_depth=12)

HALF = np.array([0.20, 0.05])
base = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
    refinement=1, qdegree=2)


def build_and_drop():
    """One rebuild, optionally a solve; return weak references only."""
    tip_a, tip_b = np.array([0.5, 0.5]) - HALF, np.array([0.5, 0.5]) + HALF
    samples = np.column_stack([np.linspace(tip_a[0], tip_b[0], 9),
                               np.linspace(tip_a[1], tip_b[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface("guide", base, samples)
    guide.discretize()
    metric = guide.refinement_metric_function(
        h_near=0.05 / 2 ** int(params.uw_depth), h_far=0.05, width=0.08,
        profile="linear")
    child = base.adapt(metric, max_levels=int(params.uw_depth))
    mesh = child.add_fault([("Flt", np.vstack([tip_a, tip_b]))])
    refs = {"finest": weakref.ref(mesh), "child": weakref.ref(child)}
    if int(params.uw_solve):
        _x, y = mesh.X
        stokes = uw.systems.Stokes(mesh)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.petsc_use_pressure_nullspace = True
        for wall in ("Top", "Bottom", "Left", "Right"):
            stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
        stokes.add_fault_bc(0, "Flt")
        stokes.solve()
        refs["solver"] = weakref.ref(stokes)
    return refs


def describe(o):
    """A short name for an object on a referrer chain."""
    if isinstance(o, dict):
        keys = list(o)[:6]
        return f"dict(len {len(o)}, keys {keys})"
    if isinstance(o, (list, tuple, set)):
        return f"{type(o).__name__}(len {len(o)})"
    return f"{type(o).__module__}.{type(o).__qualname__}"


def chain_to_root(target):
    """Breadth-first walk from ``target`` back to a module, class or type."""
    ignore = {id(sys._getframe())}
    seen = {id(target)}
    frontier = [(target, [describe(target)])]
    for _depth in range(int(params.uw_max_depth)):
        nxt = []
        for obj, path in frontier:
            for ref in gc.get_referrers(obj):
                if id(ref) in seen or isinstance(ref, types.FrameType):
                    continue
                if ref is frontier or ref is nxt or id(ref) in ignore:
                    continue
                seen.add(id(ref))
                step = path + [describe(ref)]
                if isinstance(ref, (types.ModuleType, type)):
                    return step
                if isinstance(ref, dict) and "__name__" in ref \
                        and "__builtins__" in ref:
                    return step + [f"(globals of {ref['__name__']})"]
                nxt.append((ref, step))
        frontier = nxt
    return None


def report(refs):
    """Alive or freed, and for each survivor the chain to its root."""
    for name, ref in refs.items():
        if ref() is None:
            print(f"HOLD {name}: freed")
            continue
        print(f"HOLD {name}: ALIVE")
        path = chain_to_root(ref())
        print("CHAIN " + (" <- ".join(path) if path else "no root found"))


refs = build_and_drop()
gc.collect()
report(refs)
