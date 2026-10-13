"""Minimal reproducer for the multi-field / vector-field
`DMInterpolationEvaluate_UW` bug observed during Phase H pyvista
snapshots.

Three test phases:

  1. Single-field: mesh + one degree=2 vector ``u``. Assign constants
     ``u_x=7``, ``u_y=-3`` and evaluate at u's own DOF nodes. A correct
     interpolator returns the assigned constants exactly.
  2. Multi-field, fresh: same mesh + N extra degree=1 scalars assigned
     known constants. All variables should round-trip on their own DOFs.
  3. Save+load: write the multi-field state to disk, load into a fresh
     mesh with `read_timestep`, then re-evaluate. Tests whether the
     symptom severity depends on the load path.

A correct interpolator round-trips at machine precision on all three
phases. Use this script as a regression gate while fixing the
underlying bug in `_dminterp_wrapper.pyx` /
`MeshVariable.read_timestep`.

Usage:
    pixi run -e amr-dev python -u \
        docs/developer/design/experiments/dminterp-multifield/_repro_dminterp_multifield_bug.py
"""

import numpy as np
import sympy

import underworld3 as uw
from underworld3 import VarType


W = 1.0
H = 1.0
RES = 16


def case(n_extra_scalars):
    """Build a mesh with one vector (degree=2) + N extra scalars
    (degree=1), assign known constants, evaluate at DOF nodes."""
    print(f"\n=== n_extra_scalars = {n_extra_scalars} ===", flush=True)
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )

    label = f"R{n_extra_scalars}"
    u = uw.discretisation.MeshVariable(
        f"U_{label}", mesh, 2, degree=2, vtype=VarType.VECTOR,
    )
    extras = []
    for k in range(n_extra_scalars):
        s = uw.discretisation.MeshVariable(
            f"S{k}_{label}", mesh, 1, degree=1,
            continuous=True, vtype=VarType.SCALAR,
        )
        extras.append(s)

    # Assign distinctive constant values via the array setter
    u.array[:, 0, 0] = 7.0   # u_x = 7 at every node
    u.array[:, 0, 1] = -3.0  # u_y = -3 at every node
    u._sync_lvec_to_gvec()
    for k, s in enumerate(extras):
        s.array[:, 0, 0] = float(100 + k)  # 100, 101, 102, …
        s._sync_lvec_to_gvec()
    mesh._stale_lvec = True

    # Evaluate u at u's own DOF nodes — should round-trip 7, -3
    u_eval = np.asarray(uw.function.evaluate(u.sym, u.coords)).reshape(-1, 2)
    err_u = np.max(np.abs(u_eval - np.array([7.0, -3.0])))
    u_x_max = float(np.max(np.abs(u_eval[:, 0])))
    u_y_max = float(np.max(np.abs(u_eval[:, 1])))
    print(f"  u_x: assigned 7.0,    eval max|u_x|={u_x_max:.4e}", flush=True)
    print(f"  u_y: assigned -3.0,   eval max|u_y|={u_y_max:.4e}", flush=True)
    print(f"  u_eval err vs assigned: max={err_u:.4e}", flush=True)

    # Evaluate each scalar at its own DOF nodes
    for k, s in enumerate(extras):
        expected = float(100 + k)
        s_eval = np.asarray(uw.function.evaluate(s.sym, s.coords)).flatten()
        err = np.max(np.abs(s_eval - expected))
        print(f"  S{k}: assigned {expected}, eval max={s_eval.max():.4e}  "
              f"min={s_eval.min():.4e}  err={err:.4e}", flush=True)

    # Aggregate verdict
    bad = err_u > 1e-8
    for k, s in enumerate(extras):
        expected = float(100 + k)
        s_eval = np.asarray(uw.function.evaluate(s.sym, s.coords)).flatten()
        if np.max(np.abs(s_eval - expected)) > 1e-8:
            bad = True
    return bad


def case_load(n_extra_scalars):
    """Build, assign, write checkpoint, load into fresh mesh, evaluate.

    Tests whether the read_timestep path makes the bug worse (or
    triggers it where the fresh-build path doesn't).
    """
    import os
    OUT = "output/_repro_dminterp"
    os.makedirs(OUT, exist_ok=True)
    print(f"\n=== load test, n_extra_scalars = {n_extra_scalars} ===", flush=True)

    # ---- Phase 1: capture (write checkpoint) ----
    mesh_w = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )
    label_w = f"W{n_extra_scalars}"
    u_w = uw.discretisation.MeshVariable(
        f"U_{label_w}", mesh_w, 2, degree=2, vtype=VarType.VECTOR,
    )
    extras_w = []
    for k in range(n_extra_scalars):
        s = uw.discretisation.MeshVariable(
            f"S{k}_{label_w}", mesh_w, 1, degree=1,
            continuous=True, vtype=VarType.SCALAR,
        )
        extras_w.append(s)

    u_w.array[:, 0, 0] = 7.0
    u_w.array[:, 0, 1] = -3.0
    u_w._sync_lvec_to_gvec()
    for k, s in enumerate(extras_w):
        s.array[:, 0, 0] = float(100 + k)
        s._sync_lvec_to_gvec()
    mesh_w._stale_lvec = True

    key = f"repro_n{n_extra_scalars}"
    mesh_w.write_timestep(
        key, index=0, outputPath=OUT,
        meshVars=[u_w] + extras_w,
        create_xdmf=False,
    )

    # ---- Phase 2: load into fresh mesh with the same variable layout ----
    mesh_r = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )
    label_r = f"R{n_extra_scalars}"
    u_r = uw.discretisation.MeshVariable(
        f"U_{label_r}", mesh_r, 2, degree=2, vtype=VarType.VECTOR,
    )
    extras_r = []
    for k in range(n_extra_scalars):
        s = uw.discretisation.MeshVariable(
            f"S{k}_{label_r}", mesh_r, 1, degree=1,
            continuous=True, vtype=VarType.SCALAR,
        )
        extras_r.append(s)

    u_r.read_timestep(key, u_w.clean_name, index=0, outputPath=OUT)
    for k, s in enumerate(extras_r):
        s.read_timestep(key, extras_w[k].clean_name, index=0, outputPath=OUT)

    # Confirm raw arrays loaded correctly
    print(f"  u_r.array max|u_x|={float(np.max(np.abs(u_r.array[:, 0, 0]))):.4e}, "
          f"max|u_y|={float(np.max(np.abs(u_r.array[:, 0, 1]))):.4e}",
          flush=True)
    for k, s in enumerate(extras_r):
        print(f"  S{k}.array max={float(np.max(np.abs(s.array))):.4e}",
              flush=True)

    # Now evaluate each
    u_eval = np.asarray(uw.function.evaluate(u_r.sym, u_r.coords))
    u_eval = u_eval.reshape(-1, 2)
    err_u = np.max(np.abs(u_eval - np.array([7.0, -3.0])))
    print(f"  evaluate(u.sym, u.coords): err vs [7,-3] = {err_u:.4e}  "
          f"(col0 max={u_eval[:,0].max():.4e}  col1 max={u_eval[:,1].max():.4e})",
          flush=True)

    bad = err_u > 1e-8
    for k, s in enumerate(extras_r):
        expected = float(100 + k)
        s_eval = np.asarray(uw.function.evaluate(s.sym, s.coords)).flatten()
        err = np.max(np.abs(s_eval - expected))
        print(f"  evaluate(S{k}.sym, S{k}.coords): err vs {expected} = "
              f"{err:.4e}  (max={s_eval.max():.4e}  min={s_eval.min():.4e})",
              flush=True)
        if err > 1e-8:
            bad = True
    return bad


def main():
    print("Multi-field DMInterpolationEvaluate reproducer", flush=True)
    print("Tests `uw.function.evaluate(var.sym, var.coords)` after "
          "assigning known constants.", flush=True)

    fresh_bad = []
    for n in (0, 1, 4):
        if case(n):
            fresh_bad.append(n)

    print("\n--- with save + load ---", flush=True)
    load_bad = []
    for n in (0, 1, 4):
        if case_load(n):
            load_bad.append(n)

    print()
    print(f"Fresh-build bad: {fresh_bad}", flush=True)
    print(f"Save+load bad:   {load_bad}", flush=True)
    if not fresh_bad and not load_bad:
        print("All cases pass — bug not reproduced (or fixed).", flush=True)


if __name__ == "__main__":
    main()
