"""Visual + honest-metric comparison of the metric-driven mesh
grading on an UNDEFORMED uniform Annulus.

Two methods on identical setups, side by side:
  row 1 — elastic-spring equilibrium  (smooth_mesh_interior default)
  row 2 — Monge–Ampère / BFO          (_winslow_elliptic, preserved)
columns — AMP = 0, 2, 8, 20  in  f(r0)=1+AMP·exp(-((r0-R_O)/W)^2)

r0 is a degree-1 scalar set ONCE to the initial radius (Lagrangian).

The grading number printed/annotated is the HONEST metric:
per-node mean incident edge length binned by the node's FINAL
radius, deep/near. (The old centroid-band edge_ratio averaged the
thin strong near-surface compression with the bulk Lagrangian
shift and understated grading by ~40% — that was a validation
bug, not a method failure.)

Outputs:
  /tmp/metric_mesh/meshes.png   (mesh pictures — judge visually)
  /tmp/metric_mesh/case_*.npz
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs)

OUT = "/tmp/metric_mesh"
os.makedirs(OUT, exist_ok=True)

R_O, R_I = 1.0, 0.5
RES = 16
WIDTH = 0.12
AMPS = [0.0, 2.0, 8.0, 20.0]


def mesh_triangles(m):
    dm = m.dm
    cS, cE = dm.getHeightStratum(0)
    pS, pE = dm.getDepthStratum(0)
    tris = []
    for c in range(cS, cE):
        cl = dm.getTransitiveClosure(c)[0]
        vs = [p - pS for p in cl if pS <= p < pE]
        if len(vs) == 3:
            tris.append(vs)
    return np.asarray(tris, dtype=np.int64)


def honest_ratio(coords, edges):
    """deep/near ratio of per-node mean incident edge length,
    binned by each node's FINAL radius. ~1 = no grading;
    >1 = refined near surface (the design intent)."""
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(coords[v1] - coords[v0], axis=1)
    nv = coords.shape[0]
    s = np.zeros(nv)
    c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le)
        np.add.at(c, a, 1.0)
    nodelen = s / np.maximum(c, 1.0)
    r = np.sqrt((coords ** 2).sum(axis=1))
    deep = (r >= R_I) & (r < R_I + 0.20)            # r∈[0.50,0.70)
    near = (r > R_O - 0.05) & (r <= R_O + 1e-9)     # r∈(0.95,1.00]
    if not deep.any() or not near.any():
        return float("nan")
    return float(nodelen[deep].mean() / nodelen[near].mean())


def build_case(amp, idx):
    mesh = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                              cellSize=1.0 / RES, qdegree=3)
    TRI = mesh_triangles(mesh)
    c0 = np.asarray(mesh.X.coords).copy()
    r0 = uw.discretisation.MeshVariable(
        f"r0_{idx}", mesh, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(mesh.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + amp * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return mesh, TRI, c0, f


results = {"spring": [], "ma": []}
for k, amp in enumerate(AMPS):
    # --- elastic-spring (public API default metric path) ---
    mesh, TRI, c0, f = build_case(amp, f"sp{k}")
    edges = _edge_pairs(mesh.dm)
    uw.meshing.smooth_mesh_interior(mesh, metric=f, verbose=False)
    c1 = np.asarray(mesh.X.coords).copy()
    rsp = honest_ratio(c1, edges)
    r0sp = honest_ratio(c0, edges)
    results["spring"].append((amp, c0, c1, TRI, r0sp, rsp))
    np.savez(os.path.join(OUT, f"case_spring_amp{int(amp)}.npz"),
             coords0=c0, coords1=c1, tri=TRI)
    uw.pprint(f"[spring] AMP={amp:5.1f}  honest deep/near "
              f"{r0sp:.2f} -> {rsp:.2f}")

    # --- Monge–Ampère / BFO (preserved, called directly) ---
    mesh, TRI, c0, f = build_case(amp, f"ma{k}")
    edges = _edge_pairs(mesh.dm)
    pinned = uw.meshing.smoothing._auto_pinned_labels(mesh)
    _winslow_elliptic(mesh, f, pinned, False)
    c1 = np.asarray(mesh.X.coords).copy()
    rma = honest_ratio(c1, edges)
    r0ma = honest_ratio(c0, edges)
    results["ma"].append((amp, c0, c1, TRI, r0ma, rma))
    np.savez(os.path.join(OUT, f"case_ma_amp{int(amp)}.npz"),
             coords0=c0, coords1=c1, tri=TRI)
    uw.pprint(f"[MA]     AMP={amp:5.1f}  honest deep/near "
              f"{r0ma:.2f} -> {rma:.2f}")

if uw.mpi.rank == 0:
    n = len(AMPS)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.4),
                             facecolor="white")
    th = np.linspace(0, 2 * np.pi, 240)
    rows = [("Elastic-spring equilibrium", results["spring"]),
            ("Monge–Ampère (BFO, preserved)", results["ma"])]
    for ri, (label, rows_data) in enumerate(rows):
        for ci, (amp, c0, c1, TRI, r0v, r1v) in enumerate(
                rows_data):
            ax = axes[ri, ci]
            ax.set_facecolor("white")
            ax.triplot(c1[:, 0], c1[:, 1], TRI,
                       color="black", lw=0.4)
            ax.plot(R_O * np.cos(th), R_O * np.sin(th),
                    color="tab:red", lw=1.0)
            ax.plot(R_I * np.cos(th), R_I * np.sin(th),
                    color="tab:blue", lw=1.0)
            ttl = ("uniform (AMP=0)" if amp == 0
                   else f"AMP={amp:g}")
            ax.set_title(f"{ttl}\nhonest deep/near = {r1v:.2f}",
                         fontsize=10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[ri, 0].set_ylabel(label, fontsize=11)
    fig.suptitle(
        f"Metric-driven grading, undeformed Annulus (res={RES})  "
        f"f(r0)=1+AMP·exp(-((r0-{R_O:g})/{WIDTH:g})^2)\n"
        f"honest metric = per-node mean incident edge by FINAL "
        f"radius, deep/near  (1.0 = none; exact OT ≈ 10 at AMP=8)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(OUT, "meshes.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nSaved {out_png}")
