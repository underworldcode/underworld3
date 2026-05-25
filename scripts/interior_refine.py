"""The SENSIBLE test: refine around a LOCALISED INTERIOR region
(a Gaussian blob away from both boundary rings) — what we actually
want a metric smoother to do. Interior nodes are free to
redistribute and the boundary node-count is NOT the binding
constraint here (unlike the thin surface band).

Lagrangian feature: an initial-position vector field X0v is set
ONCE; metric = 1 + AMP·exp(-|X0v-c|²/W²). It tracks the material
feature through any deformation.

Methods: volumetric spring (the fast winner) vs MA. Reported:
  - local edge ratio  far / near-blob   (>1 ⇒ refined at the blob)
  - mesh quality minA/meanA
  - wall time
Mesh pictures: /tmp/metric_mesh/interior.png  (blob centre marked)
"""
from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_spring, _winslow_elliptic, _edge_pairs,
    _auto_pinned_labels, _tri_cells, _signed_areas)

OUT = "/tmp/metric_mesh"
os.makedirs(OUT, exist_ok=True)
R_O, R_I, RES = 1.0, 0.5, 16
CX, CY, W = 0.78, 0.0, 0.12          # interior blob centre/width
AMPS = [8.0, 20.0]


def mesh_tris(m):
    dm = m.dm
    cS, cE = dm.getHeightStratum(0)
    pS, pE = dm.getDepthStratum(0)
    out = []
    for c in range(cS, cE):
        cl = dm.getTransitiveClosure(c)[0]
        vs = [p - pS for p in cl if pS <= p < pE]
        if len(vs) == 3:
            out.append(vs)
    return np.asarray(out, np.int64)


def local_ratio(coords, edges):
    """mean incident edge length: far-field / near-blob (final
    position). >1 ⇒ smaller cells at the blob (refined)."""
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(coords[v1] - coords[v0], axis=1)
    nv = coords.shape[0]
    s = np.zeros(nv); cnt = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(cnt, a, 1.0)
    nl = s / np.maximum(cnt, 1.0)
    d = np.sqrt((coords[:, 0] - CX) ** 2 + (coords[:, 1] - CY) ** 2)
    near = d < W
    far = d > 4 * W
    if not near.any() or not far.any():
        return float("nan")
    return float(nl[far].mean() / nl[near].mean())


def quality(m):
    t = _tri_cells(m.dm)
    a = np.abs(_signed_areas(np.asarray(m.X.coords), t))
    return float(a.min() / a.mean())


def case(amp, tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    X0v = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    X0v.data[:, 0] = X0[:, 0]
    X0v.data[:, 1] = X0[:, 1]
    d2 = (X0v.sym[0] - CX) ** 2 + (X0v.sym[1] - CY) ** 2
    f = 1.0 + amp * sympy.exp(-d2 / W ** 2)
    return m, f


res = []
for k, amp in enumerate(AMPS):
    m, f = case(amp, f"s{k}")
    e = _edge_pairs(m.dm); TRI = mesh_tris(m)
    p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_spring(m, f, p, False)
    ts = time.perf_counter() - t
    cs = np.asarray(m.X.coords).copy()
    print(f"[vol-spring] AMP={amp:4.0f}  far/near="
          f"{local_ratio(cs, e):.2f}  q={quality(m):.3f}  "
          f"{ts:.2f}s", flush=True)

    m, f = case(amp, f"m{k}")
    e = _edge_pairs(m.dm)
    p = _auto_pinned_labels(m)
    t = time.perf_counter()
    _winslow_elliptic(m, f, p, False)
    tm = time.perf_counter() - t
    cm = np.asarray(m.X.coords).copy()
    print(f"[MA]         AMP={amp:4.0f}  far/near="
          f"{local_ratio(cm, e):.2f}  q={quality(m):.3f}  "
          f"{tm:.2f}s", flush=True)
    res.append((amp, cs, cm, TRI))

if uw.mpi.rank == 0:
    th = np.linspace(0, 2 * np.pi, 240)
    fig, ax = plt.subplots(2, len(AMPS),
                           figsize=(7.2 * len(AMPS), 14),
                           facecolor="white")
    for ci, (amp, cs, cm, TRI) in enumerate(res):
        for ri, (cc, lab) in enumerate(
                ((cs, "VOL-SPRING"), (cm, "MA"))):
            a = ax[ri, ci]
            a.triplot(cc[:, 0], cc[:, 1], TRI,
                      color="black", lw=0.6)
            a.plot(R_O * np.cos(th), R_O * np.sin(th),
                    "tab:red", lw=1.2)
            a.plot(R_I * np.cos(th), R_I * np.sin(th),
                    "tab:blue", lw=1.2)
            a.add_patch(plt.Circle((CX, CY), W, fill=False,
                        ec="tab:green", lw=2.0))
            a.set_title(f"{lab}  AMP={amp:g}", fontsize=14)
            a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Localised INTERIOR refinement (green = blob, "
                 "metric peak). Cells should shrink inside it.",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pth = os.path.join(OUT, "interior.png")
    fig.savefig(pth, dpi=130, bbox_inches="tight")
    print("Saved", pth)
