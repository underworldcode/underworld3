"""Does boundary tangential slip relieve the volumetric spring's
touchy/anisotropic refinement? Volumetric spring, slip OFF vs ON,
on the localised INTERIOR blob (where the streakiness showed) and
the surface band. Plots a zoom for visual judgement.

SAFETY CHECK (user concern: nodes drifting off the surface): we
report max |r_final − r_orig| over boundary nodes — with the
per-ring radius projection this MUST be ~0 (slip is purely
tangential; the radial DOF is removed).
"""
from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sympy
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_spring, _edge_pairs, _auto_pinned_labels,
    _pinned_mask, _tri_cells, _signed_areas)

OUT = "/tmp/metric_mesh"
R_O, R_I, RES = 1.0, 0.5, 16
CX, CY, W = 0.78, 0.0, 0.12
AMPS = [8.0, 20.0]


def mesh_tris(m):
    dm = m.dm
    cS, cE = dm.getHeightStratum(0)
    pS, pE = dm.getDepthStratum(0)
    o = []
    for c in range(cS, cE):
        cl = dm.getTransitiveClosure(c)[0]
        vs = [p - pS for p in cl if pS <= p < pE]
        if len(vs) == 3:
            o.append(vs)
    return np.asarray(o, np.int64)


def far_near(coords, edges):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(coords[v1] - coords[v0], axis=1)
    nv = coords.shape[0]
    s = np.zeros(nv); cnt = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(cnt, a, 1.0)
    nl = s / np.maximum(cnt, 1.0)
    d = np.hypot(coords[:, 0] - CX, coords[:, 1] - CY)
    return float(nl[d > 4 * W].mean() / nl[d < W].mean())


def quality(coords, TRI):
    a = np.abs(_signed_areas(coords, TRI))
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
    return m, f, X0


def bnd_drift(m, X0):
    """max radial change of boundary nodes about the origin."""
    isb = _pinned_mask(m.dm, _auto_pinned_labels(m))
    c1 = np.asarray(m.X.coords)
    r0 = np.linalg.norm(X0[isb], axis=1)
    r1 = np.linalg.norm(c1[isb], axis=1)
    return float(np.abs(r1 - r0).max())


res = []
for amp in AMPS:
    row = []
    for slip in (False, True):
        m, f, X0 = case(amp, f"{int(amp)}{int(slip)}")
        e = _edge_pairs(m.dm); TRI = mesh_tris(m)
        p = _auto_pinned_labels(m)
        t = time.perf_counter()
        _winslow_spring(m, f, p, False, boundary_slip=slip)
        dt = time.perf_counter() - t
        c1 = np.asarray(m.X.coords).copy()
        print(f"AMP={amp:4.0f} slip={str(slip):5} "
              f"far/near={far_near(c1, e):.2f} "
              f"q={quality(c1, TRI):.3f} "
              f"bnd_drift={bnd_drift(m, X0):.2e} "
              f"{dt:.2f}s", flush=True)
        row.append((slip, c1, TRI))
    res.append((amp, row))

if uw.mpi.rank == 0:
    th = np.linspace(0, 2 * np.pi, 240)
    fig, ax = plt.subplots(2, 4, figsize=(24, 12),
                           facecolor="white")
    for ri, (amp, row) in enumerate(res):
        for ci, (slip, c1, TRI) in enumerate(row):
            # full
            a = ax[ri, 2 * ci]
            a.triplot(c1[:, 0], c1[:, 1], TRI, "k-", lw=0.5)
            a.plot(R_O * np.cos(th), R_O * np.sin(th),
                   "tab:red", lw=1.2)
            a.add_patch(plt.Circle((CX, CY), W, fill=False,
                        ec="tab:green", lw=2))
            a.set_title(f"AMP={amp:g} slip={slip} (full)",
                        fontsize=13)
            a.set_aspect("equal"); a.set_xticks([])
            a.set_yticks([])
            # zoom on blob
            a = ax[ri, 2 * ci + 1]
            a.triplot(c1[:, 0], c1[:, 1], TRI, "k-", lw=0.9)
            a.add_patch(plt.Circle((CX, CY), W, fill=False,
                        ec="tab:green", lw=2))
            a.set_xlim(CX - 3 * W, CX + 3 * W)
            a.set_ylim(CY - 3 * W, CY + 3 * W)
            a.set_title(f"AMP={amp:g} slip={slip} (blob zoom)",
                        fontsize=13)
            a.set_aspect("equal"); a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Volumetric spring: boundary slip OFF vs ON — "
                 "does tangential rim motion relieve the "
                 "anisotropy at the interior blob?", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pth = os.path.join(OUT, "slip.png")
    fig.savefig(pth, dpi=130, bbox_inches="tight")
    print("Saved", pth)
