"""Reusable diagnostics for adaptive-mesh convection runs.

This module collects the *measurement* tools that the adaptive-convection
driver needs, kept deliberately separate from the driver so they can be
imported by analysis scripts, tests, or other drivers (faults, free
surface) without dragging in a model setup.

Everything here is read-only on the mesh/fields — these helpers measure,
they never move nodes or solve.

Contents
--------
mesh_quality(mesh)        folded-element count + cell-area ratio + aspect
                          ratio + min area — the definitive "is the mesh OK"
                          check (NOT the render alone; see
                          feedback_debug_adaptive_solver_method).
nn_spacing_ratios(...)    nearest-neighbour spacing ratios of a feature
                          region (thermal boundary layer, and optionally a
                          fault corridor) to the bulk — the RIGHT measure of
                          achieved local refinement (misalignment is global
                          and useless for a feature that is a tiny fraction
                          of the nodes).
NusseltSurface(...)       surface heat-flux Nusselt number on a boundary.
vrms(mesh, V)             root-mean-square velocity (kinetic-energy proxy).
History                   per-step record accumulator with npz save/reload
                          (resume-aware).

Why these and not the render: a "looks fine" final frame can sit between
intermittent area-ratio spikes, and nodal v·n overstates a weak (Nitsche)
free-slip leak. Judge the mesh by folded/area-ratio and the physics by
vrms/Nu, at MATCHED PHYSICAL TIME between runs.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

import underworld3 as uw

# Reuse the mover's own triangle-connectivity + signed-area helpers so the
# quality metric is computed on exactly the same topology the mover sees.
# They are private but stable; fall back to a local implementation if they
# ever move.
try:
    from underworld3.meshing.smoothing import _tri_cells, _signed_areas
except Exception:  # pragma: no cover - defensive fallback
    def _tri_cells(dm):
        cStart, cEnd = dm.getHeightStratum(0)
        pStart, pEnd = dm.getDepthStratum(0)
        tris = []
        for c in range(cStart, cEnd):
            closure = dm.getTransitiveClosure(c)[0]
            vs = [p - pStart for p in closure if pStart <= p < pEnd]
            if len(vs) != 3:
                return None
            tris.append(vs)
        return np.asarray(tris, dtype=np.int64) if tris else None

    def _signed_areas(coords, tris):
        a, b, c = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


# ---------------------------------------------------------------------------
#  Mesh quality
# ---------------------------------------------------------------------------
def mesh_quality(mesh) -> dict:
    """Geometric health of the current (possibly adapted) 2D triangular mesh.

    Returns a dict with:
      folded      number of inverted cells (signed area sign-flips vs the
                  global orientation). The single most important number —
                  any value > 0 means the mover tangled the mesh.
      area_ratio  max(|A|) / min(|A|) over cells — the grading spread. On a
                  healthy adapted annulus this sits flat (~10-15); runaway
                  spikes (~100-900) are the "holes" failure mode.
      aspect      max over cells of longest-edge / shortest-edge — sliver
                  detector (mmpde stays equant, ~2-4; OT/anisotropic sliver
                  to 10+).
      min_area    smallest |A| (positivity / near-degenerate check).
      n_cells     triangle count.

    Returns folded=-1 (and NaNs) if the mesh is not all-triangle.
    """
    dm = mesh.dm
    tris = _tri_cells(dm)
    coords = np.asarray(mesh.X.coords)[:, :2]
    if tris is None:
        return dict(folded=-1, area_ratio=float("nan"),
                    aspect=float("nan"), min_area=float("nan"),
                    n_cells=0)
    sa = _signed_areas(coords, tris)
    orient = np.sign(np.median(sa)) or 1.0
    folded = int((sa * orient <= 0.0).sum())
    A = np.abs(sa)
    A_pos = A[A > 0]
    area_ratio = float(A.max() / A_pos.min()) if A_pos.size else float("inf")

    # longest/shortest edge per triangle → aspect proxy
    p = coords[tris]                                    # (n,3,2)
    e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    edges = np.stack([e0, e1, e2], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = edges.max(axis=1) / np.clip(edges.min(axis=1), 1e-30, None)
    return dict(
        folded=folded,
        area_ratio=area_ratio,
        aspect=float(np.nanmax(aspect)),
        min_area=float(A.min()),
        n_cells=int(tris.shape[0]),
    )


# ---------------------------------------------------------------------------
#  Achieved local refinement (nearest-neighbour spacing ratios)
# ---------------------------------------------------------------------------
def nn_spacing_ratios(
    mesh,
    *,
    r_inner: float = 0.5,
    r_outer: float = 1.0,
    bl_frac: float = 0.10,
    fault_polyline: Optional[np.ndarray] = None,
    fault_width: float = 0.04,
) -> dict:
    """Nearest-neighbour vertex-spacing ratios of feature regions vs bulk.

    A ratio < 1 means *finer than the bulk* (more refined). This is the
    correct refinement measure for a thin feature: global misalignment
    stays ~0.9 even with a feature left unrefined because the feature is a
    tiny fraction of the nodes.

    bl_ratio    median NN spacing in the outer thermal-boundary-layer shell
                (r > r_outer - bl_frac*(r_outer-r_inner)) / bulk.
    fault_ratio (only if fault_polyline given) median NN spacing within
                1.5*fault_width of the fault polyline / bulk.

    Bulk reference = mid-radius interior, away from any fault corridor.
    """
    from scipy.spatial import cKDTree

    C = np.asarray(mesh.X.coords)[:, :2]
    r = np.sqrt((C ** 2).sum(axis=1))
    dd, _ = cKDTree(C).query(C, k=2)
    nn = dd[:, 1]

    gap = r_outer - r_inner
    if fault_polyline is not None:
        from underworld3.utilities.geometry_tools import (
            signed_distance_pointcloud_polyline_2d as _sd)
        dist = np.abs(_sd(C, np.asarray(fault_polyline)[:, :2]))
        near = dist < (1.5 * fault_width)
    else:
        dist = np.full(len(C), 1e9)
        near = np.zeros(len(C), dtype=bool)

    bulk_mask = (dist > 0.3) & (r > r_inner + 0.1 * gap) & (r < r_outer - 0.15 * gap)
    if not bulk_mask.any():
        bulk_mask = dist > 0.3
    bulk = float(np.median(nn[bulk_mask])) if bulk_mask.any() else float("nan")

    bl_mask = r > (r_outer - bl_frac * gap)
    bl = float(np.median(nn[bl_mask])) if bl_mask.any() else float("nan")
    fin = np.isfinite(bulk) and bulk > 0
    out = dict(
        bulk_med=bulk,
        bl_med=bl,
        bl_ratio=(bl / bulk if fin else float("nan")),
        n_fault=int(near.sum()),
    )
    if fault_polyline is not None:
        fault_med = float(np.median(nn[near])) if near.any() else float("nan")
        out["fault_med"] = fault_med
        out["fault_ratio"] = (fault_med / bulk
                              if (near.any() and fin) else float("nan"))
    return out


# ---------------------------------------------------------------------------
#  Physics diagnostics
# ---------------------------------------------------------------------------
class NusseltSurface:
    """Surface-heat-flux Nusselt number on a boundary of an annulus.

    Nu = (boundary integral of -∂T/∂n) / Q_cond, where Q_cond is the
    purely-conductive heat flow through the shell,
    Q_cond = 2π / ln(r_outer / r_inner).

    The BdIntegral is built once and re-evaluated each step (it tracks the
    live T field and the deformed mesh).
    """

    def __init__(self, mesh, T, boundary_name, *,
                 r_inner: float = 0.5, r_outer: float = 1.0):
        self.q_cond = 2.0 * np.pi / np.log(r_outer / r_inner)
        X = mesh.CoordinateSystem.X
        n = mesh.Gamma_N
        qn = -(T.sym[0].diff(X[0]) * n[0] + T.sym[0].diff(X[1]) * n[1])
        self._bd = uw.maths.BdIntegral(mesh=mesh, fn=qn, boundary=boundary_name)

    def __call__(self) -> float:
        return float(self._bd.evaluate()) / self.q_cond


def vrms(mesh, V) -> float:
    """Root-mean-square velocity over the mesh vertices (KE proxy).

    Use vrms / kinetic energy as the real free-slip / convection-vigour
    indicator — NOT nodal v·n, which overstates a weak (Nitsche) leak.
    """
    v_sq = np.asarray(uw.function.evaluate(V.sym.dot(V.sym), mesh.X.coords))
    return float(np.sqrt(np.mean(v_sq)))


# ---------------------------------------------------------------------------
#  History accumulator
# ---------------------------------------------------------------------------
class History:
    """Per-step scalar record accumulator with npz persistence.

    Each ``record(**kwargs)`` appends one row; ``save(path)`` writes a flat
    npz of column arrays. Field set is fixed by the first record. Reload a
    previous run's npz with ``History.load(path)`` for resume.
    """

    def __init__(self, fields: Sequence[str]):
        self.fields = list(fields)
        self.rows: list[dict] = []

    def record(self, **kwargs):
        self.rows.append({k: kwargs.get(k, float("nan")) for k in self.fields})

    def save(self, path: str):
        if not self.rows:
            return
        cols = {k: np.asarray([row[k] for row in self.rows]) for k in self.fields}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **cols)

    @classmethod
    def load(cls, path: str) -> "History":
        z = np.load(path)
        h = cls(list(z.files))
        n = len(z[z.files[0]]) if z.files else 0
        for i in range(n):
            h.rows.append({k: float(z[k][i]) for k in z.files})
        return h

    def last(self, field: str, default=0.0):
        return self.rows[-1][field] if self.rows else default

    def column(self, field: str) -> np.ndarray:
        return np.asarray([row[field] for row in self.rows])
