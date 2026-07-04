"""Render T (P3) using the proper sub-cell triangulation
instead of Delaunay over the DOF cloud.

Each P3 triangle has 10 DOFs arranged in a barycentric pattern.
We build 9 sub-triangles per cell explicitly, finding the global
DOF index for each sub-triangle vertex via the FE coordinate
lookup (DOF coord → global DOF index via kdtree).
"""
from __future__ import annotations
import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

import underworld3 as uw

SNAP = "output/convection_zoo_snapshots_rk4sl_res20"
OUT = "/tmp/p3_proper_res20.png"
STEPS = [8, 16, 24, 32, 40]
SCHEME = "rk4_sl"
T_NAME = "T_conv_v2p1"

pv.OFF_SCREEN = True


# P3 barycentric layout: indices 0..9 are at barycentric
# coordinates (a, b, c) where a + b + c = 1.
# Standard PETSc/UFC P3 lagrange node ordering:
#   3 vertex nodes:   (1,0,0), (0,1,0), (0,0,1)
#   6 edge nodes:     two per edge, at 1/3 and 2/3
#   1 interior node:  (1/3, 1/3, 1/3)
# We don't need to know PETSc's exact ordering — we look up each
# barycentric position via spatial matching to the DOF coords.
P3_BARY = np.array([
    [1.0, 0.0, 0.0],   # vertex 0
    [0.0, 1.0, 0.0],   # vertex 1
    [0.0, 0.0, 1.0],   # vertex 2
    [2/3, 1/3, 0.0],   # edge 01 third
    [1/3, 2/3, 0.0],   # edge 01 two-thirds
    [0.0, 2/3, 1/3],   # edge 12 third
    [0.0, 1/3, 2/3],   # edge 12 two-thirds
    [1/3, 0.0, 2/3],   # edge 20 third
    [2/3, 0.0, 1/3],   # edge 20 two-thirds
    [1/3, 1/3, 1/3],   # interior
])

# The 9 sub-triangles of a P3 cell, given as 10-index lists into
# the P3_BARY layout above.
P3_SUBTRIS = np.array([
    # Row at bottom edge (near vertex 0)
    [0, 3, 8],
    [3, 9, 8],
    [3, 4, 9],
    [4, 5, 9],
    [4, 1, 5],
    # Middle row
    [8, 9, 7],
    [9, 5, 6],
    [9, 6, 7],
    # Top
    [7, 6, 2],
])


def build_p3_subtriangulation(mesh, T):
    """For each cell, find the 10 P3 DOFs by their barycentric
    positions, then emit 9 sub-triangles per cell using DOF
    indices."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    coords = np.asarray(mesh.X.coords)
    t_coords = np.asarray(T.coords)
    tree = cKDTree(t_coords)

    tris = []
    for c in range(cStart, cEnd):
        closure, _ = dm.getTransitiveClosure(c, useCone=True)
        verts = [p - pStart for p in closure
                 if pStart <= p < pEnd]
        if len(verts) != 3:
            continue
        v_coords = coords[verts]
        # Compute the 10 P3 DOF positions for this cell.
        # bary @ v_coords gives (10, 2) array of physical positions.
        p3_pos = P3_BARY @ v_coords
        # Map to global DOF indices.
        _, dof_idx = tree.query(p3_pos, k=1)
        # Emit 9 sub-triangles using these 10 indices.
        for sub in P3_SUBTRIS:
            tris.append(dof_idx[sub])
    return np.asarray(tris, dtype=np.int64)


def render_step(snap_dir, label_short):
    mesh_path = f"{snap_dir}/uw_{SCHEME}_{label_short}.mesh.00000.h5"
    if not os.path.exists(mesh_path):
        return None, None
    mesh = uw.discretisation.Mesh(mesh_path)
    T = uw.discretisation.MeshVariable(
        T_NAME, mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(f"uw_{SCHEME}_{label_short}", T_NAME, 0,
                    outputPath=snap_dir)
    tris = build_p3_subtriangulation(mesh, T)
    # Build a pyvista PolyData from the DOF cloud + sub-triangles.
    points = np.zeros((T.coords.shape[0], 3))
    points[:, :2] = T.coords
    n_tri = tris.shape[0]
    faces = np.column_stack([
        np.full(n_tri, 3, dtype=np.int64), tris]).ravel()
    pv_T = pv.PolyData(points, faces=faces)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    return mesh, pv_T


import underworld3.visualisation as vis

PANEL = 1800   # px per panel — 4× the previous, ~30 px per cell
plotter = pv.Plotter(shape=(1, len(STEPS)),
                     window_size=(PANEL * len(STEPS), PANEL),
                     off_screen=True, border=False)
plotter.set_background("white")
plotter.disable_anti_aliasing()  # show actual per-DOF detail
for col, s in enumerate(STEPS):
    mesh, pv_T = render_step(SNAP, f"step{s:04d}")
    if pv_T is None:
        continue
    edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()
    # Sub-cell edges for the P3 sub-triangulation (orange, thin)
    # so we can SEE the resolution of the rendering.
    sub_edges = pv_T.extract_all_edges()
    plotter.subplot(0, col)
    plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                     clim=(0, 1), show_edges=False,
                     lighting=False, show_scalar_bar=False,
                     interpolate_before_map=False)
    plotter.add_mesh(sub_edges, color="orange",
                     line_width=0.3, opacity=0.5, lighting=False)
    plotter.add_mesh(edges, color="black", line_width=1.2,
                     lighting=False)
    plotter.add_text(f"step {s}", font_size=24, color="black",
                     position="upper_left")
    plotter.view_xy()
    plotter.camera.zoom(1.18)
plotter.screenshot(OUT)
plotter.close()
print(f"Saved {OUT}")
