"""Render the spherical sheet mesh used in the tidal study, with the four tiger stripes.

Usage: python tidal_mesh_fig.py <sim_dir> [h]   (h defaults to 0.08, the production cell size)

Two panels: the whole sphere viewed from the south-polar side with the sub-planet
point (+x) marked, and a close-up of the stripe set. The mesh is the split mesh the
driver uses (SphericalManifold + add_fault with the driver's default geometry);
the stripes are drawn as the great-circle arcs passed to the cut.
"""
import os
import sys

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
import underworld3 as uw
import underworld3.visualisation as vis

sim = sys.argv[1]
h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
os.makedirs(sim, exist_ok=True)

# the driver's default stripe geometry (tidal_shell.py: --faults 4 --fault_len 0.52
# --fault_spacing 0.14 --fault_lat -90 --fault_lon 0 --fault_azimuth -90)
n_faults, fault_len, spacing, lat0, lon0, azim = 4, 0.52, 0.14, -90.0, 0.0, -90.0


def stripe_points(offset, half_len, n=9):
    th0, ph0 = np.radians(lat0), np.radians(lon0)
    c = np.array([np.cos(th0) * np.cos(ph0), np.cos(th0) * np.sin(ph0), np.sin(th0)])
    east = np.array([-np.sin(ph0), np.cos(ph0), 0.0])
    sub = np.cross(c, east)
    az = np.radians(azim)
    e1 = np.cos(az) * sub + np.sin(az) * np.cross(c, sub)
    e2 = np.cross(c, e1)
    cp = np.cos(offset) * c + np.sin(offset) * e2
    a = np.linspace(-half_len, half_len, n)
    return np.array([np.cos(ai) * cp + np.sin(ai) * e1 for ai in a])


sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=h, degree=1, qdegree=5)
offsets = (np.arange(n_faults) - 0.5 * (n_faults - 1)) * spacing
faults = [(f"Stripe{i}", stripe_points(off, 0.5 * fault_len)) for i, off in enumerate(offsets)]
split = sphere.add_fault(faults)

pv_mesh = vis.mesh_to_pv_mesh(split)
edges = pv_mesh.extract_all_edges()
n_cells = pv_mesh.n_cells
n_pts = pv_mesh.n_points

# the realised cut: the facets carrying each stripe's label (the cut snaps existing
# mesh nodes onto the requested arc, so these are mesh edges; no refinement)
dm = split.dm
fS, fE = dm.getHeightStratum(1)
pS, pE = dm.getDepthStratum(0)
X = split.X.coords
lines = []
n_pairs = {}
for name, _ in faults:
    lab = dm.getLabel(name)
    segs = []
    for v in lab.getValueIS().getIndices():
        iss = dm.getStratumIS(name, v)
        if iss is None:
            continue
        for p in iss.getIndices():
            if fS <= p < fE:
                segs.append(dm.getCone(p) - pS)
    segs = np.array(segs)
    pts = X[segs.reshape(-1)] * 1.002
    cells = np.column_stack([np.full(len(segs), 2), np.arange(0, 2 * len(segs), 2), np.arange(1, 2 * len(segs), 2)])
    lines.append(pv.PolyData(pts, lines=cells.reshape(-1)))
    n_pairs[name] = len(split._fault_point_pairs[name])
subplanet = pv.PolyData(np.array([[1.0, 0.0, 0.0]]))
south = pv.PolyData(np.array([[0.0, 0.0, -1.0]]))

pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000), border=False)
for k in range(2):
    pl.subplot(0, k)
    pl.set_background("white")
    pl.add_mesh(pv_mesh, color="#f2f2f2", show_edges=False, lighting=False)
    pl.add_mesh(edges, color="#404040", line_width=0.8 if k == 0 else 1.4, lighting=False)
    for blk in lines:
        pl.add_mesh(blk, color="#c0392b", line_width=4 if k == 0 else 6, lighting=False, render_lines_as_tubes=True)
    pl.add_mesh(subplanet, color="#1f77b4", point_size=18, render_points_as_spheres=True, lighting=False)
    pl.add_mesh(south, color="#c0392b", point_size=10, render_points_as_spheres=True, lighting=False)
    pl.camera.parallel_projection = True
    if k == 0:
        # oblique: sub-planet point at the right, south pole toward the viewer
        pl.camera.position = (2.0, -1.5, -3.0)
        pl.camera.focal_point = (0.0, 0.0, 0.0)
        pl.camera.up = (1.0, 0.0, 0.0)
        pl.camera.parallel_scale = 1.15
        pl.add_point_labels(subplanet, ["sub-planet point (+x)"], font_size=22, text_color="#1f77b4",
                            point_color="#1f77b4", point_size=1, shape=None, always_visible=True)
        pl.add_text(f"SphericalManifold, h = {h}: {n_cells} triangles, {n_pts} nodes (split)",
                    position="upper_left", font_size=14, color="black")
    else:
        # polar close-up: looking up at the south pole, +x (sub-planet direction) to the top
        pl.camera.position = (0.0, 0.0, -4.0)
        pl.camera.focal_point = (0.0, 0.0, -1.0)
        pl.camera.up = (1.0, 0.0, 0.0)
        pl.camera.parallel_scale = 0.45
        pl.add_text("South polar region: four stripes, length 0.52, spacing 0.14 rad\n"
                    "(mesh nodes snapped onto the arcs and duplicated, no local refinement; +x toward the top)",
                    position="upper_left", font_size=14, color="black")

out = os.path.join(sim, f"mesh_h{h:g}.png")
pl.screenshot(out)
pl.close()
print("wrote", out, "cells", n_cells, "points", n_pts, "unsplit points", sphere.X.coords.shape[0], "pairs", n_pairs)
