"""3-D split components from the REAL mesh (pyvista; data-heavy, not cetz).

Thesis: in 3-D the split doubles the interior of the PATCH — faces, edges
and vertices — while the RIM (the patch boundary) stays single, the tip
rule one dimension up. Rendered from an actual BoxInternalPatch +
split_fault mesh: the two coincident face sets pulled apart along the
normal for display, the shared rim drawn once, in black.

Run inside the fault-split-node worktree env.
"""
import os

import numpy as np
import pyvista as pv

import underworld3 as uw
from underworld3.utilities.fault_split import split_fault

pv.OFF_SCREEN = True
D = os.path.dirname(os.path.abspath(__file__))

PATCH = np.array([[0.5, 0.30, 0.30], [0.5, 0.70, 0.30],
                  [0.5, 0.70, 0.70], [0.5, 0.30, 0.70]])
mesh = uw.meshing.BoxInternalPatch(cellSize=0.12, patch_points=PATCH,
                                   patch_name="FltA")
child = split_fault(mesh, "FltA")
dm = child.dm
vS, vE = dm.getDepthStratum(0)
X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 3)


def side_faces(label):
    value = int(child.boundaries[label].value)
    faces = [int(p) for p in
             dm.getLabel(label).getStratumIS(value).getIndices()]
    polys, verts = [], {}
    pts = []
    for f in faces:
        tri = [int(p) for p in dm.getTransitiveClosure(f)[0]
               if vS <= int(p) < vE]
        ids = []
        for v in tri:
            if v not in verts:
                verts[v] = len(pts)
                pts.append(X[v - vS])
            ids.append(verts[v])
        polys.extend([3] + ids)
    return pv.PolyData(np.array(pts), faces=np.array(polys)), faces


plus, plus_faces = side_faces("FltAPlus")
minus, _ = side_faces("FltAMinus")

# rim = edges of the plus face set used exactly once
edge_use = {}
for f in plus_faces:
    for e in dm.getCone(f):
        edge_use[int(e)] = edge_use.get(int(e), 0) + 1
rim_lines = []
rim_pts = []
for e, n in edge_use.items():
    if n == 1:
        a, b = (int(q) for q in dm.getCone(e))
        rim_lines.extend([2, len(rim_pts), len(rim_pts) + 1])
        rim_pts.extend([X[a - vS], X[b - vS]])
rim = pv.PolyData(np.array(rim_pts), lines=np.array(rim_lines))

EPS = 0.05
plus_e = plus.translate((+EPS, 0, 0))
minus_e = minus.translate((-EPS, 0, 0))

pl = pv.Plotter(off_screen=True, window_size=(1200, 950))
pl.set_background("white")
pl.add_mesh(plus_e, color="#dce8fc", show_edges=True, edge_color="#4a7bf7",
            line_width=1.0, lighting=False, opacity=1.0)
pl.add_mesh(minus_e, color="#fce4ec", show_edges=True, edge_color="#e57373",
            line_width=1.0, lighting=False, opacity=1.0)
pl.add_mesh(rim, color="black", line_width=3.5, lighting=False)
box = pv.Box(bounds=(0, 1, 0, 1, 0, 1))
pl.add_mesh(box.extract_all_edges(), color="#999999", line_width=0.8,
            lighting=False)

pl.add_point_labels(
    np.array([[0.5 + EPS, 0.62, 0.76], [0.5 - EPS, 0.16, 0.30],
              [0.5, 0.26, 0.78]]),
    ["Plus copy (original points)", "Minus copy (replicas)",
     "rim: single, shared"],
    font_size=26, text_color="black", shape=None, always_visible=True,
    show_points=False)

pl.camera_position = [(1.75, -0.75, 1.25), (0.5, 0.5, 0.48), (0, 0, 1)]
out = os.path.join(D, "split-components-3d.png")
pl.screenshot(out)
pl.close()
print(f"wrote {out}")
