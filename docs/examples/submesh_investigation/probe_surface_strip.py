"""Phase 0 probe: stripping the phantom depth-3 stratum.

Compose DMPlexCreateSubmesh + DMPlexFilter(depth, 2) and verify that:

  1. The resulting DM has a clean 3-stratum chart (depth = 0, 1, 2).
  2. ``getSubpointIS()`` on the filtered DM maps to points of the
     un-filtered submesh — which we can then compose with the original
     submesh's subpoint IS to get the parent map.
  3. Coordinates are preserved: vertices still at r = r_outer.
  4. Cell closures are now exactly (cell, edges, vertices) — no phantom
     point in the middle.
  5. Standard ``Mesh._build_kd_tree_index`` (without the dim!=cdim
     early-exit branch) survives the clean chart.

Run:
    pixi run -e amr-dev python -u \\
        docs/examples/submesh_investigation/probe_surface_strip.py
"""

import numpy as np

import underworld3 as uw
from underworld3.cython.petsc_discretisation import (
    petsc_dm_create_submesh_from_label,
    petsc_dm_filter_by_label,
)


def hdr(t):
    uw.pprint(0, ""); uw.pprint(0, f"=== {t} ===")


hdr("Parent SphericalShell")
shell = uw.meshing.SphericalShell(
    radiusOuter=1.0, radiusInner=0.5, cellSize=0.25,
)
uw.pprint(0, f"shell dim={shell.dim} cdim={shell.cdim}")


hdr("Stage 1: DMPlexCreateSubmesh(Upper, marked_faces=True)")
sub1 = petsc_dm_create_submesh_from_label(
    shell.dm, "Upper", shell.boundaries.Upper.value, marked_faces=True,
)
uw.pprint(0, f"sub1 dim={sub1.getDimension()} cdim={sub1.getCoordinateDim()}")
uw.pprint(0, f"sub1 chart={sub1.getChart()}")
for d in range(4):
    try:
        s, e = sub1.getDepthStratum(d)
        uw.pprint(0, f"  depth={d}: [{s}, {e}) count={e-s}")
    except Exception:
        pass
uw.pprint(0, f"sub1 num labels = {sub1.getNumLabels()}")


hdr("Stage 2: DMPlexFilter on (depth, 2) to drop the phantom stratum")
sub2 = petsc_dm_filter_by_label(sub1, "depth", 2)
uw.pprint(0, f"sub2 dim={sub2.getDimension()} cdim={sub2.getCoordinateDim()}")
uw.pprint(0, f"sub2 chart={sub2.getChart()}")
for d in range(4):
    try:
        s, e = sub2.getDepthStratum(d)
        uw.pprint(0, f"  depth={d}: [{s}, {e}) count={e-s}")
    except Exception:
        pass


hdr("Subpoint IS composition")
sub1_sp = sub1.getSubpointIS()  # sub1 point -> parent point
sub2_sp = sub2.getSubpointIS()  # sub2 point -> sub1 point
uw.pprint(0, f"sub1->parent IS size = {sub1_sp.getSize()}")
uw.pprint(0, f"sub2->sub1 IS size   = {sub2_sp.getSize()}")
sub1_idx = sub1_sp.getIndices()
sub2_idx = sub2_sp.getIndices()
# Compose: sub2 point i -> sub1 point sub2_idx[i] -> parent point sub1_idx[sub2_idx[i]]
parent_idx = sub1_idx[sub2_idx]
uw.pprint(0, f"composed sub2->parent map size = {parent_idx.shape[0]}")
uw.pprint(0,
          f"composed index range = [{parent_idx.min()}, {parent_idx.max()}] "
          f"(parent chart = {shell.dm.getChart()})")


hdr("Surface vertex coordinates after strip")
coords = sub2.getCoordinatesLocal().array.reshape(-1, sub2.getCoordinateDim())
radii = np.linalg.norm(coords, axis=1)
uw.pprint(0, f"sub2 vertex count = {coords.shape[0]}")
uw.pprint(0, f"radii: [{radii.min():.10e}, {radii.max():.10e}]")
uw.pprint(0, f"|r - 1| max = {np.abs(radii - 1.0).max():.3e}")


hdr("Cell closure layout on the stripped DM")
cS, cE = sub2.getHeightStratum(0)
for cid in [cS, cS+1, cS+10, cE-1]:
    pts, _ = sub2.getTransitiveClosure(cid)
    uw.pprint(0, f"  closure(cell {cid}) = {pts.tolist()}  (len={len(pts)})")
uw.pprint(0, "  Expect length 7 = 1 cell + 3 edges + 3 vertices (no phantom).")


hdr("Label survival on the stripped DM")
names = [sub2.getLabelName(i) for i in range(sub2.getNumLabels())]
uw.pprint(0, f"labels: {names}")
for nm in names:
    if nm == "Centre":
        continue
    lab = sub2.getLabel(nm)
    if lab is None: continue
    try:
        vis = lab.getValueIS()
        vals = vis.getIndices().tolist() if vis is not None else []
        sizes = {int(v): lab.getStratumIS(int(v)).getSize() for v in vals}
        uw.pprint(0, f"  {nm:<24} values={vals} sizes={sizes}")
    except Exception as e:
        uw.pprint(0, f"  {nm:<24} value probe failed: {e!r}")


hdr("Wrap the stripped DM in uw.Mesh")
try:
    surf = uw.discretisation.Mesh(
        sub2, degree=shell.degree, qdegree=shell.qdegree,
        coordinate_system_type=shell.CoordinateSystemType, verbose=False,
    )
    uw.pprint(0, f"OK: dim={surf.dim}, cdim={surf.cdim}, "
                 f"vertices={surf.X.coords.shape[0]}")
    uw.pprint(0,
              f"X.coords radii: [{np.linalg.norm(np.asarray(surf.X.coords), axis=1).min():.6e}, "
              f"{np.linalg.norm(np.asarray(surf.X.coords), axis=1).max():.6e}]")
except Exception as e:
    uw.pprint(0, f"FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

uw.pprint(0, ""); uw.pprint(0, "Probe complete.")
