"""Diagnose what stratum the 'Upper'/'Lower' DMPlex labels actually mark.

If they mark edges (1-stratum) rather than vertices (0-stratum), the
_winslow_build_adjacency code's `if pStart <= idx < pEnd:` check rejects
them all and `is_boundary` stays empty — meaning Winslow has been
smoothing ALL vertices including the surface.
"""
import numpy as np
import underworld3 as uw

MESH = ("output/convection_zoo_snapshots_launchclip_winslow_n5/"
        "uw_bdf2_sl_step0025.mesh.00000.h5")

mesh = uw.discretisation.Mesh(MESH)
dm = mesh.dm

pStart, pEnd = dm.getDepthStratum(0)   # vertices
eStart, eEnd = dm.getDepthStratum(1)   # edges
cStart, cEnd = dm.getHeightStratum(0)  # cells (height 0 in 2D)

print(f"vertex stratum: [{pStart}, {pEnd})  ({pEnd-pStart} verts)")
print(f"edge   stratum: [{eStart}, {eEnd})  ({eEnd-eStart} edges)")
print(f"cell   stratum: [{cStart}, {cEnd})  ({cEnd-cStart} cells)")

for lname in ("Upper", "Lower"):
    print(f"\nlabel '{lname}':")
    label = dm.getLabel(lname)
    if label is None:
        print("  None")
        continue
    vIS = label.getValueIS()
    if vIS is None:
        print("  no values")
        continue
    vals = vIS.getIndices()
    print(f"  values: {list(vals)}")
    for v in vals:
        iset = label.getStratumIS(int(v))
        if iset is None:
            continue
        ids = iset.getIndices()
        n_vert = int(np.sum((ids >= pStart) & (ids < pEnd)))
        n_edge = int(np.sum((ids >= eStart) & (ids < eEnd)))
        n_cell = int(np.sum((ids >= cStart) & (ids < cEnd)))
        print(f"  value={v}: {len(ids)} points total — "
              f"verts={n_vert} edges={n_edge} cells={n_cell}")
        if len(ids):
            print(f"    first 8 ids: {list(ids[:8])}")
