"""
Phase 0 probe: surface submesh extraction via DMPlexCreateSubmesh.

Answers four investigation questions before any prototype is written:

1. Does ``petsc_dm_create_submesh_from_label(marked_faces=True)`` produce
   a DM with ``getDimension()=2`` and ``getCoordinateDim()=3`` on a
   spherical shell's upper boundary?
2. Does that surface DM expose ``getSubpointIS()`` the same way the
   ``DMPlexFilter``-derived submeshes do?
3. What labels survive on the surface DM? A closed sphere surface has
   no boundary of its own, so we expect parent boundary labels to land
   on the surface either as empty strata or as the surface's *own*
   cells (depending on how PETSc maps them).
4. Do surface vertex coordinates sit at ``r = radiusOuter`` to
   machine precision?

This is investigation code under docs/examples/, run via:

    pixi run -e amr-dev python -u \
        docs/examples/submesh_investigation/probe_surface_extraction.py
"""

import numpy as np
from petsc4py import PETSc

import underworld3 as uw
from underworld3.cython.petsc_discretisation import (
    petsc_dm_create_submesh_from_label,
)


def _hdr(title):
    uw.pprint(0, "")
    uw.pprint(0, f"=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Build a parent SphericalShell and inspect its boundary labels
# ---------------------------------------------------------------------------

_hdr("Parent SphericalShell")

r_outer = 1.0
r_inner = 0.5
cellsize = 0.25  # coarse — we only need a few cells for the probe

shell = uw.meshing.SphericalShell(
    radiusOuter=r_outer,
    radiusInner=r_inner,
    cellSize=cellsize,
)

uw.pprint(0, f"parent dim     = {shell.dm.getDimension()}")
uw.pprint(0, f"parent cdim    = {shell.dm.getCoordinateDim()}")
uw.pprint(0, f"parent cells   = "
            f"{shell.dm.getHeightStratum(0)[1] - shell.dm.getHeightStratum(0)[0]}")
uw.pprint(0, f"parent boundaries: {[b.name for b in shell.boundaries]}")

upper_value = shell.boundaries.Upper.value
uw.pprint(0, f"  Upper.value = {upper_value}")

# Confirm the Upper label has a non-empty face stratum on the parent
upper_label = shell.dm.getLabel("Upper")
upper_is = upper_label.getStratumIS(upper_value)
upper_count = upper_is.getSize() if upper_is is not None else 0
uw.pprint(0, f"  parent has {upper_count} marked points (faces) for Upper")


# ---------------------------------------------------------------------------
# 2. Extract the surface via DMPlexCreateSubmesh
# ---------------------------------------------------------------------------

_hdr("DMPlexCreateSubmesh(Upper, marked_faces=True)")

surf_dm = petsc_dm_create_submesh_from_label(
    shell.dm, "Upper", upper_value, marked_faces=True,
)

uw.pprint(0, f"surface DM type    = {type(surf_dm).__name__}")
uw.pprint(0, f"surface dim        = {surf_dm.getDimension()}")
uw.pprint(0, f"surface cdim       = {surf_dm.getCoordinateDim()}")

sS, sE = surf_dm.getHeightStratum(0)
uw.pprint(0, f"surface cells      = {sE - sS}")

vS, vE = surf_dm.getDepthStratum(0)
uw.pprint(0, f"surface vertices   = {vE - vS}")


# ---------------------------------------------------------------------------
# 3. Subpoint IS — point-level parent ↔ surface map
# ---------------------------------------------------------------------------

_hdr("Subpoint IS")

try:
    subpoint_is = surf_dm.getSubpointIS()
except Exception as e:
    uw.pprint(0, f"getSubpointIS FAILED: {e!r}")
    subpoint_is = None

if subpoint_is is not None:
    indices = subpoint_is.getIndices()
    uw.pprint(0, f"subpoint_is size   = {subpoint_is.getSize()}")
    uw.pprint(0, f"first 10 indices   = {indices[:10].tolist()}")
    uw.pprint(0, f"index range        = [{indices.min()}, {indices.max()}]")

    p_chart = shell.dm.getChart()
    uw.pprint(0, f"parent chart       = {p_chart}")
    uw.pprint(0,
              "  (indices should lie inside parent chart — they are "
              "point IDs in the parent's numbering)")


# ---------------------------------------------------------------------------
# 4. Surface vertex coordinates — embedded in 3-space at r=r_outer
# ---------------------------------------------------------------------------

_hdr("Surface vertex coordinates")

coords_local = surf_dm.getCoordinatesLocal().array
# Layout: flat array of cdim-tuples per vertex (cdim is on the SURFACE DM)
cdim = surf_dm.getCoordinateDim()
coords = coords_local.reshape(-1, cdim)
uw.pprint(0, f"coords shape       = {coords.shape}")
uw.pprint(0, f"cdim used          = {cdim}")

radii = np.linalg.norm(coords, axis=1)
uw.pprint(0, f"radius range       = [{radii.min():.10e}, {radii.max():.10e}]")
uw.pprint(0, f"|r - r_outer| max  = {np.abs(radii - r_outer).max():.3e}")
uw.pprint(0, f"|r - r_outer| < 1e-10: "
            f"{bool(np.all(np.abs(radii - r_outer) < 1e-10))}")


# ---------------------------------------------------------------------------
# 5. Label survival
# ---------------------------------------------------------------------------

_hdr("Label survival on surface DM")

# Enumerate all labels on the surface DM directly first — don't risk
# probing parent label *names* against the surface DM (we hit a silent
# PETSc abort doing that). This loop reaches the real survivor list
# without touching any label by name.
import sys

uw.pprint(0, "  Enumerating submesh labels by index:")
sys.stdout.flush()
n_labels = surf_dm.getNumLabels()
uw.pprint(0, f"    getNumLabels = {n_labels}")
sys.stdout.flush()
all_label_names = []
for i in range(n_labels):
    nm = surf_dm.getLabelName(i)
    all_label_names.append(nm)
    uw.pprint(0, f"    [{i}] {nm}")
    sys.stdout.flush()

uw.pprint(0, "  Stratum sizes (only labels listed above):")
sys.stdout.flush()
# Skip "Centre" — known hard-abort pseudo-label
# (project_annulus_centre_pseudo_label memory)
for nm in all_label_names:
    if nm == "Centre":
        uw.pprint(0, f"    {nm:<24} skipped (pseudo-label, would hard-abort)")
        sys.stdout.flush()
        continue
    lab = surf_dm.getLabel(nm)
    if lab is None:
        uw.pprint(0, f"    {nm:<24} getLabel returned None")
        sys.stdout.flush()
        continue
    # Some labels have multiple strata; iterate values
    try:
        n_vals = lab.getNumValues()
    except Exception as e:
        uw.pprint(0, f"    {nm:<24} getNumValues failed: {e!r}")
        sys.stdout.flush()
        continue
    try:
        vals_is = lab.getValueIS()
        vals = vals_is.getIndices().tolist() if vals_is is not None else []
    except Exception as e:
        uw.pprint(0, f"    {nm:<24} getValueIS failed: {e!r}")
        sys.stdout.flush()
        continue
    sizes = {}
    for v in vals:
        sis = lab.getStratumIS(int(v))
        sizes[int(v)] = sis.getSize() if sis is not None else 0
    uw.pprint(0, f"    {nm:<24} values={vals} sizes={sizes}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 6. Probe chart layout of the surface DM
# ---------------------------------------------------------------------------

_hdr("Surface DM chart layout")

cS_, cE_ = surf_dm.getHeightStratum(0)
eS_, eE_ = surf_dm.getHeightStratum(1)  # edges (in 2D mesh)
vS_, vE_ = surf_dm.getDepthStratum(0)

uw.pprint(0, f"  cells  : [{cS_}, {cE_})  count={cE_-cS_}")
uw.pprint(0, f"  edges  : [{eS_}, {eE_})  count={eE_-eS_}")
uw.pprint(0, f"  vertices: [{vS_}, {vE_})  count={vE_-vS_}")

# Try one cell's closure to see point ordering
cell_id = cS_
closure_pts, _ = surf_dm.getTransitiveClosure(cell_id)
uw.pprint(0, f"  closure(cell {cell_id}) = {closure_pts.tolist()}")
uw.pprint(0, f"  cell_num_points (entities[dim]=3) → last 3 points:")
uw.pprint(0, f"    {closure_pts[-3:].tolist()}")
uw.pprint(0, f"  Are last-3 points in [{vS_}, {vE_})? "
            f"{bool(np.all((closure_pts[-3:] >= vS_) & (closure_pts[-3:] < vE_)))}")

# ---------------------------------------------------------------------------
# 7. Wrap the surface DM in uw.Mesh (loud failures are diagnostic)
# ---------------------------------------------------------------------------

_hdr("Probe: can we wrap surf_dm in uw.discretisation.Mesh?")

try:
    surf_mesh = uw.discretisation.Mesh(
        surf_dm,
        degree=shell.degree,
        qdegree=shell.qdegree,
        coordinate_system_type=shell.CoordinateSystemType,
        verbose=False,
    )
    uw.pprint(0, f"  uw.Mesh wrap OK: dim={surf_mesh.dim}, cdim={surf_mesh.cdim}")
    uw.pprint(0, f"  surf_mesh.X.coords shape = {surf_mesh.X.coords.shape}")
    radii_uw = np.linalg.norm(surf_mesh.X.coords, axis=1)
    uw.pprint(0,
              f"  X.coords radii: [{radii_uw.min():.10e}, "
              f"{radii_uw.max():.10e}]")
except Exception as e:
    uw.pprint(0, f"  uw.Mesh wrap FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

uw.pprint(0, "")
uw.pprint(0, "Probe complete.")
