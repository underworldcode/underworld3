"""
Probe: Can DMComposite manage rock/air sub-DMs from DMPlexFilter?

Tests:
1. Create full mesh, filter into rock + air sub-DMs
2. Wrap in DMComposite
3. Check: global Vec size, scatter to sub-Vecs, IS mappings
4. Check: do interface nodes appear in both sub-DMs?
5. Can we set up fields on the rock sub-DM and solve within the composite?

This is an investigation — not a production pattern.
"""

from petsc4py import PETSc
import underworld3 as uw
from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label
import numpy as np

r_internal = 1.0; r_inner = 0.5; r_outer_full = 1.5; cellsize = 1/16

# --- Create full mesh and filter ---

print("Creating full mesh...", flush=True)
full_mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full, radiusInternal=r_internal,
    radiusInner=r_inner, cellSize=cellsize)

full_dm = full_mesh.dm

print("Filtering rock and air sub-DMs...", flush=True)
rock_dm = petsc_dm_filter_by_label(full_dm, "Inner", 101)
air_dm = petsc_dm_filter_by_label(full_dm, "Outer", 102)

# Basic info
print(f"\nFull mesh chart: {full_dm.getChart()}", flush=True)
print(f"Rock submesh chart: {rock_dm.getChart()}", flush=True)
print(f"Air submesh chart: {air_dm.getChart()}", flush=True)

# Count vertices
for name, dm in [("Full", full_dm), ("Rock", rock_dm), ("Air", air_dm)]:
    depth = dm.getLabel("depth")
    v_is = depth.getStratumIS(0)
    n_verts = v_is.getSize() if v_is else 0
    c_is = depth.getStratumIS(2)
    n_cells = c_is.getSize() if c_is else 0
    print(f"  {name}: {n_verts} vertices, {n_cells} cells", flush=True)

# --- Check subpoint IS (interface overlap) ---

print("\n--- Subpoint IS (submesh -> parent mapping) ---", flush=True)
rock_subpoint = rock_dm.getSubpointIS()
air_subpoint = air_dm.getSubpointIS()

if rock_subpoint:
    rock_pts = set(rock_subpoint.getIndices())
    print(f"Rock subpoint IS size: {len(rock_pts)}", flush=True)
else:
    rock_pts = set()
    print("Rock subpoint IS: None", flush=True)

if air_subpoint:
    air_pts = set(air_subpoint.getIndices())
    print(f"Air subpoint IS size: {len(air_pts)}", flush=True)
else:
    air_pts = set()
    print("Air subpoint IS: None", flush=True)

if rock_pts and air_pts:
    overlap = rock_pts & air_pts
    print(f"Overlap (shared points): {len(overlap)}", flush=True)

    # What depth are the shared points?
    full_depth = full_dm.getLabel("depth")
    overlap_by_depth = {}
    for pt in overlap:
        for d in range(3):
            d_is = full_depth.getStratumIS(d)
            if d_is and pt in set(d_is.getIndices()):
                overlap_by_depth[d] = overlap_by_depth.get(d, 0) + 1
    print(f"  By depth: {overlap_by_depth} (0=vertices, 1=edges, 2=cells)", flush=True)

# --- Try DMComposite ---

print("\n--- DMComposite test ---", flush=True)

# DMComposite needs sub-DMs with sections (fields defined)
# Let's add a simple scalar field to each
rock_dm.setNumFields(1)
fe_rock = PETSc.FE().createDefault(2, 1, True, 1, comm=PETSc.COMM_WORLD)
rock_dm.setField(0, fe_rock)
rock_dm.createDS()

air_dm.setNumFields(1)
fe_air = PETSc.FE().createDefault(2, 1, True, 1, comm=PETSc.COMM_WORLD)
air_dm.setField(0, fe_air)
air_dm.createDS()

# Create composite
comp = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
comp.addDM(rock_dm)
comp.addDM(air_dm)
comp.setUp()

# Global vector
gvec = comp.createGlobalVec()
print(f"Composite global Vec size: {gvec.getSize()}", flush=True)

# Check individual sub-DM vector sizes
rock_gvec = rock_dm.createGlobalVector()
air_gvec = air_dm.createGlobalVector()
print(f"Rock global Vec size: {rock_gvec.getSize()}", flush=True)
print(f"Air global Vec size: {air_gvec.getSize()}", flush=True)
print(f"Sum: {rock_gvec.getSize() + air_gvec.getSize()}", flush=True)
print(f"Full mesh would have: {full_dm.getChart()[1]} points (but DOFs depend on section)", flush=True)

# Get IS mappings
gISs = comp.getGlobalISs()
print(f"\nGlobal IS count: {len(gISs)}", flush=True)
for i, gis in enumerate(gISs):
    print(f"  IS[{i}]: size={gis.getSize()}, range=[{gis.getIndices().min()}, {gis.getIndices().max()}]", flush=True)

# Scatter test: set rock values to 1, air to 2, scatter back
rock_gvec.set(1.0)
air_gvec.set(2.0)

# Gather into composite (petsc4py uses scatterArray/gatherArray)
comp.scatter(gvec, [rock_gvec, air_gvec])
print(f"\nAfter scatter: rock sum={rock_gvec.sum():.0f}, air sum={air_gvec.sum():.0f}", flush=True)

# Set sub-vecs and gather back
rock_gvec.set(1.0)
air_gvec.set(2.0)
comp.gather(gvec, PETSc.InsertMode.INSERT_VALUES, [rock_gvec, air_gvec])
arr = gvec.getArray()
print(f"Composite Vec after gather: min={arr.min()}, max={arr.max()}", flush=True)
print(f"  Values==1 (rock): {(arr == 1.0).sum()}", flush=True)
print(f"  Values==2 (air): {(arr == 2.0).sum()}", flush=True)

# The key question: can we map composite DOFs back to full mesh DOFs?
# rock subpoint IS maps rock_dm point -> full_dm point
# air subpoint IS maps air_dm point -> full_dm point
# But the composite IS maps composite index -> concatenated index
# We need: composite index -> full mesh DOF
print("\n--- Mapping composite -> full mesh ---", flush=True)
print(f"Rock subpoint IS gives rock_dm points -> full_dm points", flush=True)
print(f"  e.g. rock point 0 -> full point {rock_subpoint.getIndices()[0]}", flush=True)
print(f"  e.g. rock point 100 -> full point {rock_subpoint.getIndices()[100]}", flush=True)
if air_subpoint:
    print(f"  e.g. air point 0 -> full point {air_subpoint.getIndices()[0]}", flush=True)
    print(f"  e.g. air point 100 -> full point {air_subpoint.getIndices()[100]}", flush=True)

print("\n--- Done ---", flush=True)
