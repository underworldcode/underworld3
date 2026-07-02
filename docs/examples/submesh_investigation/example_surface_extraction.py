"""Surface submesh extraction: documented example.

Demonstrates the third submesh flavour: extracting the upper surface of
a ``SphericalShell`` as a 2-manifold embedded in 3-space, with the same
submesh lineage state as ``Mesh.extract_region`` and
``coarsened_companion``.

Parallel to:
    test_region_ds_submesh.py     (subdomain via DMPlexFilter)
    example_refined_companion.py  (resolution level via dm.refine())

This example deliberately stops at the *mesh* layer -- solving on the
surface (lateral diffusion / Laplace-Beltrami) is the Session 2
problem. Once that lands, this example will be the natural place to
add an end-to-end "diffuse a Gaussian on the sphere" demo.

Run:
    pixi run -e amr-dev python -u \\
        docs/examples/submesh_investigation/example_surface_extraction.py
"""

import numpy as np
import underworld3 as uw

import surface_submesh_prototype as ssp


def banner(msg):
    uw.pprint(0, f"\n{'='*70}\n{msg}\n{'='*70}")


# ---------------------------------------------------------------------------
# 1. Parent SphericalShell (3D)
# ---------------------------------------------------------------------------

banner("Parent mesh: SphericalShell")

r_outer, r_inner = 1.0, 0.5
shell = uw.meshing.SphericalShell(
    radiusOuter=r_outer,
    radiusInner=r_inner,
    cellSize=0.2,
)
cS, cE = shell.dm.getHeightStratum(0)
uw.pprint(
    0,
    f"shell: dim={shell.dim}, cdim={shell.cdim}, "
    f"cells={cE - cS}, boundaries={[b.name for b in shell.boundaries]}",
)


# ---------------------------------------------------------------------------
# 2. Extract the upper surface as a uw.Mesh
# ---------------------------------------------------------------------------

banner("Extract upper surface (DMPlexCreateSubmesh on Upper)")

surface = ssp.extract_surface(shell, "Upper")
sS, sE = surface.dm.getHeightStratum(0)
uw.pprint(
    0,
    f"surface: dim={surface.dim}, cdim={surface.cdim}, "
    f"cells={sE - sS}, vertices={surface.X.coords.shape[0]}",
)
uw.pprint(0, f"  parent linked: {surface.parent is shell}")
uw.pprint(0, f"  registered with parent: {surface in shell._registered_submeshes}")
uw.pprint(
    0,
    f"  surviving boundaries: "
    f"{[b.name for b in surface.boundaries] if surface.boundaries else 'none'}",
)


# ---------------------------------------------------------------------------
# 3. Geometric sanity: every surface vertex sits on the parent sphere
# ---------------------------------------------------------------------------

banner("Geometric sanity")

radii = np.linalg.norm(surface.X.coords, axis=1)
uw.pprint(
    0,
    f"vertex radii: min={radii.min():.12e}, max={radii.max():.12e}",
)
max_dev = np.abs(radii - r_outer).max()
uw.pprint(0, f"|r - r_outer| max = {max_dev:.3e}  (expect < 1e-10)")
assert max_dev < 1e-10, "surface vertices not on the parent sphere"


# ---------------------------------------------------------------------------
# 4. Round-trip a scalar via the standard restrict/prolongate path
# ---------------------------------------------------------------------------

banner("Scalar round-trip: parent -> surface -> parent")

# Scalar fields on parent and surface
T_parent = uw.discretisation.MeshVariable(
    "T_parent", shell, num_components=1, degree=1,
)
T_surface = uw.discretisation.MeshVariable(
    "T_surface", surface, num_components=1, degree=1,
)
T_back = uw.discretisation.MeshVariable(
    "T_back", shell, num_components=1, degree=1,
)

# Plant a recognisable field on the parent: latitude as a function of z.
# (Any 3-coord function does; pick something that varies smoothly.)
parent_coords = np.asarray(T_parent.coords)  # (N, 3)
new_T = np.zeros_like(T_parent.data)
new_T[:, 0] = np.arctan2(
    np.sqrt(parent_coords[:, 0]**2 + parent_coords[:, 1]**2),
    parent_coords[:, 2],
)
T_parent.pack_raw_data_to_petsc(new_T, sync=True)

# parent -> surface (KDTree at 1e-10 picks the 252 surface DOFs)
surface.restrict(T_parent, T_surface)

# surface -> a fresh parent variable
surface.prolongate(T_surface, T_back)

# Compare on surface DOFs only (rest of parent should be zero in T_back)
nonzero = np.where(np.abs(T_back.data[:, 0]) > 0)[0]
diff = T_parent.data[nonzero, 0] - T_back.data[nonzero, 0]
uw.pprint(
    0,
    f"parent surface DOFs touched: {nonzero.shape[0]} "
    f"(surface has {surface.X.coords.shape[0]} vertices)",
)
uw.pprint(0, f"max |T_parent - T_back| on touched DOFs: {np.abs(diff).max():.3e}")
assert np.abs(diff).max() < 1e-12, "round-trip not bit-exact at surface DOFs"


# ---------------------------------------------------------------------------
# 5. Subpoint IS sanity (point-level parent <-> surface map)
# ---------------------------------------------------------------------------

banner("Subpoint IS sanity")

subpts = surface.subpoint_is
uw.pprint(0, f"subpoint_is size: {subpts.getSize()}")
parent_chart = shell.dm.getChart()
indices = subpts.getIndices()
uw.pprint(0, f"parent chart: {parent_chart}")
uw.pprint(0, f"subpoint index range: [{indices.min()}, {indices.max()}]")
assert indices.min() >= parent_chart[0]
assert indices.max() < parent_chart[1]


banner("EXAMPLE PASSED (Session 1 lineage + transfer)")
uw.pprint(
    0,
    "Solving on the surface (lateral diffusion / Laplace-Beltrami) is\n"
    "Session 2 -- it needs manifold-aware FE assembly through the JIT\n"
    "and solver layers and is tracked in the surface-submesh plan.",
)
