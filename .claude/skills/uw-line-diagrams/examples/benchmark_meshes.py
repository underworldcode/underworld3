"""The benchmark mesh figures in the house style: DFG cylinder channel, Waters and King
start-up box, plain channel. Run from a pixi environment with underworld3 built:

    python benchmark_meshes.py [outdir]

Writes mesh_<name>.{pdf,svg,png} into outdir (default: ./figures).
"""
import os, sys
import numpy as np
import underworld3 as uw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from uwfig import *

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
W, H, R, C = 2.2, 0.41, 0.05, (0.2, 0.2)


def cylinder_mesh(resolution=10, circle_ratio=0.25, refinement=1):
    """The DFG channel with the cylinder cells a quarter of the bulk, refined once with the
    new vertices snapped back to the circle (the mesh the benchmark runs use)."""
    from enum import Enum
    class boundaries(Enum):
        bottom = 1; right = 2; top = 3; left = 4; inclusion = 5; All_Boundaries = 1001
    csize = 1.0 / resolution
    os.makedirs(".meshes", exist_ok=True)
    mesh_file = f".meshes/dfg_cylinder_{resolution}.msh"
    if uw.mpi.rank == 0 and not os.path.exists(mesh_file):
        import pygmsh
        with pygmsh.geo.Geometry() as geom:
            geom.characteristic_length_max = csize
            inclusion = geom.add_circle((C[0], C[1], 0.0), R, make_surface=False, mesh_size=circle_ratio * csize)
            domain = geom.add_rectangle(xmin=0.0, ymin=0.0, xmax=W, ymax=H, z=0, holes=[inclusion], mesh_size=csize)
            for i, b in enumerate((boundaries.bottom, boundaries.right, boundaries.top, boundaries.left)):
                geom.add_physical(domain.surface.curve_loop.curves[i], label=b.name)
            geom.add_physical(inclusion.curve_loop.curves, label=boundaries.inclusion.name)
            geom.add_physical(domain.surface, label="Elements")
            geom.generate_mesh(dim=2, verbose=False)
            geom.save_geometry(mesh_file)
    uw.mpi.comm.barrier()

    def snap(dm):
        c2 = dm.getCoordinatesLocal()
        coords = c2.array.reshape(-1, 2) - np.array(C)
        r = np.sqrt((coords ** 2).sum(1)).reshape(-1, 1)
        idx = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(dm, "inclusion")
        coords[idx] *= R / r[idx]
        c2.array[...] = (coords + np.array(C)).reshape(-1); dm.setCoordinatesLocal(c2)

    return uw.discretisation.Mesh(mesh_file, markVertices=True, useMultipleTags=True, useRegions=True,
                                  refinement=refinement, refinement_callback=snap, boundaries=boundaries, qdegree=3)


# ---------------- 1. DFG cylinder channel with two zoom insets ----------------
tris = mesh_triangles(cylinder_mesh())
fig, ax = figure(width_mm=183, aspect=0.40)
ax.set_position([0.03, 0.44, 0.94, 0.54])
draw_mesh(ax, tris)
boundary(ax, [(0, 0), (W, 0), (W, H), (0, H)])
circle_body(ax, C, R)
inflow_profile(ax, 0.0, 0.0, H, 1.5, side="left", scale=0.12)
inflow_profile(ax, W, 0.0, H, 1.5, side="right", scale=0.12)
label(ax, -0.26, H / 2, "inlet", rot=90, color=VELOCITY)
label(ax, W + 0.26, H / 2, "outlet", rot=270, color=VELOCITY)
label(ax, W / 2, H + 0.07, "no-slip"); label(ax, W / 2, -0.07, "no-slip")
leader(ax, (0.42, 0.33), (C[0] + R * 0.75, C[1] + R * 0.7), "no-slip, $R = 0.05$")
axes_glyph(ax, (-0.32, -0.15), 0.1)
ax.set_xlim(-0.36, W + 0.34); ax.set_ylim(-0.18, H + 0.17)
for pos, (x0, x1, y0, y1) in (([0.06, 0.02, 0.36, 0.38], (0.05, 0.55, 0.0, H)),
                              ([0.58, 0.02, 0.36, 0.38], (C[0] - 2 * R, C[0] + 2 * R, C[1] - 2 * R, C[1] + 2 * R))):
    axz = fig.add_axes(pos); axz.set_aspect("equal")
    draw_mesh(axz, tris, lw=0.35)
    circle_body(axz, C, R)
    axz.set_xlim(x0, x1); axz.set_ylim(y0, y1); axz.set_xticks([]); axz.set_yticks([])
    for s in axz.spines.values(): s.set_edgecolor(INK); s.set_linewidth(0.6)
save(fig, os.path.join(OUT, "mesh_dfg_cylinder_1_20"))

# ---------------- 2. Waters and King start-up box: walls at top and bottom, body force G ----------------
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 16, qdegree=3, regular=True)
fig, ax = figure(width_mm=90, aspect=0.95)
draw_mesh(ax, mesh)
boundary(ax, [(-1, -1), (1, -1), (1, 1), (-1, 1)])
label(ax, 0, 1.14, "no-slip"); label(ax, 0, -1.14, "no-slip")
label(ax, -1.22, 0, "$v = 0$", rot=90, color=VELOCITY); label(ax, 1.22, 0, "$v = 0$", rot=270, color=VELOCITY)
uniform_arrows(ax, -0.25, 0.25, np.linspace(-0.8, 0.8, 5), kind="force")
label(ax, 0.45, 0.0, "$G$", ha="left", color=FORCE)
axes_glyph(ax, (-1.45, -1.35), 0.25)
ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.45, 1.45)
save(fig, os.path.join(OUT, "mesh_waters_king_box"))

# ---------------- 3. Plain channel (no cylinder) ----------------
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(W, H), cellSize=0.05, qdegree=3)
fig, ax = figure(width_mm=183, aspect=0.27)
draw_mesh(ax, mesh, lw=0.3)
boundary(ax, [(0, 0), (W, 0), (W, H), (0, H)])
inflow_profile(ax, 0.0, 0.0, H, 1.5, side="left", scale=0.12)
inflow_profile(ax, W, 0.0, H, 1.5, side="right", scale=0.12)
label(ax, -0.26, H / 2, "inlet", rot=90, color=VELOCITY)
label(ax, W + 0.26, H / 2, "outlet", rot=270, color=VELOCITY)
label(ax, W / 2, H + 0.08, "no-slip"); label(ax, W / 2, -0.08, "no-slip")
axes_glyph(ax, (-0.32, -0.18), 0.1)
ax.set_xlim(-0.36, W + 0.34); ax.set_ylim(-0.22, H + 0.2)
save(fig, os.path.join(OUT, "mesh_channel_box"))
print("wrote", OUT)
