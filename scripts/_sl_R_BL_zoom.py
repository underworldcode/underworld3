"""Zoom-in on the inner BL for R=1.5, R=2, R=3.
Matplotlib-based so the viewport is controllable."""
import os
import numpy as np
import sympy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import underworld3 as uw


BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/R_compare')
R_LIST = [1.5, 2.0, 3.0, 4.0, 6.0, 10.0]


def load(R):
    out = os.path.join(BASE, f"R{R}")
    m = uw.discretisation.Mesh(
        os.path.join(out, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=out)
    return m, T


def cells_from_mesh(m):
    """Triangle vertex-index triples (in mesh.X.coords order)."""
    dm = m.dm
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, pEnd = dm.getDepthStratum(0)
    tris = []
    for c in range(cStart, cEnd):
        closure = dm.getTransitiveClosure(c)[0]
        vs = [p - pStart for p in closure if pStart <= p < pEnd]
        if len(vs) == 3:
            tris.append(vs)
    return np.asarray(tris, dtype=np.int64)


# Zoom region: a sector around one of the inner-BL plumes
# (the right-most plume sits at θ=0 for our mode-5 IC + symmetry)
ZOOM_XLIM = (0.40, 0.85)
ZOOM_YLIM = (-0.22, 0.22)

# Sample |∇T| on a fine grid for the bg color
nx, ny = 360, 360
xs = np.linspace(*ZOOM_XLIM, nx)
ys = np.linspace(*ZOOM_YLIM, ny)
Xc, Yc = np.meshgrid(xs, ys, indexing='xy')
pts = np.column_stack([Xc.ravel(), Yc.ravel()])
# Filter to in-annulus points
r_pt = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
in_ann = (r_pt > 0.51) & (r_pt < 0.99)

# Pre-pass: compute |∇T| max across all R for shared clim
g_max = 0.0
gT_panels = []
edges_panels = []
for R in R_LIST:
    m, T = load(R)
    X = m.CoordinateSystem.X
    gT_sym = sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                        + T.sym[0].diff(X[1]) ** 2)
    # Evaluate at safe points (mask outside annulus); project bad
    # points onto an in-annulus radius and overwrite later
    pts_safe = pts.copy()
    bad = ~in_ann
    th_b = np.arctan2(pts_safe[bad, 1], pts_safe[bad, 0])
    pts_safe[bad, 0] = 0.75 * np.cos(th_b)
    pts_safe[bad, 1] = 0.75 * np.sin(th_b)
    g = np.asarray(uw.function.evaluate(
        gT_sym, pts_safe)).reshape(-1)
    g[bad] = np.nan
    g = g.reshape(ny, nx)
    g_max = max(g_max, float(np.nanmax(g)))
    coords = np.asarray(m.X.coords)
    tris = cells_from_mesh(m)
    gT_panels.append(g)
    edges_panels.append((coords, tris))
print(f"global |∇T|max = {g_max:.3e}")


# Plot
fig, axes = plt.subplots(1, len(R_LIST),
                         figsize=(5.5 * len(R_LIST), 5.0))
for ax, R, g, (coords, tris) in zip(
        axes, R_LIST, gT_panels, edges_panels):
    cs = ax.pcolormesh(Xc, Yc, g, cmap='Greens',
                       vmin=0.0, vmax=g_max,
                       shading='gouraud')
    # Mesh edges from the actual triangles
    tri_obj = mtri.Triangulation(coords[:, 0], coords[:, 1],
                                  triangles=tris)
    ax.triplot(tri_obj, color='black', lw=0.5, alpha=0.7)
    # Inner ring outline
    th_r = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax.plot(0.5 * np.cos(th_r), 0.5 * np.sin(th_r),
            color='red', lw=1.2)
    ax.set_xlim(*ZOOM_XLIM)
    ax.set_ylim(*ZOOM_YLIM)
    ax.set_aspect('equal')
    ax.set_title(f"R={R}  inner-BL zoom", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

cax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
fig.colorbar(cs, cax=cax, label=r"$|\nabla T|$")
fig.subplots_adjust(left=0.02, right=0.9, top=0.94,
                    bottom=0.04, wspace=0.04)

out = os.path.join(BASE, "plot_R_BL_zoom.png")
fig.savefig(out, dpi=170, bbox_inches='tight')
print(f"wrote {out}")
