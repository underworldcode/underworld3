"""Latest state of a running convection run: the newest four snapshots (T, mesh in black)
and Nu and vrms against time from its run_log. Usage: render_watch.py RUN_DIR OUT_PREFIX TITLE"""
import glob, re, sys
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import LinearTriInterpolator, Triangulation

run, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
cmap = LinearSegmentedColormap.from_list(
    "bnr", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f5a3a3", "#e34948", "#a42f2f"])
rows = {}
for line in open(f"{run}/run_log.txt"):
    p = line.split()
    if p and p[0].isdigit():
        rows[int(p[0])] = (float(p[1]), float(p[4]), float(p[5]))
steps = sorted(int(m.group(1)) for f in glob.glob(f"{run}/step*.mesh.00000.h5")
               if (m := re.search(r"step(\d+)\.mesh\.00000\.h5$", f)))
show = steps[-4:]
fig, axes = plt.subplots(1, max(len(show), 1), figsize=(4.4 * max(len(show), 1), 4.8))
axes = np.atleast_1d(axes)
for ax, s in zip(axes, show):
    with h5py.File(f"{run}/step{s:04d}.mesh.00000.h5") as h:
        xy = h["geometry/vertices"][...]; cells = h["viz/topology/cells"][...]
    with h5py.File(f"{run}/step{s:04d}.mesh.T_v2p1.00000.h5") as h:
        T = h["visualization/T_v2p1"][...].ravel()
    tri = Triangulation(xy[:, 0], xy[:, 1], cells)
    ax.tripcolor(tri, T, shading="gouraud", cmap=cmap, vmin=0, vmax=1)
    ax.triplot(tri, color="0.35", lw=0.1)
    # Streamlines: the P2 velocity interpolated linearly onto a regular grid,
    # masked outside the annulus. Line width scales with speed.
    with h5py.File(f"{run}/step{s:04d}.mesh.V_v2p1.00000.h5") as h:
        vx = h["fields/coordinates"][...]; vv = h["fields/V_v2p1"][...]
    # The velocity at the mesh vertices (the P2 nodes include them), on the
    # mesh's own triangles: a Delaunay of the nodes is invalid on a graded mesh.
    from scipy.spatial import cKDTree
    _, at = cKDTree(vx).query(xy)
    vt = tri
    vv = vv[at]
    g = np.linspace(-1, 1, 160); GX, GY = np.meshgrid(g, g)
    U = LinearTriInterpolator(vt, vv[:, 0])(GX, GY); V = LinearTriInterpolator(vt, vv[:, 1])(GX, GY)
    sp = np.hypot(U, V); R = np.hypot(GX, GY)
    out_ = (R < 0.5) | (R > 1.0)
    U = np.ma.masked_where(out_, U); V = np.ma.masked_where(out_, V)
    smax = float(np.nanmax(np.ma.filled(sp, np.nan))) or 1.0
    ax.streamplot(GX, GY, U, V, color="k", density=1.4, arrowsize=0.7,
                  linewidth=0.3 + 1.2 * np.ma.filled(sp, 0) / smax)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"step {s}, t = {rows.get(s, (np.nan,))[0]:.4f}")
fig.suptitle(title + "\n(T: blue cold, red hot)", fontsize=10)
fig.tight_layout(); fig.savefig(out + "_frames.png", dpi=110)
r = np.array([rows[k] for k in sorted(rows)])
fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
a.plot(r[:, 0], r[:, 2], color="#2a6fdb"); a.axhline(1, color="0.7", lw=0.8); a.set_ylabel("Nu (surface)")
b.plot(r[:, 0], r[:, 1], color="#2a6fdb"); b.set_ylabel("vrms (area integral)"); b.set_xlabel("t")
for z in (a, b): z.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"{title}: step {sorted(rows)[-1]}, t = {r[-1, 0]:.4f}, Nu = {r[-1, 2]:.3f}, vrms = {r[-1, 1]:.3g}")
fig.tight_layout(); fig.savefig(out + "_history.png", dpi=120)
print("rendered", show, "last step", sorted(rows)[-1])
