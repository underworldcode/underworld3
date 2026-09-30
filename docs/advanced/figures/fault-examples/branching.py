"""A branching rupture: intersecting fault traces and their Delta CFF.

The raw geometry INTERSECTS: a dextral trunk, a splay branching off its
midpoint at ~30 degrees (a T junction), and a conjugate fault crossing
it outright (an X junction). ``prepare_fault_network`` converts both to
offset-junction form automatically (angle-corrected ligaments, loudly),
which is what makes the set splittable at all. The trunk and the splay
then rupture TOGETHER (frictionless) while the conjugate is welded as a
receiver, and the map shows Delta CFF on trunk-parallel planes — with
zooms at the two junctions, where the ligament-scale stress transfer
lives. Measured ligament sensitivity for this representation:
~/+Simulations/fault_junction_ligament/ (the branch response is
converged for a given plug size; ligaments of 1-2 local cells).
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import underworld3 as uw
from underworld3.meshing.surfaces import prepare_fault_network
from underworld3.utilities import fault_contact

import common

pv.OFF_SCREEN = True
D = os.path.dirname(os.path.abspath(__file__))
MU_P = 0.4
H = 0.012
LIG_F = 1.5

# raw, INTERSECTING traces — the preparer makes them legal
TREND = np.degrees(np.arctan2(0.10, 0.70))          # trunk trend ~8 deg
M_RAW = ("Trunk", np.array([[0.15, 0.45], [0.85, 0.55]]))
S_RAW = ("Splay", np.array([[0.50, 0.50], [0.80, 0.76]]))     # T at trunk
C_RAW = ("Conj", np.array([[0.30, 0.33], [0.42, 0.64]]))      # X crossing

prepared, report = prepare_fault_network(
    [M_RAW, S_RAW, C_RAW], spacing=H, ligament=LIG_F, verbose=True)
names = [n for n, _ in prepared]
SLIP = [n for n in names if n.startswith(("Trunk", "Splay"))]
WELD = [n for n in names if n.startswith("Conj")]
ETA_WELD = 200.0 * common.ETA / 0.2

T_J = np.array([0.50, 0.50])                        # the T junction
X_J = M_RAW[1][0] + (M_RAW[1][1] - M_RAW[1][0]) * 0.293   # approx X point


def solve_state(child, free):
    stokes = common.stokes_on(child,
                              common.boundary_simple_shear(child, TREND))
    for n in names:
        stokes.add_fault_bc(0.0 if (free and n in SLIP) else ETA_WELD,
                            boundary=n)
    fault_contact.solve_with_fault(stokes, picard=2)
    x, y = child.X
    v, p = stokes.Unknowns.u, stokes.Unknowns.p
    comps = {}
    for cname, expr in (
            ("sxx", -p.sym[0] + 2 * common.ETA * v.sym[0].diff(x)),
            ("syy", -p.sym[0] + 2 * common.ETA * v.sym[1].diff(y)),
            ("sxy", common.ETA * (v.sym[0].diff(y) + v.sym[1].diff(x)))):
        s_var = uw.discretisation.MeshVariable(
            f"{cname}_{'a' if free else 'b'}", child, 1, degree=0,
            continuous=False)
        proj = uw.systems.Projection(child, s_var)
        proj.uw_function = expr
        proj.smoothing = 0.0
        proj.solve()
        row = common.split_mesh_cell_rows(child, s_var)
        comps[cname] = np.asarray(s_var.data[:, 0])[row].copy()
    return stokes, comps


t0 = time.perf_counter()
child = common.base_mesh(H).add_fault(prepared)
s1, c1 = solve_state(child, free=True)
print(f"[timing] slipping solve: {time.perf_counter() - t0:.1f} s",
      flush=True)
for n in SLIP:
    coords, jumps, normals = fault_contact.fault_pair_jumps(
        s1, n, s1._rotated_freeslip_info)
    tang = np.column_stack([-normals[:, 1], normals[:, 0]])
    V = np.einsum("ij,ij->i", jumps, tang)
    print(f"  {n:8s} peak slip {np.abs(V).max():.4f}", flush=True)
t0 = time.perf_counter()
_s0, c0 = solve_state(child, free=False)
print(f"[timing] welded solve: {time.perf_counter() - t0:.1f} s",
      flush=True)

# Delta CFF on trunk-parallel receiver planes
beta = np.radians(TREND)
nx, ny = -np.sin(beta), np.cos(beta)
tx, ty = np.cos(beta), np.sin(beta)


def resolve(c):
    s_nn = c["sxx"] * nx * nx + 2 * c["sxy"] * nx * ny + c["syy"] * ny * ny
    s_t = (c["sxx"] * tx * nx + c["sxy"] * (tx * ny + ty * nx)
           + c["syy"] * ty * ny)
    return s_nn, s_t


nn0, t_0 = resolve(c0)
nn1, t_1 = resolve(c1)
tau_dir = np.sign(np.median(t_0))
dcff = tau_dir * (t_1 - t_0) + MU_P * (nn1 - nn0)

pts, faces = common.split_mesh_cell_render(child)
fc = np.asarray(faces).reshape(-1, 4)[:, 1:]
cent = np.asarray(pts)[fc].mean(axis=1)
dcff, gauge = common.far_field_anchor(
    cent, dcff, [p for _n, p in prepared], cut=0.18)
print(f"far-field gauge removed: {gauge:+.4f}", flush=True)

# ---- renders: the map and the two junction zooms ---------------------------
COLOUR = {"Trunk": "black", "Splay": "black", "Conj": "#6a1b9a"}


def render(png, scale, focal):
    pvm = pv.PolyData(np.asarray(pts, dtype=float),
                      faces=np.asarray(faces, dtype=np.int64))
    pvm.cell_data["dcff"] = dcff
    pl = pv.Plotter(off_screen=True, window_size=(900, 850))
    pl.set_background("white")
    pl.add_mesh(pvm, scalars="dcff", cmap="RdBu_r", clim=(-1.0, 1.0),
                lighting=False, show_scalar_bar=False)
    for n, p in prepared:
        line = pv.lines_from_points(
            np.column_stack([p, np.full(len(p), 1e-3)]))
        pl.add_mesh(line, color=COLOUR[n.split("_")[0]],
                    line_width=5 if n.startswith("Trunk") else 4,
                    lighting=False)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = scale
    pl.camera.focal_point = (focal[0], focal[1], 0.0)
    pl.screenshot(png)
    pl.close()
    return png


map_png = render(os.path.join(D, "_branching_map.png"), 0.42, (0.5, 0.52))
tz_png = render(os.path.join(D, "_branching_tzoom.png"), 0.075, T_J)
xz_png = render(os.path.join(D, "_branching_xzoom.png"), 0.075, X_J)

fig = plt.figure(figsize=(12.6, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1.0, 0.05])
axm = fig.add_subplot(gs[:, 0])
axm.imshow(plt.imread(map_png))
axm.set_xticks([])
axm.set_yticks([])
axm.set_title(r"$\Delta$CFF on trunk-parallel planes ($\mu' = 0.4$): "
              "trunk + splay rupture together,\nconjugate welded; "
              "junctions are auto-converted offset ligaments", fontsize=9.5)
for row, (png, label) in enumerate(
        ((tz_png, "the branch point (T): splay fed through the ligament"),
         (xz_png, "the crossing (X): four tips, one intact plug"))):
    ax = fig.add_subplot(gs[row, 1])
    ax.imshow(plt.imread(png))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=8.5)

from matplotlib import cm, colors as mcolors
sm = cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap="RdBu_r")
cax = fig.add_subplot(gs[:, 2])
fig.colorbar(sm, cax=cax, label=r"$\Delta$CFF (per unit stress drop)")
fig.suptitle("A branching rupture on an intersecting fault network",
             fontsize=11.5)
fig.tight_layout()
out = os.path.join(D, "branching.png")
fig.savefig(out, dpi=200)
print("wrote", out, flush=True)
