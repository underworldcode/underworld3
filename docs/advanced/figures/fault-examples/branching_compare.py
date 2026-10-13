"""What does interrupting the trunk cost? Continuous vs cut at the
junction, against the isolated trunk.

Same network as ``branching.py`` (trunk + splay slipping, conjugate
welded), prepared two ways:

- ``through=["Trunk"]`` — the trunk is the MASTER: never cut; the
  conjugate yields on both sides of the crossing and the splay is
  trimmed where it abuts (T junctions never cut the through-going
  trace).
- default — the X crossing cuts both traces, so the trunk is
  interrupted mid-length by the junction plug.

The isolated continuous trunk (no other faults) is the reference
profile. The comparison answers: how much slip does the interruption
forfeit, and what does it do to the Delta CFF field?
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
TREND = np.degrees(np.arctan2(0.10, 0.70))
M_RAW = ("Trunk", np.array([[0.15, 0.45], [0.85, 0.55]]))
S_RAW = ("Splay", np.array([[0.50, 0.50], [0.80, 0.76]]))
C_RAW = ("Conj", np.array([[0.30, 0.33], [0.42, 0.64]]))
T_HAT = (M_RAW[1][1] - M_RAW[1][0])
T_HAT = T_HAT / np.linalg.norm(T_HAT)
ETA_WELD = 200.0 * common.ETA / 0.2


def run_case(tag, faults, weld_names, want_field):
    child = common.base_mesh(H).add_fault(faults)
    names = [n for n, _ in faults]

    def one_state(free):
        stokes = common.stokes_on(
            child, common.boundary_simple_shear(child, TREND))
        for n in names:
            stokes.add_fault_bc(
                0.0 if (free and n not in weld_names) else ETA_WELD,
                boundary=n)
        fault_contact.solve_with_fault(stokes, picard=2)
        return stokes

    t0 = time.perf_counter()
    s1 = one_state(True)
    # trunk slip vs GLOBAL trunk arc length, across however many pieces
    ss, vv = [], []
    for n in names:
        if not n.startswith("Trunk"):
            continue
        coords, jumps, normals = fault_contact.fault_pair_jumps(
            s1, n, s1._rotated_freeslip_info)
        tang = np.column_stack([-normals[:, 1], normals[:, 0]])
        ss.append((coords - M_RAW[1][0]) @ T_HAT)
        vv.append(np.abs(np.einsum("ij,ij->i", jumps, tang)))
    s_all = np.concatenate(ss)
    v_all = np.concatenate(vv)
    order = np.argsort(s_all)
    profile = (s_all[order], v_all[order])

    dcff_pack = None
    if want_field:
        def stress(stokes, tagc):
            x, y = child.X
            v, p = stokes.Unknowns.u, stokes.Unknowns.p
            comps = {}
            for cname, expr in (
                    ("sxx", -p.sym[0] + 2 * common.ETA * v.sym[0].diff(x)),
                    ("syy", -p.sym[0] + 2 * common.ETA * v.sym[1].diff(y)),
                    ("sxy", common.ETA * (v.sym[0].diff(y)
                                          + v.sym[1].diff(x)))):
                s_var = uw.discretisation.MeshVariable(
                    f"{cname}_{tagc}", child, 1, degree=0,
                    continuous=False)
                proj = uw.systems.Projection(child, s_var)
                proj.uw_function = expr
                proj.smoothing = 0.0
                proj.solve()
                row = common.split_mesh_cell_rows(child, s_var)
                comps[cname] = np.asarray(s_var.data[:, 0])[row].copy()
            return comps

        c1 = stress(s1, "a")
        s0 = one_state(False)
        c0 = stress(s0, "b")
        beta = np.radians(TREND)
        nx, ny = -np.sin(beta), np.cos(beta)
        tx, ty = np.cos(beta), np.sin(beta)

        def resolve(c):
            s_nn = (c["sxx"] * nx * nx + 2 * c["sxy"] * nx * ny
                    + c["syy"] * ny * ny)
            s_t = (c["sxx"] * tx * nx + c["sxy"] * (tx * ny + ty * nx)
                   + c["syy"] * ty * ny)
            return s_nn, s_t

        nn0, t_0 = resolve(c0)
        nn1, t_1 = resolve(c1)
        dcff = np.sign(np.median(t_0)) * (t_1 - t_0) + MU_P * (nn1 - nn0)
        pts, faces = common.split_mesh_cell_render(child)
        fc = np.asarray(faces).reshape(-1, 4)[:, 1:]
        cent = np.asarray(pts)[fc].mean(axis=1)
        dcff, gauge = common.far_field_anchor(
            cent, dcff, [p for _n, p in faults], cut=0.18)
        print(f"[{tag}] gauge {gauge:+.4f}", flush=True)
        dcff_pack = (pts, faces, dcff)

    print(f"[{tag}] trunk peak slip {profile[1].max():.4f} "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)
    return profile, dcff_pack, faults


# case i: trunk through-going (master)
prep_i, _ = prepare_fault_network(
    [(n, p.copy()) for n, p in (M_RAW, S_RAW, C_RAW)], spacing=H,
    ligament=LIG_F, through=["Trunk"], verbose=True)
weld_i = [n for n, _p in prep_i if n.startswith("Conj")]
prof_i, field_i, faults_i = run_case("through", prep_i, weld_i, True)

# case ii: default — the crossing cuts the trunk
prep_ii, _ = prepare_fault_network(
    [(n, p.copy()) for n, p in (M_RAW, S_RAW, C_RAW)], spacing=H,
    ligament=LIG_F, verbose=True)
weld_ii = [n for n, _p in prep_ii if n.startswith("Conj")]
prof_ii, field_ii, faults_ii = run_case("cut", prep_ii, weld_ii, True)

# reference: the trunk alone
prof_ref, _, _ = run_case("isolated", [("Trunk", M_RAW[1].copy())],
                          [], False)


def render(pack, faults, png):
    pts, faces, dcff = pack
    pvm = pv.PolyData(np.asarray(pts, dtype=float),
                      faces=np.asarray(faces, dtype=np.int64))
    pvm.cell_data["dcff"] = dcff
    pl = pv.Plotter(off_screen=True, window_size=(900, 850))
    pl.set_background("white")
    pl.add_mesh(pvm, scalars="dcff", cmap="RdBu_r", clim=(-1.0, 1.0),
                lighting=False, show_scalar_bar=False)
    for n, p in faults:
        line = pv.lines_from_points(
            np.column_stack([p, np.full(len(p), 1e-3)]))
        pl.add_mesh(line, color=("#6a1b9a" if n.startswith("Conj")
                                 else "black"),
                    line_width=5 if n.startswith("Trunk") else 4,
                    lighting=False)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.42
    pl.camera.focal_point = (0.5, 0.52, 0.0)
    pl.screenshot(png)
    pl.close()
    return png


png_i = render(field_i, faults_i, os.path.join(D, "_bcmp_through.png"))
png_ii = render(field_ii, faults_ii, os.path.join(D, "_bcmp_cut.png"))

fig = plt.figure(figsize=(13.0, 5.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, 1.0, 0.05])
axp = fig.add_subplot(gs[0, 0])
axp.plot(*prof_ref, "k--", lw=1.3, label="trunk alone (continuous)")
axp.plot(*prof_i, "o-", ms=2.5, lw=1.0, color="#1565c0",
         label="network, trunk through-going")
axp.plot(*prof_ii, "o-", ms=2.5, lw=1.0, color="#e65100",
         label="network, trunk cut at the crossing")
axp.set_xlabel("arc length along the trunk")
axp.set_ylabel("|slip|")
axp.set_title("what the interruption costs the trunk", fontsize=10)
axp.legend(fontsize=8)
for png, col, title in ((png_i, 1, "trunk through-going"),
                        (png_ii, 2, "trunk cut at the crossing")):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(plt.imread(png))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
from matplotlib import cm, colors as mcolors
sm = cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap="RdBu_r")
cax = fig.add_subplot(gs[0, 3])
fig.colorbar(sm, cax=cax, label=r"$\Delta$CFF")
fig.suptitle("Continuous vs interrupted: the same network, two "
             "junction policies", fontsize=11.5)
fig.tight_layout()
out = os.path.join(D, "branching-compare.png")
fig.savefig(out, dpi=200)
print("wrote", out, flush=True)
