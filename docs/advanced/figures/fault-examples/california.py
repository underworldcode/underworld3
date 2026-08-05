"""A California-like planform: one curved trunk fault, four minor faults.

The trunk (a kinked polyline — every control point is pulled onto a
mesh vertex by add_fault) slips freely under a regional compression
oriented to drive it; the minor faults are welded probes at different
positions and orientations. One slip event, four different verdicts:
minors in the tip lobes are pushed TOWARD failure (red), minors
broadside of the slipped section are relaxed (blue) — the sign flip
that the en echelon example's fixed geometry cannot show.

Field: Delta CFF on trunk-parallel planes (mu' = 0.4). Probes: each
minor's per-node before/after cloud in the Mohr plane, coloured by its
own Delta CFF (computed in the minor's OWN orientation).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.utilities import fault_contact

import common

pv.OFF_SCREEN = True
D = os.path.dirname(os.path.abspath(__file__))
MU_P = 0.4
TAU0 = 1.0
PHI = 70.0                     # compression axis: drives the ~25 deg trunk
ETA_WELD = 200.0 * common.ETA / 0.2
# declared confining pressure + cohesion: neither changes Delta CFF,
# they place the failure envelope over a fully compressive circle
P0 = 1.0
COH = 0.75

TRUNK = np.array([[0.15, 0.35], [0.30, 0.42], [0.45, 0.475],
                  [0.60, 0.52], [0.72, 0.555], [0.85, 0.62]])
MINORS = {
    "m1": np.array([[0.42, 0.62], [0.58, 0.66]]),     # broadside above
    "m2": np.array([[0.86, 0.70], [0.94, 0.76]]),     # off the far tip
    "m3": np.array([[0.25, 0.20], [0.40, 0.22]]),     # broadside below
    "m4": np.array([[0.76, 0.40], [0.84, 0.33]]),     # conjugate, in the shadow
}
TRUNK_STRIKE = np.arctan2(TRUNK[-1, 1] - TRUNK[0, 1],
                          TRUNK[-1, 0] - TRUNK[0, 0])


def build_and_solve(trunk_free):
    faults = [("Trunk", TRUNK)] + [(k, v) for k, v in MINORS.items()]
    child = common.base_mesh(0.035).add_fault(faults)
    stokes = common.stokes_on(child, common.pure_shear_drive(child, PHI,
                                                             TAU0))
    stokes.add_fault_bc(0 if trunk_free else ETA_WELD, boundary="Trunk")
    for k in MINORS:
        stokes.add_fault_bc(ETA_WELD, boundary=k)
    fault_contact.solve_with_fault(stokes, picard=2)
    probes = {}
    for k, pts in MINORS.items():
        t_hat = pts[1] - pts[0]
        s, xy, sig, tau = common.probe_nodes(stokes, k, t_hat, ETA_WELD)
        probes[k] = (sig, tau)
    return child, stokes, probes


def stress_components(child, stokes, tag):
    x, y = child.X
    v = stokes.Unknowns.u
    p = stokes.Unknowns.p
    exprs = dict(
        sxx=-p.sym[0] + 2 * common.ETA * v.sym[0].diff(x),
        syy=-p.sym[0] + 2 * common.ETA * v.sym[1].diff(y),
        sxy=common.ETA * (v.sym[0].diff(y) + v.sym[1].diff(x)))
    out = {}
    for name, expr in exprs.items():
        s_var = uw.discretisation.MeshVariable(f"{name}_{tag}", child, 1,
                                               degree=1)
        proj = uw.systems.Projection(child, s_var)
        proj.uw_function = expr
        proj.smoothing = 0.0
        proj.solve()
        out[name] = np.asarray(s_var.data[:, 0]).copy()
    return out, s_var


cache = os.path.join(D, "_california_probes.npz")
if os.path.exists(cache):
    data = dict(np.load(cache, allow_pickle=True))
    print("loaded cached run")
else:
    child, s1, probes1 = build_and_solve(trunk_free=True)
    comp1, s_var = stress_components(child, s1, "a")
    s0 = common.stokes_on(child, common.pure_shear_drive(child, PHI,
                                                         TAU0))
    s0.add_fault_bc(ETA_WELD, boundary="Trunk")
    for k in MINORS:
        s0.add_fault_bc(ETA_WELD, boundary=k)
    fault_contact.solve_with_fault(s0, picard=2)
    comp0, _ = stress_components(child, s0, "b")
    probes0 = {}
    for k, pts in MINORS.items():
        t_hat = pts[1] - pts[0]
        _s, _xy, sig, tau = common.probe_nodes(s0, k, t_hat, ETA_WELD)
        probes0[k] = (sig, tau)

    # Delta CFF on trunk-parallel planes
    nx, ny = -np.sin(TRUNK_STRIKE), np.cos(TRUNK_STRIKE)
    tx, ty = np.cos(TRUNK_STRIKE), np.sin(TRUNK_STRIKE)

    def resolve(c):
        s_nn = (c["sxx"] * nx * nx + 2 * c["sxy"] * nx * ny
                + c["syy"] * ny * ny)
        s_t = (c["sxx"] * tx * nx + c["sxy"] * (tx * ny + ty * nx)
               + c["syy"] * ty * ny)
        return s_nn, s_t

    nn0, t0 = resolve(comp0)
    nn1, t1 = resolve(comp1)
    tau_dir = np.sign(np.median(t0))
    data = dict(field_dcff=tau_dir * (t1 - t0) + MU_P * (nn1 - nn0),
                field_points=np.asarray(
                    vis.meshVariable_to_pv_mesh_object(s_var).points))
    for k in MINORS:
        data[f"{k}_sig0"], data[f"{k}_tau0"] = probes0[k]
        data[f"{k}_sig1"], data[f"{k}_tau1"] = probes1[k]
    np.savez(cache, **data)
    data = dict(np.load(cache, allow_pickle=True))

# ---- the field render ------------------------------------------------------
dcff_field, GAUGE_C = common.far_field_anchor(
    data["field_points"], data["field_dcff"],
    [TRUNK] + list(MINORS.values()), cut=0.22)
print(f"far-field gauge constant removed: {GAUGE_C:+.4f}")
pvm = pv.PolyData(np.asarray(data["field_points"], dtype=float))
pvm.point_data["dcff"] = dcff_field
pvm = pvm.delaunay_2d()
pl = pv.Plotter(off_screen=True, window_size=(950, 850))
pl.set_background("white")
lim = 0.4 * TAU0
pl.add_mesh(pvm, scalars="dcff", cmap="RdBu_r", clim=(-lim, lim),
            show_edges=False, lighting=False,
            scalar_bar_args=dict(title="dCFF", color="black"))


def polyline(pts):
    line = pv.lines_from_points(
        np.column_stack([pts, np.full(len(pts), 0.001)]))
    return line


pl.add_mesh(polyline(TRUNK), color="black", line_width=4.5,
            lighting=False)
minor_cols = {"m1": "#1a6b1a", "m2": "#6a1b9a", "m3": "#00695c",
              "m4": "#e65100"}
for k, pts in MINORS.items():
    pl.add_mesh(polyline(pts), color=minor_cols[k], line_width=4.0,
                lighting=False)
pl.view_xy()
pl.camera.parallel_projection = True
pl.camera.parallel_scale = 0.42
pl.camera.focal_point = (0.52, 0.47, 0.0)
field_png = os.path.join(D, "_california_field.png")
pl.screenshot(field_png)
pl.close()

# ---- the figure: field + one mini-Mohr per minor ---------------------------
fig = plt.figure(figsize=(12.6, 6.2))
gs = fig.add_gridspec(2, 4, width_ratios=[2.4, 1, 1, 0.12])

axf = fig.add_subplot(gs[:, 0])
axf.imshow(plt.imread(field_png))
axf.set_xticks([])
axf.set_yticks([])
axf.set_title(r"$\Delta$CFF on trunk-parallel planes ($\mu' = 0.4$);"
              "\ntrunk slips freely, minors welded as probes",
              fontsize=9)

panels = [("m1", 0, 1), ("m2", 0, 2), ("m3", 1, 1), ("m4", 1, 2)]
for k, r, ccol in panels:
    ax = fig.add_subplot(gs[r, ccol])
    # each solve carries its own pressure-gauge constant; anchor the
    # welded probes to the ANALYTIC ambient normal stress for this
    # minor's orientation, and the after-probes to the (far-field
    # anchored) difference on top of that
    t_hat = MINORS[k][1] - MINORS[k][0]
    c0 = float(np.median(data[f"{k}_sig0"])
               - common.ambient_sigma_n(PHI, t_hat, TAU0))
    sig0 = data[f"{k}_sig0"] - c0
    tau0 = data[f"{k}_tau0"]
    sig1 = data[f"{k}_sig1"] - c0 - GAUGE_C / MU_P
    tau1 = data[f"{k}_tau1"]
    tau_dir = np.sign(np.median(tau0))
    dcff = tau_dir * (tau1 - tau0) + MU_P * (sig1 - sig0)
    sc0, sc1 = P0 - sig0, P0 - sig1
    ss = np.linspace(-0.4, P0 + 1.8, 80)
    strength = np.maximum(COH + MU_P * ss, 0.0)
    for sgn in (+1, -1):
        ax.plot(ss, sgn * strength, "-", color="0.4", lw=0.9)
    ax.fill_between(ss, strength, 2.4, color="#c62828", alpha=0.06,
                    lw=0)
    ax.fill_between(ss, -strength, -2.4, color="#c62828", alpha=0.06,
                    lw=0)
    tt = np.linspace(0, 2 * np.pi, 150)
    ax.plot(P0 + TAU0 * np.cos(tt), TAU0 * np.sin(tt), "-",
            color="0.88", lw=0.7)
    ax.scatter(sc0, tau0, s=10, facecolors="none", edgecolors="0.6",
               linewidths=0.8)
    for j in range(0, len(sc0), 2):
        ax.annotate("", xytext=(sc0[j], tau0[j]), xy=(sc1[j], tau1[j]),
                    arrowprops=dict(arrowstyle="->", lw=0.5,
                                    color="0.6"))
    pts = ax.scatter(sc1, tau1, c=dcff, cmap="RdBu_r", s=20,
                     vmin=-0.25, vmax=0.25, zorder=5,
                     edgecolors="0.3", linewidths=0.25)
    ax.axhline(0, color="0.92", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, P0 + 1.8)
    ax.set_ylim(-1.6, 1.6)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(minor_cols[k])
        spine.set_linewidth(1.8)
    ax.set_title(f"{k}: median $\\Delta$CFF {np.median(dcff):+.2f}",
                 fontsize=8, color=minor_cols[k])

cax = fig.add_subplot(gs[:, 3])
fig.colorbar(pts, cax=cax, label=r"node $\Delta$CFF")
fig.suptitle("One slip event, four verdicts: minor faults in the lobes "
             "load, broadside minors relax", fontsize=11)
fig.tight_layout()
out = os.path.join(D, "california.png")
fig.savefig(out, dpi=200)
print("wrote", out)
