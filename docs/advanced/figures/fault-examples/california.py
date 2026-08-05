"""A schematic southern California: the San Andreas slips, its
neighbours read the stress.

Geography (schematic, not to scale, x = east / y = north): the SAN
ANDREAS trunk runs NW-SE with its Big Bend; the GARLOCK heads ENE off
the bend; three EAST CALIFORNIA SHEAR ZONE strands and a SAN
JACINTO-like fault sit INBOARD (northeast of the SAF, in the
continental crust — not in the Pacific plate). The regional drive is
right-lateral SIMPLE SHEAR parallel to the plate-boundary trend
(~N40W), which resolves DEXTRAL on the NW-striking faults and
SINISTRAL on the Garlock — exactly the real senses, and the map's
half-arrows are drawn from the MEASURED slip, not assumed.

The SAF slips freely (the extreme event: the whole trunk drops its
shear); every other fault is welded — a per-node stress probe. Field:
Delta CFF on boundary-parallel planes, symmetric-log colour. Probes:
Garlock / pooled ECSZ / San Jacinto clouds against the cohesive
envelope (P0 = 1, C = 0.75 — neither enters Delta CFF).
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
TREND = 132.0                 # plate-boundary trend (~N42W), degrees CCW of E
ETA_WELD = 200.0 * common.ETA / 0.2
P0 = 1.0
COH = 0.75

SAF = np.array([[0.88, 0.06], [0.72, 0.24], [0.55, 0.36],
                [0.38, 0.44], [0.22, 0.58], [0.10, 0.78]])
MINORS = {
    "Garlock": np.array([[0.44, 0.47], [0.74, 0.57]]),
    "E1": np.array([[0.62, 0.64], [0.56, 0.78]]),
    "E2": np.array([[0.72, 0.62], [0.66, 0.76]]),
    "E3": np.array([[0.81, 0.68], [0.75, 0.82]]),
    "SJF": np.array([[0.84, 0.24], [0.70, 0.38]]),
}
GROUPS = (("Garlock (sinistral)", ("Garlock",), "#6a1b9a"),
          ("ECSZ, 3 strands (dextral)", ("E1", "E2", "E3"), "#1a6b1a"),
          ("San Jacinto (dextral)", ("SJF",), "#e65100"))
COLOUR = {"Garlock": "#6a1b9a", "E1": "#1a6b1a", "E2": "#1a6b1a",
          "E3": "#1a6b1a", "SJF": "#e65100"}


def build_and_solve(trunk_free):
    faults = [("SAF", SAF)] + [(k, v) for k, v in MINORS.items()]
    child = common.base_mesh(0.02).add_fault(faults)
    stokes = common.stokes_on(child,
                              common.boundary_simple_shear(child, TREND,
                                                           TAU0))
    stokes.add_fault_bc(0 if trunk_free else ETA_WELD, boundary="SAF")
    for k in MINORS:
        stokes.add_fault_bc(ETA_WELD, boundary=k)
    fault_contact.solve_with_fault(stokes, picard=2)
    probes = {}
    for k, pts in MINORS.items():
        _s, _xy, sig, tau = common.probe_nodes(stokes, k, pts[1] - pts[0],
                                               ETA_WELD)
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
    # the trunk's own slip: report the sense, drawn on the map
    t_saf = SAF[-1] - SAF[0]
    s_saf, V_saf = common.slip_vs_position(
        s1, t_saf, centre=SAF.mean(axis=0), name="SAF")
    comp1, s_var = stress_components(child, s1, "a")

    s0 = common.stokes_on(child,
                          common.boundary_simple_shear(child, TREND,
                                                       TAU0))
    s0.add_fault_bc(ETA_WELD, boundary="SAF")
    for k in MINORS:
        s0.add_fault_bc(ETA_WELD, boundary=k)
    fault_contact.solve_with_fault(s0, picard=2)
    comp0, _ = stress_components(child, s0, "b")
    probes0 = {}
    for k, pts in MINORS.items():
        _s, _xy, sig, tau = common.probe_nodes(s0, k, pts[1] - pts[0],
                                               ETA_WELD)
        probes0[k] = (sig, tau)

    # Delta CFF on boundary-parallel planes
    beta = np.radians(TREND)
    nx, ny = -np.sin(beta), np.cos(beta)
    tx, ty = np.cos(beta), np.sin(beta)

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
                    vis.meshVariable_to_pv_mesh_object(s_var).points),
                saf_v=V_saf)
    for k in MINORS:
        data[f"{k}_sig0"], data[f"{k}_tau0"] = probes0[k]
        data[f"{k}_sig1"], data[f"{k}_tau1"] = probes1[k]
    np.savez(cache, **data)
    data = dict(np.load(cache, allow_pickle=True))

t_hat_saf = (SAF[-1] - SAF[0]) / np.linalg.norm(SAF[-1] - SAF[0])
v_med = float(np.median(data["saf_v"]))
# dextral: an observer on the fault sees the far side move to the
# right. With the tangent pointing NW and the split's Plus side on its
# LEFT (SW, the Pacific side), a POSITIVE jump (v+ - v-) along +t means
# the Pacific side moves NW relative to North America — right-lateral.
sense = "right-lateral" if v_med > 0 else "LEFT-LATERAL?!"
print(f"SAF slip: median tangential jump {v_med:+.3f} ({sense})")

# ---- the field render ------------------------------------------------------
dcff_field, GAUGE_C = common.far_field_anchor(
    data["field_points"], data["field_dcff"],
    [SAF] + list(MINORS.values()), cut=0.18)
print(f"far-field gauge constant removed: {GAUGE_C:+.4f}")
pvm = pv.PolyData(np.asarray(data["field_points"], dtype=float))
LT = 0.02
pvm.point_data["dcff"] = common.signed_log(dcff_field, LT)
pvm = pvm.delaunay_2d()
pl = pv.Plotter(off_screen=True, window_size=(1000, 950))
pl.set_background("white")
lim = float(common.signed_log(0.5, LT))
pl.add_mesh(pvm, scalars="dcff", cmap="RdBu_r", clim=(-lim, lim),
            show_edges=False, lighting=False,
            annotations=common.signed_log_annotations(
                (-0.4, -0.1, -0.02, 0.0, 0.02, 0.1, 0.4), LT),
            scalar_bar_args=dict(title="dCFF (log scale)", color="black",
                                 n_labels=0))


def polyline(pts):
    return pv.lines_from_points(
        np.column_stack([pts, np.full(len(pts), 0.001)]))


pl.add_mesh(polyline(SAF), color="black", line_width=5.0, lighting=False)
for k, pts in MINORS.items():
    pl.add_mesh(polyline(pts), color=COLOUR[k], line_width=4.0,
                lighting=False)

# measured slip sense on the SAF, as half-arrows either side of mid-trunk
mid = 0.5 * (SAF[2] + SAF[3])
n_saf = np.array([-t_hat_saf[1], t_hat_saf[0]])
sense_sign = np.sign(v_med)
for pm in (+1.0, -1.0):
    base = mid + pm * 0.05 * n_saf - pm * sense_sign * 0.06 * t_hat_saf
    pl.add_arrows(np.array([np.append(base, 0.002)]),
                  np.array([np.append(pm * sense_sign * t_hat_saf, 0.0)]),
                  mag=0.11, color="black")

pl.add_point_labels(
    np.array([[0.26, 0.40, 0.002], [0.64, 0.49, 0.002],
              [0.58, 0.82, 0.002], [0.84, 0.20, 0.002]]),
    ["San Andreas", "Garlock", "ECSZ", "San Jacinto"],
    font_size=22, text_color="black", shape=None, always_visible=True,
    show_points=False)

pl.view_xy()
pl.camera.parallel_projection = True
pl.camera.parallel_scale = 0.48
pl.camera.focal_point = (0.48, 0.44, 0.0)
field_png = os.path.join(D, "_california_field.png")
pl.screenshot(field_png)
pl.close()

# ---- the figure: field + one Mohr panel per fault group --------------------
fig = plt.figure(figsize=(13.2, 6.6))
gs = fig.add_gridspec(3, 3, width_ratios=[2.6, 1.15, 0.06])

axf = fig.add_subplot(gs[:, 0])
axf.imshow(plt.imread(field_png))
axf.set_xticks([])
axf.set_yticks([])
axf.set_title(r"$\Delta$CFF on boundary-parallel planes ($\mu' = 0.4$);"
              "\nSAF slips (right-lateral), neighbours welded as probes"
              "\n(schematic geometry, not to scale)", fontsize=9)

for row, (label, members, col) in enumerate(GROUPS):
    ax = fig.add_subplot(gs[row, 1])
    ss = np.linspace(-0.4, P0 + 1.9, 80)
    strength = np.maximum(COH + MU_P * ss, 0.0)
    for sgn in (+1, -1):
        ax.plot(ss, sgn * strength, "-", color="0.4", lw=0.9)
    ax.fill_between(ss, strength, 2.6, color="#c62828", alpha=0.06, lw=0)
    ax.fill_between(ss, -strength, -2.6, color="#c62828", alpha=0.06,
                    lw=0)
    tt = np.linspace(0, 2 * np.pi, 150)
    ax.plot(P0 + TAU0 * np.cos(tt), TAU0 * np.sin(tt), "-", color="0.88",
            lw=0.7)
    medians = []
    for k in members:
        t_hat = MINORS[k][1] - MINORS[k][0]
        c0 = float(np.median(data[f"{k}_sig0"])
                   - common.ambient_sigma_n_simple(TREND, t_hat, TAU0))
        sig0 = data[f"{k}_sig0"] - c0
        tau0 = data[f"{k}_tau0"]
        sig1 = data[f"{k}_sig1"] - c0 - GAUGE_C / MU_P
        tau1 = data[f"{k}_tau1"]
        tau_dir = np.sign(np.median(tau0))
        dcff = tau_dir * (tau1 - tau0) + MU_P * (sig1 - sig0)
        medians.append(np.median(dcff))
        sc0, sc1 = P0 - sig0, P0 - sig1
        ax.scatter(sc0, tau0, s=9, facecolors="none", edgecolors="0.6",
                   linewidths=0.7)
        for j in range(0, len(sc0), 3):
            ax.annotate("", xytext=(sc0[j], tau0[j]),
                        xy=(sc1[j], tau1[j]),
                        arrowprops=dict(arrowstyle="->", lw=0.45,
                                        color="0.6"))
        pts = ax.scatter(sc1, tau1, c=dcff, cmap="RdBu_r", s=16,
                         vmin=-0.3, vmax=0.3, zorder=5,
                         edgecolors="0.3", linewidths=0.2)
    ax.axhline(0, color="0.92", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, P0 + 1.9)
    ax.set_ylim(-1.7, 1.7)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(col)
        spine.set_linewidth(1.8)
    ax.set_title(f"{label}: median $\\Delta$CFF "
                 f"{np.median(medians):+.2f}", fontsize=8.5, color=col)

cax = fig.add_subplot(gs[:, 2])
fig.colorbar(pts, cax=cax, label=r"node $\Delta$CFF")
fig.suptitle("A San Andreas slip event, read by its neighbours",
             fontsize=11.5)
fig.tight_layout()
out = os.path.join(D, "california.png")
fig.savefig(out, dpi=200)
print("wrote", out)
