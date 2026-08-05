"""A schematic southern California: the San Andreas with a smooth
restraining bend (the Big Bend), read by its neighbours.

Geography (schematic, not to scale, x = east / y = north): the SAN
ANDREAS is ONE continuous dextral trace with a smooth tanh S-bend — the
smoothed version of a left-stepping stepover. Under right-lateral shear
the bend is RESTRAINING: the bend zone is where the TRANSVERSE RANGES
belong, right beside the Garlock. The GARLOCK (resolving sinistral —
the real sense, from the kinematics), three EAST CALIFORNIA SHEAR ZONE
strands and a SAN JACINTO-like fault sit inboard as welded probes.

The curved trace is sampled as a polyline, and the fault's constraint
frame uses the curve's ANALYTIC normal (``add_fault_bc(normal=...)``).
Without it the per-node normal averages the adjacent facet normals and
zig-zags at the sampling kinks; the no-opening constraint then forbids
smooth slip past each kink — sawtooth tractions that GROW under mesh
refinement (measured: ~/+Simulations/curved_fault_roughness/). With the
analytic normal the polyline behaves as the smooth fault it represents.

The San Andreas slips freely under right-lateral simple shear parallel
to the plate-boundary trend; the slip half-arrows on the map come from
the MEASURED jump. Field: Delta CFF on boundary-parallel planes, P0
(cell) stress — continuous-P1 projection of the rough near-fault stress
rings at the node scale (measured) — rendered on the split mesh's true
connectivity.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy
import pyvista as pv

import underworld3 as uw
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

# One continuous dextral trace with a smooth restraining bend: the
# centreline runs along the trend from A, offset by a tanh step of
# BEND_W toward the SW (the CCW normal) — an S in map view, stepping
# LEFT, so right-lateral slip must converge through the bend. Max
# deviation from the trend: atan(BEND_W / (2 LAM)) ~ 30 degrees, about
# the real Big Bend.
_A = np.array([0.88, 0.06])
_t = np.array([np.cos(np.radians(TREND)), np.sin(np.radians(TREND))])
_n = np.array([-_t[1], _t[0]])                  # CCW normal (points SW)
S_END = 0.94
BEND_W = 0.07                                   # total SW offset
BEND_S0 = 0.45                                  # bend centre, arc parameter
LAM = 0.06                                      # bend half-width


def _w(s):
    return 0.5 * BEND_W * (1.0 + np.tanh((s - BEND_S0) / LAM))


def saf_trace(n_seg=47):
    """The smooth trace sampled as a polyline (kinks land on mesh
    vertices; the analytic normal makes them harmless)."""
    s = np.linspace(0.0, S_END, n_seg + 1)
    return _A + np.outer(s, _t) + np.outer(_w(s), _n)


def saf_normal(child):
    """The EXACT unit-normal direction of the smooth trace as a sympy
    row matrix in mesh coordinates: with X(s) = A + s t + w(s) n and
    s(X) = (X - A).t, the tangent is t + w'(s) n and the normal its
    quarter turn, n - w'(s) t (normalisation is the caller's)."""
    x, y = child.X
    s = (x - _A[0]) * _t[0] + (y - _A[1]) * _t[1]
    wp = (0.5 * BEND_W / LAM) * (1 - sympy.tanh((s - BEND_S0) / LAM) ** 2)
    return sympy.Matrix([[_n[0] - wp * _t[0], _n[1] - wp * _t[1]]])


SAF_PTS = saf_trace()

MINORS = {
    "Garlock": np.array([[0.52, 0.44], [0.80, 0.53]]),
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
    faults = [("SAF", SAF_PTS)] + [(k, v) for k, v in MINORS.items()]
    child = common.base_mesh(0.012).add_fault(faults)
    stokes = common.stokes_on(child,
                              common.boundary_simple_shear(child, TREND,
                                                           TAU0))
    # the analytic normal in BOTH states, so the slipping and welded
    # solves share one constraint frame and difference cleanly
    stokes.add_fault_bc(0 if trunk_free else ETA_WELD, boundary="SAF",
                        normal=saf_normal(child))
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
        # P0 (cell) stress: projecting the rough near-fault stress onto
        # CONTINUOUS P1 rings at the node scale (measured: residual rms
        # 0.26 at half-wavelength h/2); cellwise averages are honest and
        # pixel-scale at this resolution.
        s_var = uw.discretisation.MeshVariable(f"{name}_{tag}", child, 1,
                                               degree=0, continuous=False)
        proj = uw.systems.Projection(child, s_var)
        proj.uw_function = expr
        proj.smoothing = 0.0
        proj.solve()
        row = common.split_mesh_cell_rows(child, s_var)
        out[name] = np.asarray(s_var.data[:, 0])[row].copy()
    return out


cache = os.path.join(D, "_california_probes.npz")
if os.path.exists(cache):
    data = dict(np.load(cache, allow_pickle=True))
    print("loaded cached run")
else:
    t_wall = time.perf_counter()
    child, s1, probes1 = build_and_solve(trunk_free=True)
    print(f"[timing] slipping solve + probes: "
          f"{time.perf_counter() - t_wall:.1f} s")
    t_wall = time.perf_counter()
    s_saf, V_saf = common.slip_vs_position(
        s1, _t, centre=_A + 0.47 * _t, name="SAF")
    comp1 = stress_components(child, s1, "a")

    s0 = common.stokes_on(child,
                          common.boundary_simple_shear(child, TREND,
                                                       TAU0))
    s0.add_fault_bc(ETA_WELD, boundary="SAF", normal=saf_normal(child))
    for k in MINORS:
        s0.add_fault_bc(ETA_WELD, boundary=k)
    fault_contact.solve_with_fault(s0, picard=2)
    comp0 = stress_components(child, s0, "b")
    print(f"[timing] welded solve + all projections: "
          f"{time.perf_counter() - t_wall:.1f} s")
    probes0 = {}
    for k, pts in MINORS.items():
        _s, _xy, sig, tau = common.probe_nodes(s0, k, pts[1] - pts[0],
                                               ETA_WELD)
        probes0[k] = (sig, tau)

    # Delta CFF on boundary-parallel planes (per cell)
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
    _pts, _faces = common.split_mesh_cell_render(child)
    # cell centroids, for the far-field gauge anchor
    fc = np.asarray(_faces).reshape(-1, 4)[:, 1:]
    _cent = np.asarray(_pts)[fc].mean(axis=1)
    data = dict(field_dcff=tau_dir * (t1 - t0) + MU_P * (nn1 - nn0),
                field_points=_pts, field_faces=_faces,
                field_centroids=_cent, saf_v=V_saf)
    for k in MINORS:
        data[f"{k}_sig0"], data[f"{k}_tau0"] = probes0[k]
        data[f"{k}_sig1"], data[f"{k}_tau1"] = probes1[k]
    np.savez(cache, **data)
    data = dict(np.load(cache, allow_pickle=True))

v_med = float(np.median(data["saf_v"]))
# dextral: with the tangent pointing NW and the split's Plus side on
# its LEFT (SW, the Pacific side), a POSITIVE jump (v+ - v-) along +t
# means the Pacific side moves NW relative to North America.
sense = "right-lateral" if v_med > 0 else "LEFT-LATERAL?!"
print(f"SAF slip: median tangential jump {v_med:+.3f} ({sense})")

# ---- the field render ------------------------------------------------------
dcff_field, GAUGE_C = common.far_field_anchor(
    data["field_centroids"], data["field_dcff"],
    [SAF_PTS[:26], SAF_PTS[24:]] + list(MINORS.values()), cut=0.18)
print(f"far-field gauge constant removed: {GAUGE_C:+.4f}")
pvm = pv.PolyData(np.asarray(data["field_points"], dtype=float),
                  faces=np.asarray(data["field_faces"], dtype=np.int64))
pvm.cell_data["dcff"] = dcff_field
pl = pv.Plotter(off_screen=True, window_size=(1000, 950))
pl.set_background("white")
pl.add_mesh(pvm, scalars="dcff", cmap="RdBu_r", clim=(-1.0, 1.0),
            show_edges=False, lighting=False,
            scalar_bar_args=dict(title="dCFF", color="black"))


def polyline(pts):
    return pv.lines_from_points(
        np.column_stack([pts, np.full(len(pts), 0.001)]))


pl.add_mesh(polyline(SAF_PTS), color="black", line_width=5.0,
            lighting=False)
for k, pts in MINORS.items():
    pl.add_mesh(polyline(pts), color=COLOUR[k], line_width=4.0,
                lighting=False)

# measured slip sense, half-arrows either side of the southern leg
mid = _A + 0.20 * _t
sense_sign = np.sign(v_med)
for pm in (+1.0, -1.0):
    base = mid + pm * 0.05 * _n - pm * sense_sign * 0.06 * _t
    pl.add_arrows(np.array([np.append(base, 0.002)]),
                  np.array([np.append(pm * sense_sign * _t, 0.0)]),
                  mag=0.11, color="black")

pl.add_point_labels(
    np.array([[0.30, 0.44, 0.002], [0.86, 0.12, 0.002],
              [0.68, 0.50, 0.002], [0.60, 0.82, 0.002],
              [0.84, 0.20, 0.002], [0.40, 0.24, 0.002]]),
    ["San Andreas (N)", "San Andreas (S)", "Garlock", "ECSZ",
     "San Jacinto", "Transverse Ranges\n(restraining bend)"],
    font_size=20, text_color="black", shape=None, always_visible=True,
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
              "\nthe San Andreas slips (right-lateral) through its "
              "restraining bend, neighbours welded as probes\n"
              "(schematic geometry, not to scale)",
              fontsize=9)

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
fig.suptitle("A San Andreas slip event (Big Bend), read by its "
             "neighbours", fontsize=11.5)
fig.tight_layout()
out = os.path.join(D, "california.png")
fig.savefig(out, dpi=200)
print("wrote", out)
