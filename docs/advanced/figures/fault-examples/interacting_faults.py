"""Interacting faults, King-style: stress transfer read on the Mohr plane.

Two en echelon faults (right-stepping, overlapping). The SOURCE slips
freely; the RECEIVER is welded — a passive per-node stress probe, the
instrument built in the Mohr examples. Two solves on the SAME mesh:

  (0) both faults welded          -> the ambient regional state
  (1) source slips, receiver weld -> the perturbed state

The difference is the classical Coulomb stress transfer:

- the FIELD: Delta CFF = tau_dir * d sigma_xy + mu' * d sigma_yy on
  receiver-parallel planes (King's mu' = 0.4), the familiar lobes;
- ON the receiver: each node's probe MOVES in the (sigma, tau) plane,
  and the displacement decomposes the CFF change into its shear and
  normal(unclamping) parts — Coulomb stress transfer in the diagram
  students already know how to read.

Rotating the regional compression axis phi (the boundary velocities
rotate with it) changes the interaction pattern; the phi sweep shows
the receiver cloud pushed toward failure under one orientation and
away under another.
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
MU_P = 0.4                              # King's effective friction
TAU0 = 1.0
ETA_WELD = 200.0 * common.ETA / 0.18
PHIS = (20.0, 45.0, 70.0)               # regional compression axis

SOURCE = np.array([[0.22, 0.46], [0.58, 0.46]])
RECEIVER = np.array([[0.42, 0.54], [0.78, 0.54]])
T_HAT = np.array([1.0, 0.0])            # both faults horizontal


def solve_pair(phi, source_free):
    child = common.base_mesh(0.04).add_fault(
        [("Source", SOURCE), ("Receiver", RECEIVER)])
    stokes = common.stokes_on(child, common.pure_shear_drive(child, phi,
                                                             TAU0))
    stokes.add_fault_bc(0 if source_free else ETA_WELD, boundary="Source")
    stokes.add_fault_bc(ETA_WELD, boundary="Receiver")
    fault_contact.solve_with_fault(stokes, picard=2)
    s, xy, sig, tau = common.probe_nodes(stokes, "Receiver", T_HAT,
                                         ETA_WELD)
    return child, stokes, (s, xy, sig, tau)


def stress_fields(child, stokes):
    """sigma_xy and sigma_yy (full stress, pressure included) projected
    to continuous P1 — the components CFF on horizontal planes needs."""
    x, y = child.X
    v = stokes.Unknowns.u
    p = stokes.Unknowns.p
    out = []
    for name, expr in (("sxy", common.ETA * (v.sym[0].diff(y)
                                             + v.sym[1].diff(x))),
                       ("syy", -p.sym[0] + 2 * common.ETA
                        * v.sym[1].diff(y))):
        s_var = uw.discretisation.MeshVariable(
            f"{name}_{stress_fields.counter}", child, 1, degree=1)
        proj = uw.systems.Projection(child, s_var)
        proj.uw_function = expr
        proj.smoothing = 0.0
        proj.solve()
        out.append(np.asarray(s_var.data[:, 0]).copy())
    stress_fields.counter += 1
    return out, s_var                    # arrays share the P1 layout


stress_fields.counter = 0

cache = os.path.join(D, "_interacting_probes.npz")
if os.path.exists(cache):
    data = dict(np.load(cache, allow_pickle=True))
    print("loaded cached sweep")
else:
    data = {}
    for phi in PHIS:
        _c0, s0, probe0 = solve_pair(phi, source_free=False)
        child, s1, probe1 = solve_pair(phi, source_free=True)
        data[f"probe0_{phi}"] = np.array(probe0[0]), probe0[1], probe0[2], \
            probe0[3]
        data[f"p0_s_{phi}"], data[f"p0_xy_{phi}"] = probe0[0], probe0[1]
        data[f"p0_sig_{phi}"], data[f"p0_tau_{phi}"] = probe0[2], probe0[3]
        data[f"p1_sig_{phi}"], data[f"p1_tau_{phi}"] = probe1[2], probe1[3]
        if phi == 45.0:
            # the Delta CFF FIELD needs both solves' stress components on
            # one mesh: redo the welded reference ON the slipping child
            (sxy1, syy1), s_var = stress_fields(child, s1)
            s0b = common.stokes_on(child, common.pure_shear_drive(
                child, phi, TAU0))
            s0b.add_fault_bc(ETA_WELD, boundary="Source")
            s0b.add_fault_bc(ETA_WELD, boundary="Receiver")
            fault_contact.solve_with_fault(s0b, picard=2)
            (sxy0, syy0), _ = stress_fields(child, s0b)
            tau_dir = np.sign(np.median(data[f"p0_tau_{phi}"]))
            dcff = tau_dir * (sxy1 - sxy0) + MU_P * (syy1 - syy0)
            pvm = vis.meshVariable_to_pv_mesh_object(s_var)
            data["field_points"] = np.asarray(pvm.points)
            data["field_dcff"] = dcff
        print(f"phi {phi}: receiver ambient tau "
              f"{np.median(data[f'p0_tau_{phi}']):+.3f}, sigma_n "
              f"{np.median(data[f'p0_sig_{phi}']):+.3f}")
    np.savez(cache, **{k: v for k, v in data.items()
                       if not k.startswith("probe0")})
    data = dict(np.load(cache, allow_pickle=True))

# ---- figure A: the field + the receiver's Mohr move (phi = 45) -----------
# gauge the difference to the far field (see common.far_field_anchor)
dcff_field, GAUGE_C = common.far_field_anchor(
    data["field_points"], data["field_dcff"], (SOURCE, RECEIVER))
print(f"far-field gauge constant removed: {GAUGE_C:+.4f}")
pvm = pv.PolyData(np.asarray(data["field_points"], dtype=float))
pvm.point_data["dcff"] = dcff_field
pvm = pvm.delaunay_2d()
pl = pv.Plotter(off_screen=True, window_size=(900, 700))
pl.set_background("white")
lim = 0.4 * TAU0
pl.add_mesh(pvm, scalars="dcff", cmap="RdBu_r", clim=(-lim, lim),
            show_edges=False, lighting=False,
            scalar_bar_args=dict(title="dCFF", color="black"))
for pts, col, w in ((SOURCE, "black", 4.0), (RECEIVER, "#1a6b1a", 4.0)):
    line = pv.Line(tuple(pts[0]) + (0.001,), tuple(pts[1]) + (0.001,))
    pl.add_mesh(line, color=col, line_width=w, lighting=False)
pl.view_xy()
pl.camera.parallel_projection = True
pl.camera.parallel_scale = 0.30
pl.camera.focal_point = (0.5, 0.5, 0.0)
field_png = os.path.join(D, "_interacting_field.png")
pl.screenshot(field_png)
pl.close()


def mohr_panel(ax, phi, legend=False):
    # anchor the welded probes to the analytic ambient for this drive,
    # the after-probes to the far-field-anchored difference on top
    c0 = float(np.median(data[f"p0_sig_{phi}"])
               - common.ambient_sigma_n(phi, T_HAT, TAU0))
    sig0 = data[f"p0_sig_{phi}"] - c0
    tau0 = data[f"p0_tau_{phi}"]
    sig1 = data[f"p1_sig_{phi}"] - c0 - GAUGE_C / MU_P
    tau1 = data[f"p1_tau_{phi}"]
    tau_dir = np.sign(np.median(tau0))
    dcff = tau_dir * (tau1 - tau0) + MU_P * (sig1 - sig0)
    sc0, sc1 = -sig0, -sig1              # geo convention
    ss = np.linspace(0, 2.0, 40)
    for sgn in (+1, -1):
        ax.plot(ss, sgn * MU_P * ss, "--", color="0.55", lw=0.9,
                label=(r"$\tau = \pm\mu'\sigma$ trend" if sgn > 0
                       and legend else None))
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(TAU0 * np.cos(tt), TAU0 * np.sin(tt), "-", color="0.85",
            lw=0.8, label="regional circle" if legend else None)
    ax.scatter(sc0, tau0, s=16, facecolors="none", edgecolors="0.55",
               linewidths=1.0,
               label="receiver nodes, before" if legend else None)
    for k in range(0, len(sc0), 2):
        ax.annotate("", xytext=(sc0[k], tau0[k]), xy=(sc1[k], tau1[k]),
                    arrowprops=dict(arrowstyle="->", lw=0.6,
                                    color="0.65"))
    pts = ax.scatter(sc1, tau1, c=dcff, cmap="RdBu_r", s=26,
                     vmin=-0.4, vmax=0.4, zorder=5, edgecolors="0.3",
                     linewidths=0.3,
                     label="after (colour: dCFF)" if legend else None)
    ax.axhline(0, color="0.9", lw=0.5)
    ax.axvline(0, color="0.9", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title(rf"$\phi = {phi:.0f}°$  "
                 rf"(median dCFF {np.median(dcff):+.2f})", fontsize=9)
    ax.set_xlabel(r"$\sigma$ (compression +)", fontsize=8)
    return pts


fig = plt.figure(figsize=(11.5, 4.9))
axf = fig.add_subplot(1, 2, 1)
axf.imshow(plt.imread(field_png))
axf.set_xticks([])
axf.set_yticks([])
axf.set_title(r"$\Delta$CFF field ($\mu' = 0.4$), source slips freely;"
              "\nreceiver (green) welded as a probe", fontsize=9)
axm = fig.add_subplot(1, 2, 2)
pts = mohr_panel(axm, 45.0, legend=True)
axm.set_ylabel(r"$\tau$", fontsize=9)
axm.legend(fontsize=7, loc="lower right")
fig.colorbar(pts, ax=axm, label=r"node $\Delta$CFF", shrink=0.8)
fig.suptitle("En echelon interaction: the receiver's probes move in the "
             "Mohr plane", fontsize=11)
fig.tight_layout()
out = os.path.join(D, "interacting-faults.png")
fig.savefig(out, dpi=200)
print("wrote", out)

# ---- figure B: rotate the regional stress --------------------------------
fig, axes = plt.subplots(1, len(PHIS), figsize=(11.5, 4.2),
                         sharey=True)
for ax, phi in zip(axes, PHIS):
    pts = mohr_panel(ax, phi, legend=(phi == PHIS[0]))
axes[0].set_ylabel(r"$\tau$", fontsize=9)
axes[0].legend(fontsize=6.5, loc="lower right")
fig.colorbar(pts, ax=axes, label=r"node $\Delta$CFF", shrink=0.8)
fig.suptitle("Rotating the regional compression axis changes the "
             "interaction", fontsize=11)
out = os.path.join(D, "interacting-rotation.png")
fig.savefig(out, dpi=200)
print("wrote", out)
