"""The fault-strength ladder: one fault, one drive, four laws.

A horizontal fault under far-field simple shear (resolved shear stress
tau_infty = eta * rate = 1 on the fault plane), solved with each rung of
the constitutive ladder. Left: the slip profiles V(s), with the exact
zero-slip tips appended (positions read through the DOF pairing — the
generic fault_slip arc-length starts at the first pair, not the tip).
Right: the shear-stress field sigma_xy for each rung, stacked for
context — the stress drop shadows the fault where it slips, and the
welded/stuck rungs leave the far-field stress untouched.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.utilities import fault_contact

import common

pv.OFF_SCREEN = True
D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
SIGMA_N = 2.0                       # prescribed normal stress for friction

segment = common.fault_segment(0.0, half_length=HALF)
runs = []


def render_shear_stress(stokes, child, tag):
    """sigma_xy projected to continuous P1 (smoothing 0), rendered as a
    cropped planform panel with the fault trace overlaid."""
    x, y = child.X
    v = stokes.Unknowns.u
    s_var = uw.discretisation.MeshVariable(f"Sxy_{tag}", child, 1, degree=1)
    proj = uw.systems.Projection(child, s_var)
    proj.uw_function = common.ETA * (v.sym[0].diff(y) + v.sym[1].diff(x))
    proj.smoothing = 0.0
    proj.solve()

    pv_m = vis.meshVariable_to_pv_mesh_object(s_var)
    pv_m.point_data["S"] = np.asarray(s_var.data[:, 0])
    pl = pv.Plotter(off_screen=True, window_size=(560, 340))
    pl.set_background("white")
    pl.add_mesh(pv_m, scalars="S", cmap="RdBu_r", clim=(0.0, 2.0),
                show_edges=False, lighting=False, show_scalar_bar=False)
    trace = pv.Line((0.5 - HALF, 0.5, 0.001), (0.5 + HALF, 0.5, 0.001))
    pl.add_mesh(trace, color="black", line_width=2.5, lighting=False)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.26
    pl.camera.focal_point = (0.5, 0.5, 0.0)
    out = os.path.join(D, f"_ladder_stress_{tag}.png")
    pl.screenshot(out)
    pl.close()
    return out


def solve_case(label, register, tag):
    child = common.split_with_fault(common.base_mesh(0.04), segment)
    stokes = common.stokes_on(child, common.simple_shear(child))
    register(stokes)
    fault_contact.solve_with_fault(stokes, picard=2)
    s, V = common.slip_vs_position(stokes, tangent=(1.0, 0.0))
    # the unsplit tips carry exactly zero slip by construction
    s = np.concatenate([[-HALF], s, [HALF]])
    V = np.concatenate([[0.0], V, [0.0]])
    panel = render_shear_stress(stokes, child, tag)
    runs.append((label, s, np.abs(V), panel))
    print(f"{label:34s} peak slip {np.abs(V).max():.4f}")


solve_case("frictionless",
           lambda st: st.add_fault_bc(0, boundary="Fault"), "free")
solve_case(r"viscous, $\eta_f = \eta/a$",
           lambda st: st.add_fault_bc(common.ETA / HALF, boundary="Fault"),
           "visc")
solve_case(r"Coulomb, $\mu\sigma_n = 0.6 < \tau_\infty$",
           lambda st: fault_contact.add_coulomb_fault_bc(
               st, 0.3, "Fault", sigma_n=SIGMA_N, V0=1e-4), "weak")
solve_case(r"rate-state (steady, $f_{ss}\sigma_n \approx 0.86$)",
           lambda st: fault_contact.add_rate_state_fault_bc(
               st, 0.42, "Fault", a=0.02, b=0.01, V0=1e-3, Dc=1e-2,
               sigma_n=SIGMA_N), "rs")
solve_case(r"Coulomb, $\mu\sigma_n = 1.2 > \tau_\infty$ (stuck)",
           lambda st: fault_contact.add_coulomb_fault_bc(
               st, 0.6, "Fault", sigma_n=SIGMA_N, V0=1e-4), "stuck")

colors = ["#c62828", "#e57373", "#d9960a", "#4a7bf7", "#555555"]

fig = plt.figure(figsize=(9.8, 6.2))
gs = gridspec.GridSpec(len(runs), 2, width_ratios=[1.7, 1.0],
                       wspace=0.12, hspace=0.15)

ax = fig.add_subplot(gs[:, 0])
for (label, s, V, _panel), col in zip(runs, colors):
    ax.plot(s, V, ".-", ms=3.5, lw=1.0, color=col, label=label)
ss = np.linspace(-HALF, HALF, 200)
ax.plot(ss, np.sqrt(np.maximum(HALF**2 - ss**2, 0)) / HALF
        * runs[0][2].max(), "k--", lw=0.8,
        label="elliptical profile (shape)")
ax.set_xlabel("position along the fault $s$")
ax.set_ylabel("slip rate $|V(s)|$")
ax.set_title("The fault-strength ladder: one fault, four laws "
             r"($\tau_\infty = 1$)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(-HALF * 1.15, HALF * 1.15)

for k, ((label, _s, _V, panel), col) in enumerate(zip(runs, colors)):
    axi = fig.add_subplot(gs[k, 1])
    axi.imshow(plt.imread(panel))
    axi.set_xticks([])
    axi.set_yticks([])
    for spine in axi.spines.values():
        spine.set_color(col)
        spine.set_linewidth(2.2)
    if k == 0:
        axi.set_title(r"shear stress $\sigma_{xy}$ (0 → 2, "
                      r"white $= \tau_\infty$)", fontsize=8)

fig.tight_layout()
out = os.path.join(D, "ladder.png")
fig.savefig(out, dpi=200)
print("wrote", out)
