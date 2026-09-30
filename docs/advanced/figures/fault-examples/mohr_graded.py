"""The graded fault: depth-dependent strength sampled along one fault.

Gravity joins the Mohr experiment. With constant density and closed
(velocity-Dirichlet) walls the flow is untouched — pressure absorbs the
body force exactly — but the WELDED fault's per-node probes now sample
the hydrostatic gradient: every node sits at its own depth, so each
fault orientation contributes a horizontal STREAK of points in the
(sigma, tau) plane rather than a single value. The streak is longest
for the vertical fault (largest depth range along the fault), a single
dot for the horizontal one, and the whole cloud scatters about the
family of Mohr circles between the shallowest and deepest fault points.

The per-node sampling is native: sigma_n comes from the no-opening
reaction de-smeared NODE BY NODE, and tau from the weld's own law
tau = eta_f V(s) at each node — nothing is averaged.

Gauge: a closed box fixes pressure only up to a constant (the solver
uses a mean-zero gauge). The plot re-anchors it so p = 0 at the top
surface — the shift is exactly rho g H / 2, known analytically, and is
applied to the plotted sigma only.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import underworld3 as uw
from underworld3.utilities import fault_contact

import common

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
A_RATE, GAMMA = 0.5, 1.0
RHO_G = 0.75                       # modest: streaks visible, circle intact
ETA_WELD = 200.0 * common.ETA / HALF
R_ANALYTIC = common.ETA * np.sqrt(4 * A_RATE**2 + GAMMA**2)

angles = np.arange(0.0, 180.0 - 1e-9, 22.5)

cache = os.path.join(D, "_mohr_graded_probes.npz")
if os.path.exists(cache):
    d = np.load(cache)
    theta_all, y_all, sig_all, tau_all = (d["theta"], d["y"], d["sig"],
                                          d["tau"])
    print(f"loaded {len(sig_all)} cached node probes")
else:
    theta_all, y_all, sig_all, tau_all = [], [], [], []
    for theta in angles:
        child = common.split_with_fault(
            common.base_mesh(0.04), common.fault_segment(theta, HALF))
        stokes = common.stokes_on(
            child, common.shear_plus_stretch(child, A_RATE, GAMMA))
        stokes.bodyforce = [0.0, -RHO_G]
        stokes.add_fault_bc(ETA_WELD, boundary="Fault")
        fault_contact.solve_with_fault(stokes, picard=2)

        # per-NODE probes, matched by the along-fault coordinate
        coords, jumps, normals = fault_contact.fault_pair_jumps(
            stokes, "Fault", stokes._rotated_freeslip_info)
        t_hat = np.array([np.cos(np.radians(theta)),
                          np.sin(np.radians(theta))])
        s_v = (coords - common.CENTRE) @ t_hat
        order_v = np.argsort(s_v)
        V = (jumps @ t_hat)[order_v]
        y = coords[order_v, 1]
        s_n, sig = common.normal_traction(stokes)
        assert len(sig) == len(V), "pair-node sets disagree"
        mid = common.inner(s_v[order_v])
        theta_all.extend([theta] * int(mid.sum()))
        y_all.extend(y[mid])
        sig_all.extend(sig[mid])
        tau_all.extend(ETA_WELD * V[mid])
        print(f"theta {theta:6.1f}: {int(mid.sum())} nodes, "
              f"sigma_n [{sig[mid].min():.3f}, {sig[mid].max():.3f}]")
    theta_all, y_all, sig_all, tau_all = (np.array(theta_all),
                                          np.array(y_all),
                                          np.array(sig_all),
                                          np.array(tau_all))
    np.savez(cache, theta=theta_all, y=y_all, sig=sig_all, tau=tau_all)

# GEO convention + the top-zero pressure anchor (mean-zero solver gauge
# shifted by the known rho g H / 2)
sc = -sig_all + RHO_G * 0.5
depth = 1.0 - y_all

fig, ax = plt.subplots(figsize=(7.8, 5.6))
# the family of circles between the shallowest and deepest fault points:
# fault nodes span y in [0.3, 0.7] -> pressure rho g (1 - y)
tt = np.linspace(0, 2 * np.pi, 300)
for pshift, style in ((RHO_G * 0.3, ":"), (RHO_G * 0.5, "-"),
                      (RHO_G * 0.7, ":")):
    ax.plot(pshift + R_ANALYTIC * np.cos(tt), R_ANALYTIC * np.sin(tt),
            style, color="0.75", lw=0.9,
            label=("Mohr circles: shallowest / centre / deepest"
                   if style == "-" else None))
ax.axhline(0, color="0.85", lw=0.6)
ax.axvline(0, color="0.85", lw=0.6)

pts = ax.scatter(sc, tau_all, c=depth, cmap="viridis", s=22, zorder=5,
                 edgecolors="none")
fig.colorbar(pts, ax=ax, label="depth below surface", shrink=0.8)

# annotate one streak: the vertical fault has the largest depth range
vert = theta_all == 90.0
if vert.any():
    ax.annotate("one fault (90°):\na streak, not a point",
                xy=(sc[vert].min(), tau_all[vert].mean()),
                xytext=(-1.30, -1.30), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
flat = theta_all == 0.0
if flat.any():
    ax.annotate("horizontal fault (0°):\none depth, one point",
                xy=(sc[flat].mean(), tau_all[flat].mean()),
                xytext=(-1.15, 1.22), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

ax.set_xlabel(r"normal stress $\sigma$ (compression positive, "
              r"$p = 0$ at the surface)")
ax.set_ylabel(r"shear traction $\tau$")
ax.set_title(rf"The graded fault: hydrostatic load $\rho g = {RHO_G}$, "
             "per-node probes")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="center")
fig.tight_layout()
out = os.path.join(D, "mohr-graded.png")
fig.savefig(out, dpi=200)
print("wrote", out)
