"""The Mohr circle meets the friction envelope.

The frictional sequel to mohr_circle/mohr_animate: the same rotating
fault, but now carrying Coulomb friction with reaction-fed normal
stress. Three regimes appear as the fault rotates:

- STUCK — the ambient resolved stress lies inside the envelope
  |tau| < mu |sigma_n| (compressive side): the fault transmits the
  full stress and its probe sits ON the Mohr circle;
- SLIDING — the ambient stress would exceed the envelope: the fault
  slips, drops the shear traction to its strength, and the probe is
  pinned to the yield line tau = ±mu |sigma_n|;
- HELD SHUT — under tensile normal stress bare friction has no
  strength: a real fault would OPEN, no static solution exists, and
  the bilateral no-opening constraint manufactures one by gluing the
  surfaces (tensile reaction). The solver converges; the physics has
  failed. The probes ride the axis at tau = 0, marked as unphysical.

sigma_n comes from the no-opening constraint's reaction; tau is read
from the Coulomb law at the measured slip rate — exact in both
regimes, because the regularised law IS the traction the fault
carries. Outputs: a static summary figure and the animated build.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import underworld3 as uw
from underworld3.utilities import fault_contact

import common

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
A_RATE, GAMMA = 0.5, 1.0
MU, V0 = 0.7, 1e-4
R_ANALYTIC = common.ETA * np.sqrt(4 * A_RATE**2 + GAMMA**2)
STEP = 7.5
angles = np.arange(0.0, 180.0 + 1e-9, STEP)

cache = os.path.join(D, "_mohr_friction_probes.npz")
if os.path.exists(cache):
    probes = np.load(cache)["probes"]
    assert len(probes) == len(angles)
    print(f"loaded {len(probes)} cached probes")
else:
    rows = []
    for theta in angles:
        child = common.split_with_fault(
            common.base_mesh(0.04), common.fault_segment(theta, HALF))
        stokes = common.stokes_on(
            child, common.shear_plus_stretch(child, A_RATE, GAMMA))
        fault_contact.add_coulomb_fault_bc(stokes, MU, "Fault",
                                           sigma_n="reaction", V0=V0)
        fault_contact.solve_with_fault(stokes, picard=3)
        s, V, _leak = common.slip_profile(stokes)
        s_n, sig = common.normal_traction(stokes)
        v_med = float(np.median(V[common.inner(s)]))
        sigma_n = float(np.median(sig[common.inner(s_n)]))
        sigma_eff = max(-sigma_n, 0.0)
        tau = MU * sigma_eff * (2 / np.pi) * np.arctan(v_med / V0)
        rows.append((theta, sigma_n, tau, v_med))
        print(f"theta {theta:6.1f}: sigma_n {sigma_n:8.4f}  "
              f"tau {tau:8.4f}  V {v_med:9.5f}")
    probes = np.array(rows)
    np.savez(cache, probes=probes)

centre = 0.0                        # the welded sweep's measured gauge
# GEOLOGICAL convention on the stress plane: compression positive.
scg = -probes[:, 1]
# bare friction: strength vanishes the moment the stress turns tensile
held_shut = scg < -1e-6
sliding = (np.abs(probes[:, 3]) > 5 * V0) & ~held_shut


def draw_stress_plane(ax):
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(centre + R_ANALYTIC * np.cos(tt), R_ANALYTIC * np.sin(tt),
            "-", color="0.8", lw=0.9,
            label="ambient stress (welded circle)")
    ss = np.linspace(0, 1.35 * R_ANALYTIC, 50)
    for sgn in (+1, -1):
        ax.plot(ss, sgn * MU * ss, "--", color="0.35", lw=1.0,
                label=(r"friction envelope $\tau = \pm\mu\sigma$"
                       if sgn > 0 else None))
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(centre, color="0.85", lw=0.6)
    ax.axvspan(-1.6 * R_ANALYTIC, 0, color="0.92", zorder=0)
    ax.text(-1.45 * R_ANALYTIC, 1.1 * R_ANALYTIC,
            "fault would open:\nno static solution\n(held shut by the\n"
            "no-opening constraint)", fontsize=7.5, va="top",
            color="0.35")
    ax.set_xlabel(r"normal stress $\sigma$ (compression positive)")
    ax.set_ylabel(r"shear traction $\tau$")
    ax.set_xlim(centre - 1.5 * R_ANALYTIC, centre + 1.5 * R_ANALYTIC)
    ax.set_ylim(-1.35 * R_ANALYTIC, 1.35 * R_ANALYTIC)
    ax.set_aspect("equal")


# ---- the static summary -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.4))
draw_stress_plane(ax)
stuck = ~sliding & ~held_shut
ax.plot(scg[stuck], probes[stuck, 2], "o", ms=7,
        color="#c62828", label="stuck: on the circle", zorder=5)
ax.plot(scg[sliding], probes[sliding, 2], "s", ms=6,
        color="#d9960a", label="sliding: on the envelope", zorder=5)
ax.plot(scg[held_shut], probes[held_shut, 2], "x", ms=8, mew=2.0,
        color="0.45", label="held shut (unphysical)", zorder=5)
ax.legend(fontsize=8, loc="upper left")
ax.set_title(rf"Coulomb fault probes, $\mu = {MU}$: "
             "the stress switches to the yield envelope")
fig.tight_layout()
out = os.path.join(D, "mohr-friction.png")
fig.savefig(out, dpi=200)
print("wrote", out)

# ---- the animated build ------------------------------------------------------
frames = []
for k in range(len(probes)):
    theta, sig_k, tau_k, v_k = probes[k]
    shut_k = bool(held_shut[k])
    slide_k = (abs(v_k) > 5 * V0) and not shut_k
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.6, 4.6),
        gridspec_kw=dict(width_ratios=[1.0, 1.25]))

    axl.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                                edgecolor="0.4"))
    t = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    n = np.array([-t[1], t[0]])
    c = common.CENTRE
    axl.plot([c[0] - HALF * t[0], c[0] + HALF * t[0]],
             [c[1] - HALF * t[1], c[1] + HALF * t[1]],
             "-", color="#c62828", lw=2.5)
    axl.annotate("", xytext=c, xy=c + 0.14 * n,
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="0.35"))
    axl.text(*(c + 0.18 * n), r"$\hat n$", fontsize=9, ha="center",
             color="0.35")
    T = sig_k * n + tau_k * t
    scale = 0.16 / R_ANALYTIC
    axl.annotate("", xytext=c, xy=c + scale * T,
                 arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                 color="#4a7bf7"))
    if slide_k:
        # half-arrows for the slip sense
        off = 0.035 * n
        sgn = np.sign(v_k)
        for pm in (+1, -1):
            axl.annotate("", xytext=c + pm * off - pm * sgn * 0.09 * t,
                         xy=c + pm * off + pm * sgn * 0.09 * t,
                         arrowprops=dict(arrowstyle="->", lw=1.4,
                                         color="#d9960a"))
    axl.text(0.06, 0.9, rf"$\theta = {theta:.1f}°$", fontsize=12)
    status, scol = (("HELD SHUT (unphysical)", "0.45") if shut_k
                    else ("SLIDING", "#d9960a") if slide_k
                    else ("stuck", "#c62828"))
    axl.text(0.06, 0.82, status, fontsize=10, color=scol)
    axl.text(0.06, 0.135, r"$\sigma\cdot\hat n$: traction on the plane",
             fontsize=9, color="#4a7bf7", transform=axl.transAxes)
    axl.set_xlim(-0.06, 1.06)
    axl.set_ylim(-0.06, 1.06)
    axl.set_aspect("equal")
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_title(rf"Coulomb fault, $\mu = {MU}$, rotating ...",
                  fontsize=10)

    draw_stress_plane(axr)
    axr.legend(fontsize=7, loc="upper left")
    stuck_prev = (~sliding & ~held_shut)[:k + 1]
    slide_prev = sliding[:k + 1]
    shut_prev = held_shut[:k + 1]
    axr.plot(scg[:k + 1][stuck_prev], probes[:k + 1][stuck_prev, 2],
             "o", ms=5, mfc="none", mec="#c62828", mew=1.2)
    axr.plot(scg[:k + 1][slide_prev], probes[:k + 1][slide_prev, 2],
             "s", ms=5, mfc="none", mec="#d9960a", mew=1.2)
    axr.plot(scg[:k + 1][shut_prev], probes[:k + 1][shut_prev, 2],
             "x", ms=6, mew=1.6, color="0.45")
    mark, mcol = (("x", "0.45") if shut_k else
                  ("s", "#d9960a") if slide_k else ("o", "#c62828"))
    axr.plot([-sig_k], [tau_k], mark, ms=9, mew=2.2, color=mcol,
             zorder=6)
    axr.set_title("... its traction cannot leave the envelope",
                  fontsize=10)

    fig.suptitle("A frictional fault against the Mohr circle",
                 fontsize=11)
    fig.tight_layout()
    frame = os.path.join(D, f"_mohrf_frame_{k:03d}.png")
    fig.savefig(frame, dpi=110)
    plt.close(fig)
    frames.append(frame)

images = [Image.open(f) for f in frames]
out = os.path.join(D, "mohr-friction-build.gif")
images[0].save(out, save_all=True,
               append_images=images[1:] + [images[-1]] * 6,
               duration=280, loop=0)
print("wrote", out, f"({len(frames)} frames)")
