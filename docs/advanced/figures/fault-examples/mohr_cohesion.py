"""Cohesive Mohr-Coulomb: strength survives into the tensile sector.

The cohesive sequel to mohr_friction: the fault's yield envelope is
tau = +-(C + mu sigma) (compression positive), declining through mild
tension and reaching ZERO at sigma = -C/mu. Three physical regimes and
one UNPHYSICAL one appear as the fault rotates:

- stuck (on the circle) — now around BOTH principal poles, because
  cohesion holds shear through mild tension;
- sliding, pinned to the declining envelope;
- HELD SHUT — beyond sigma = -C/mu the strength is zero and the normal
  stress is tensile: a real fault would OPEN, no static solution
  exists, and the bilateral no-opening constraint manufactures one by
  gluing the surfaces together (a tensile constraint reaction). The
  solver converges; the physics has failed. Detect it from the SIGN of
  the recovered normal traction.

The law is not a canned option — it is registered as a sympy
expression in the canonical symbols, which is the whole design: a new
fault rheology is four lines.

All Mohr figures on this page use the GEOLOGICAL sign convention:
compression positive, tension on the negative axis.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy
from PIL import Image

import underworld3 as uw
from underworld3.utilities import fault_contact

import common

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
A_RATE, GAMMA = 0.5, 1.0
MU, C, V0 = 0.6, 0.6, 1e-4
R_ANALYTIC = common.ETA * np.sqrt(4 * A_RATE**2 + GAMMA**2)
STEP = 7.5
angles = np.arange(0.0, 180.0 + 1e-9, STEP)


def register_cohesive_law(stokes):
    """Mohr-Coulomb with cohesion as a symbolic law. normal_stress is
    the SIGNED effective normal stress (reaction-fed; negative in
    tension), and the strength clamp is the law's own: Max(0, C + mu s)
    declines through mild tension and vanishes at s = -C/mu."""
    V = fault_contact.slip_rate
    S = fault_contact.normal_stress
    law = fault_contact.SymbolicFaultLaw(
        sympy.Max(C + MU * S, 0) * (2 / sympy.pi) * sympy.atan(V / V0))
    fault_contact.add_frictionless_fault_bc(stokes, "Fault")
    fault_contact._register_law(stokes, "Fault", law)


cache = os.path.join(D, "_mohr_cohesion_probes.npz")
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
        register_cohesive_law(stokes)
        fault_contact.solve_with_fault(stokes, picard=3)
        s, V, _leak = common.slip_profile(stokes)
        s_n, sig = common.normal_traction(stokes)
        v_med = float(np.median(V[common.inner(s)]))
        sigma_n = float(np.median(sig[common.inner(s_n)]))
        strength = max(C + MU * (-sigma_n), 0.0)
        tau = strength * (2 / np.pi) * np.arctan(v_med / V0)
        rows.append((theta, sigma_n, tau, v_med))
        print(f"theta {theta:6.1f}: sigma_n {sigma_n:8.4f}  "
              f"tau {tau:8.4f}  V {v_med:9.5f}")
    probes = np.array(rows)
    np.savez(cache, probes=probes)

# GEO convention for plotting: compression positive
sc = -probes[:, 1]
tau = probes[:, 2]
# beyond sigma = -C/mu the strength is zero and the fault is in
# tension: it would open — the constraint holds it shut (unphysical)
held_shut = sc < -C / MU + 1e-6
sliding = (np.abs(probes[:, 3]) > 5 * V0) & ~held_shut


def draw_stress_plane(ax):
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_ANALYTIC * np.cos(tt), R_ANALYTIC * np.sin(tt),
            "-", color="0.8", lw=0.9,
            label="ambient stress (welded circle)")
    s_env = np.linspace(-C / MU, 1.5 * R_ANALYTIC, 60)
    for sgn in (+1, -1):
        ax.plot(s_env, sgn * (C + MU * s_env), "--", color="0.35",
                lw=1.0, label=(r"envelope $\tau = \pm(C + \mu\sigma)$,"
                               r" zero at $\sigma = -C/\mu$"
                               if sgn > 0 else None))
    ax.axvspan(-1.6 * R_ANALYTIC, -C / MU, color="0.92", zorder=0)
    ax.text(-1.5 * R_ANALYTIC, 1.1 * R_ANALYTIC,
            "fault would open:\nno static solution\n(held shut by the\n"
            "no-opening constraint)", fontsize=7.5, va="top",
            color="0.35")
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)
    ax.set_xlabel(r"normal stress $\sigma$ (compression positive)")
    ax.set_ylabel(r"shear traction $\tau$")
    ax.set_xlim(-1.6 * R_ANALYTIC, 1.6 * R_ANALYTIC)
    ax.set_ylim(-1.35 * R_ANALYTIC, 1.35 * R_ANALYTIC)
    ax.set_aspect("equal")


fig, ax = plt.subplots(figsize=(7.6, 5.4))
draw_stress_plane(ax)
stuck = ~sliding & ~held_shut
ax.plot(sc[stuck], tau[stuck], "o", ms=7, color="#c62828",
        label="stuck: on the circle", zorder=5)
ax.plot(sc[sliding], tau[sliding], "s", ms=6, color="#d9960a",
        label="sliding: on the envelope", zorder=5)
ax.plot(sc[held_shut], tau[held_shut], "x", ms=8, mew=2.0,
        color="0.45", label="held shut (unphysical)", zorder=5)
ax.legend(fontsize=8, loc="lower right")
ax.set_title(rf"Cohesive Mohr-Coulomb fault, $C = {C}$, $\mu = {MU}$")
fig.tight_layout()
out = os.path.join(D, "mohr-cohesion.png")
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
    T = sig_k * n + tau_k * t
    scale = 0.16 / R_ANALYTIC
    axl.annotate("", xytext=c, xy=c + scale * T,
                 arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                 color="#4a7bf7"))
    if slide_k:
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
    axl.set_xlim(-0.06, 1.06)
    axl.set_ylim(-0.06, 1.06)
    axl.set_aspect("equal")
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_title(rf"cohesive fault, $C = {C}$, $\mu = {MU}$ ...",
                  fontsize=10)

    draw_stress_plane(axr)
    axr.legend(fontsize=7, loc="lower right")
    stuck_prev = (~sliding & ~held_shut)[:k + 1]
    slide_prev = sliding[:k + 1]
    shut_prev = held_shut[:k + 1]
    axr.plot(sc[:k + 1][stuck_prev], tau[:k + 1][stuck_prev], "o", ms=5,
             mfc="none", mec="#c62828", mew=1.2)
    axr.plot(sc[:k + 1][slide_prev], tau[:k + 1][slide_prev], "s", ms=5,
             mfc="none", mec="#d9960a", mew=1.2)
    axr.plot(sc[:k + 1][shut_prev], tau[:k + 1][shut_prev], "x", ms=6,
             mew=1.6, color="0.45")
    mark, mcol = (("x", "0.45") if shut_k else
                  ("s", "#d9960a") if slide_k else ("o", "#c62828"))
    axr.plot([sc[k]], [tau[k]], mark, ms=9, mew=2.2, color=mcol,
             zorder=6)
    axr.set_title("... strength declines to zero, then the fault "
                  "would open", fontsize=10)

    fig.suptitle("Cohesion keeps more of the Mohr circle", fontsize=11)
    fig.tight_layout()
    frame = os.path.join(D, f"_mohrc_frame_{k:03d}.png")
    fig.savefig(frame, dpi=110)
    plt.close(fig)
    frames.append(frame)

images = [Image.open(f) for f in frames]
out = os.path.join(D, "mohr-cohesion-build.gif")
images[0].save(out, save_all=True,
               append_images=images[1:] + [images[-1]] * 6,
               duration=280, loop=0)
print("wrote", out, f"({len(frames)} frames)")
