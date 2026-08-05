"""The Mohr circle, built frame by frame as the fault rotates.

The teaching version of mohr_circle.py: the left panel shows the fault
physically rotating in the box; the right panel shows its stress probe
sweeping around the Mohr circle at TWICE the rate — the double-angle
rule as motion, not as a formula. Each frame is a full welded-fault
solve (the probes are measured, not drawn), assembled into a GIF.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import common

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
A_RATE, GAMMA = 0.5, 1.0
R_ANALYTIC = common.ETA * np.sqrt(4 * A_RATE**2 + GAMMA**2)
STEP = 7.5
angles = np.arange(0.0, 180.0 + 1e-9, STEP)

# The sweep is 25 welded-fault solves; cache it so iterating on the
# animation's look does not re-measure the physics.
cache = os.path.join(D, "_mohr_probes.npz")
if os.path.exists(cache):
    probes = np.load(cache)["probes"]
    assert len(probes) == len(angles)
    print(f"loaded {len(probes)} cached probes")
else:
    probes = []
    for theta in angles:
        sigma_n, tau = common.mohr_probe(theta, A_RATE, GAMMA,
                                         half_length=HALF)
        probes.append((theta, sigma_n, tau))
        print(f"theta {theta:6.1f}: sigma_n {sigma_n:8.4f}  "
              f"tau {tau:8.4f}")
    probes = np.array(probes)
    np.savez(cache, probes=probes)
centre = float(np.mean(probes[:, 1]))
# GEOLOGICAL convention on the stress plane: compression positive.
# The left panel's traction vector stays in physical (tension-positive)
# components; only the Mohr diagram flips.
cg = -centre
scg = -probes[:, 1]

frames = []
for k in range(len(probes)):
    theta, sig_k, tau_k = probes[k]
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.6, 4.6),
        gridspec_kw=dict(width_ratios=[1.0, 1.25]))

    # ---- left: the fault in the box -------------------------------------
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
    # the traction vector on the fault plane, from the MEASURED probe:
    # purely normal (aligned with n-hat) exactly at the principal
    # orientations — the same instant the probe crosses tau = 0
    T = sig_k * n + tau_k * t
    scale = 0.16 / R_ANALYTIC
    axl.annotate("", xytext=c, xy=c + scale * T,
                 arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                 color="#4a7bf7"))
    axl.text(0.06, 0.9, rf"$\theta = {theta:.1f}°$", fontsize=12)
    axl.text(0.06, 0.135, r"$\sigma\cdot\hat n$: traction on the plane",
             fontsize=9, color="#4a7bf7", transform=axl.transAxes)
    if abs(tau_k) < 0.05 * R_ANALYTIC:
        axl.text(0.5, 0.06, "principal orientation: traction ∥ normal",
                 fontsize=10, color="#c62828", ha="center",
                 transform=axl.transAxes)
    axl.set_xlim(-0.06, 1.06)
    axl.set_ylim(-0.06, 1.06)
    axl.set_aspect("equal")
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_title("the (welded) fault rotates ...", fontsize=10)

    # ---- right: the probe sweeps the circle, twice as fast --------------
    tt = np.linspace(0, 2 * np.pi, 300)
    axr.plot(cg + R_ANALYTIC * np.cos(tt), R_ANALYTIC * np.sin(tt),
             "-", color="0.75", lw=0.9)
    axr.axhline(0, color="0.8", lw=0.6)
    axr.axvline(cg, color="0.8", lw=0.6)
    axr.plot([cg - R_ANALYTIC, cg + R_ANALYTIC], [0, 0], "D",
             ms=5, color="0.3", zorder=4)
    axr.text(cg + R_ANALYTIC, -0.16 * R_ANALYTIC,
             "principal\nstresses", fontsize=7, ha="center", va="top",
             color="0.3")
    axr.plot(scg[:k + 1], probes[:k + 1, 2], "o", ms=5,
             mfc="none", mec="#c62828", mew=1.2)
    axr.plot([cg, -sig_k], [0.0, tau_k], "-",
             color="#4a7bf7", lw=1.2)
    axr.plot([-sig_k], [tau_k], "o", ms=9, color="#c62828", zorder=5)
    axr.text(0.04, 0.94, r"... its stress probe sweeps at $2\theta$",
             fontsize=10, transform=axr.transAxes)
    axr.set_xlabel(r"normal stress $\sigma$ (compression positive)")
    axr.set_ylabel(r"shear traction $\tau$")
    axr.set_xlim(cg - 1.35 * R_ANALYTIC, cg + 1.35 * R_ANALYTIC)
    axr.set_ylim(-1.35 * R_ANALYTIC, 1.35 * R_ANALYTIC)
    axr.set_aspect("equal")

    fig.suptitle("Building the Mohr circle with a rotating fault",
                 fontsize=11)
    fig.tight_layout()
    frame = os.path.join(D, f"_mohr_frame_{k:03d}.png")
    fig.savefig(frame, dpi=110)
    plt.close(fig)
    frames.append(frame)

images = [Image.open(f) for f in frames]
out = os.path.join(D, "mohr-circle-build.gif")
images[0].save(out, save_all=True, append_images=images[1:] +
               [images[-1]] * 6,          # hold the completed circle
               duration=280, loop=0)
print("wrote", out, f"({len(frames)} frames)")
