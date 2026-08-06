"""What the Coulomb Failure Function is, and what a change in it means.

A definition figure for the teaching decks, to come BEFORE any slide that
asks a student to read a Delta CFF number.

Left: the anatomy. With compression positive and the envelope at
tau = C + mu' sigma, the Coulomb failure function

    CFF = tau - (C + mu' sigma)

is just the SIGNED VERTICAL GAP between the stress point and the failure
line: negative below it (safe), zero on it (failing). A change in it,

    dCFF = d(tau) - mu' d(sigma)        [compression positive]
         = d(tau) + mu' d(sigma_n)      [tension positive, as measured]

is therefore how much the point moved TOWARDS the envelope -- part from
the shear it gained, part from being unclamped.

Right: the same thing measured, on a case the decks use two slides
later. The San Jacinto gauge of california_clocks.py, at the real San
Jacinto's own mapped strike (135 deg), before and after the San Andreas
slips. Nothing here is drawn by hand: both points are solves, and the
cache is the one the clocks animation already built.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))

MU_P, COH, P0 = 0.4, 0.75, 1.0        # as everywhere else in the set
SITE, STRIKE = "SJF", 135.0
C_BEFORE, C_AFTER, C_ENV = "0.45", "#c62828", "0.3"

cache = os.path.join(D, "_california_clocks.npz")
data = dict(np.load(cache))
angles = data["angles"]
k = int(np.argmin(np.abs(angles - STRIKE)))
assert abs(angles[k] - STRIKE) < 1e-6, "strike is not a sampled orientation"

# measured, tension-positive; plotted compression-positive about P0
sig0, tau0 = data[f"{SITE}_amb_sig"][k], data[f"{SITE}_amb_tau"][k]
sig1, tau1 = data[f"{SITE}_slip_sig"][k], data[f"{SITE}_slip_tau"][k]
x0, x1 = P0 - sig0, P0 - sig1
# resolve in the direction the plane is ALREADY being sheared
d = np.sign(tau0)
y0, y1 = d * tau0, d * tau1
dcff = (y1 - y0) - MU_P * (x1 - x0)
print(f"{SITE} at {STRIKE:.0f}°:  before ({x0:+.3f}, {y0:+.3f})   "
      f"after ({x1:+.3f}, {y1:+.3f})")
print(f"  d(tau) {y1 - y0:+.3f}   -mu' d(sigma) {-MU_P * (x1 - x0):+.3f}"
      f"   ->  dCFF {dcff:+.3f}")


def envelope(ax, xs):
    ax.plot(xs, COH + MU_P * xs, "-", color=C_ENV, lw=1.6,
            label=r"failure envelope  $\tau = C + \mu'\sigma$")
    ax.fill_between(xs, COH + MU_P * xs, 3.0, color="#c62828", alpha=0.07,
                    lw=0)


def gap_marker(ax, x, y, col, label=None, side=1):
    """The vertical gap from the point up to the envelope -- which IS
    minus the Coulomb failure function."""
    ytop = COH + MU_P * x
    ax.plot([x, x], [y, ytop], ":", color=col, lw=1.4)
    ax.annotate("", xy=(x, ytop), xytext=(x, y),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color=col))
    if label:
        ax.text(x + side * 0.05, 0.5 * (y + ytop), label, fontsize=9,
                color=col, ha="left" if side > 0 else "right", va="center")


fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.4, 5.2))

# ---- left: the anatomy -----------------------------------------------------
xs = np.linspace(0.0, 2.6, 60)
envelope(axa, xs)
axa.axhline(0, color="0.9", lw=0.6)

# Left panel makes ONE point: CFF is a gap, and dCFF is the change in
# that gap. The decomposition is left to the right-hand panel, so the
# gap arrows never share a vertical line with a component arrow.
ax0, ay0 = 1.95, 0.50
ax1, ay1 = 1.40, 0.92
axa.plot([ax0], [ay0], "o", ms=11, color=C_BEFORE, zorder=5)
axa.plot([ax1], [ay1], "o", ms=11, color=C_AFTER, zorder=5)
axa.text(ax0 + 0.07, ay0, "before", fontsize=10, color=C_BEFORE,
         va="center")
axa.text(ax1 - 0.07, ay1, "after", fontsize=10, color=C_AFTER,
         ha="right", va="center")
gap_marker(axa, ax0, ay0, C_BEFORE, r"$-\,$CFF$_{\rm before}$", side=1)
gap_marker(axa, ax1, ay1, C_AFTER, r"$-\,$CFF$_{\rm after}$", side=-1)
axa.annotate("", xy=(ax1, ay1), xytext=(ax0, ay0),
             arrowprops=dict(arrowstyle="-|>", lw=1.6, color="0.35",
                             connectionstyle="arc3,rad=-0.25"))

axa.text(0.03, 0.965,
         r"$\mathrm{CFF} = \tau - (C + \mu'\sigma)$" "\n"
         r"$\Delta\mathrm{CFF} = \Delta\tau - \mu'\Delta\sigma$",
         transform=axa.transAxes, fontsize=12, va="top",
         bbox=dict(fc="white", ec="0.8", pad=4.0))
axa.text(0.03, 0.035,
         "CFF is the signed gap to the envelope: negative below it,\n"
         r"zero on it. $\Delta$CFF is how much that gap CLOSED.",
         transform=axa.transAxes, fontsize=9.5, va="bottom", color="0.25")
axa.set_xlim(0.0, 2.6)
axa.set_ylim(-0.25, 1.85)
axa.set_xlabel(r"normal stress $\sigma$ (compression positive)")
axa.set_ylabel(r"shear stress $\tau$ (in the slip direction)")
axa.set_title("The Coulomb failure function", fontsize=11)
axa.legend(fontsize=8.5, loc="upper right")

# ---- right: the measured example -------------------------------------------
envelope(axb, xs)
axb.axhline(0, color="0.9", lw=0.6)
axb.plot([x0], [y0], "o", ms=11, color=C_BEFORE, zorder=5,
         label="before the earthquake")
axb.plot([x1], [y1], "o", ms=11, color=C_AFTER, zorder=5,
         label="after the San Andreas slips")
# Right panel carries the DECOMPOSITION, on real numbers -- and no gap
# arrows here, so nothing shares a vertical with the dtau component.
axb.annotate("", xy=(x1, y0), xytext=(x0, y0),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1565c0"))
axb.annotate("", xy=(x1, y1), xytext=(x1, y0),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#2e7d32"))
axb.text(0.5 * (x0 + x1), y0 + 0.12,
         rf"$-\mu'\Delta\sigma = {-MU_P * (x1 - x0):+.2f}$" "\nunclamped:"
         " towards failure", fontsize=9, color="#1565c0", ha="center",
         va="bottom")
axb.text(x1 - 0.06, 0.5 * (y0 + y1),
         rf"$\Delta\tau = {y1 - y0:+.2f}$" "\nshear lost:\naway from"
         " failure", fontsize=9, color="#2e7d32", ha="right", va="center")
axb.text(0.03, 0.965,
         f"San Jacinto, at its own strike ({STRIKE:.0f}°)\n"
         rf"$\Delta$CFF $= {y1 - y0:+.2f} {-MU_P * (x1 - x0):+.2f}"
         rf" = \mathbf{{{dcff:+.2f}}}$",
         transform=axb.transAxes, fontsize=11.5, va="top",
         bbox=dict(fc="white", ec="0.8", pad=4.0))
axb.text(0.03, 0.035,
         "The two terms pull opposite ways, and the shear term wins:\n"
         "this fault was moved AWAY from failure. Both points are\n"
         "measured — two Underworld3 solves, read on the same plane.",
         transform=axb.transAxes, fontsize=9, va="bottom", color="0.25")
axb.set_xlim(0.0, 2.6)
axb.set_ylim(-0.25, 1.85)
axb.set_xlabel(r"normal stress $\sigma$ (compression positive)")
axb.set_title("A measured case: the San Andreas relaxes the San Jacinto",
              fontsize=11)
axb.legend(fontsize=8.5, loc="upper right")

fig.tight_layout()
out = os.path.join(D, "cff-explained.png")
fig.savefig(out, dpi=200)
print("wrote", out)
