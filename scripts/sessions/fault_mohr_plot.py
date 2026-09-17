"""Mohr diagram and animation from the checkpoints of ``fault_mohr_tumble.py``.

Reads every ``sample_*/probe.npz`` in the run directory — no solving — so
the figures can be restyled, or redrawn while a sweep is still running.

Conventions (geological): compression positive, principal stresses ordered
:math:`\\sigma_1 \\ge \\sigma_2 \\ge \\sigma_3`. The shear traction is plotted
SIGNED so the whole circle appears: its magnitude is the measured
:math:`|\\tau|` and its sign is that of :math:`n_1 n_3`, the fault normal's
components along the :math:`\\sigma_1` and :math:`\\sigma_3` axes. In 2-D
this is the double-angle rule, :math:`\\tau = \\tfrac12(\\sigma_1-\\sigma_3)
\\sin 2\\theta`; in 3-D every orientation then falls inside the largest
circle and outside the two smaller ones.

::

    python scripts/sessions/fault_mohr_plot.py -uw_dim 3 -uw_animate 1
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import underworld3 as uw

params = uw.Params(
    uw_dim=2,
    uw_output_dir=os.path.expanduser("~/+Simulations/fault_mohr_triaxial"),
    uw_animate=0,
    uw_frame_stride=1,          # every n-th sample becomes a frame
)

DIM = params.uw_dim
run_dir = os.path.join(params.uw_output_dir, f"tumble_{DIM}d")
records = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(run_dir, "sample_*", "probe.npz")))]
if not records:
    raise SystemExit(f"no probe records under {run_dir}")

ACCENT = "#b03a2e"
INK = "#1b242c"
FAINT = "#9aa7b1"


def principal_frame(stress):
    """Compression-positive principal stresses (descending) and axes."""
    values, vectors = np.linalg.eigh(-stress)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


sigmas, frame = principal_frame(records[0]["stress"])
normals = np.array([r["normal"] for r in records])
local = normals @ frame                       # components along sigma_1.. axes
sigma_n = -np.array([float(r["sigma_n"]) for r in records])
tau_mag = np.array([np.linalg.norm(r["tau"]) for r in records])
tau_signed = np.sign(local[:, 0] * local[:, -1]) * tau_mag
sigma_err = np.abs(-sigma_n - np.array([float(r["sigma_exact"]) for r in records]))
tau_err = np.array([np.linalg.norm(r["tau"] - r["tau_exact"]) for r in records])
radius_max = 0.5 * (sigmas[0] - sigmas[-1])
summary = (f"{len(records)} faults   max error: "
           f"$\\sigma_n$ {sigma_err.max() / radius_max:.1%}, "
           f"$\\tau$ {tau_err.max() / radius_max:.1%} of $R$")
print(summary.replace("$", "").replace("\\sigma_n", "sigma_n").replace("\\tau", "tau"))


def circle(ax, a, b, **kw):
    t = np.linspace(0.0, 2.0 * np.pi, 361)
    c, r = 0.5 * (a + b), 0.5 * abs(a - b)
    ax.plot(c + r * np.cos(t), r * np.sin(t), **kw)


def mohr_axes(ax):
    if DIM == 3:
        # admissible region: inside the sigma_1-sigma_3 circle, outside the
        # other two
        g = np.linspace(sigmas[-1], sigmas[0], 500)
        big = 0.5 * (sigmas[0] - sigmas[-1])
        c_big = 0.5 * (sigmas[0] + sigmas[-1])
        outer = np.sqrt(np.maximum(big**2 - (g - c_big) ** 2, 0.0))
        r12, c12 = 0.5 * (sigmas[0] - sigmas[1]), 0.5 * (sigmas[0] + sigmas[1])
        r23, c23 = 0.5 * (sigmas[1] - sigmas[2]), 0.5 * (sigmas[1] + sigmas[2])
        inner = np.maximum(np.sqrt(np.maximum(r12**2 - (g - c12) ** 2, 0.0)),
                           np.sqrt(np.maximum(r23**2 - (g - c23) ** 2, 0.0)))
        for sign in (1, -1):
            ax.fill_between(g, sign * inner, sign * outer, color="#e9eef2", lw=0)
        circle(ax, sigmas[0], sigmas[1], color=INK, lw=0.8)
        circle(ax, sigmas[1], sigmas[2], color=INK, lw=0.8)
    circle(ax, sigmas[0], sigmas[-1], color=INK, lw=1.1)
    ax.axhline(0.0, color=FAINT, lw=0.6)
    for k, s in enumerate(sigmas):
        ax.plot(s, 0.0, "D", color=INK, ms=4, zorder=6)
        ax.annotate(f"$\\sigma_{k + 1}$", (s, 0.0), textcoords="offset points",
                    xytext=(12, -16), ha="center", fontsize=10, color=INK, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    pad = 0.2 * radius_max
    ax.set_xlim(sigmas[-1] - pad, sigmas[0] + pad)
    ax.set_ylim(-radius_max - pad, radius_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(r"normal stress $\sigma_n$ (compression positive)")
    ax.set_ylabel(r"shear traction $\tau$ (signed)")


def colour_values():
    """2-D: fault angle to the sigma_1 axis. 3-D: normal component along sigma_2."""
    if DIM == 2:
        return np.degrees(np.arctan2(local[:, 1], local[:, 0])) % 180.0, "twilight", (0, 180), r"normal angle to $\sigma_1$ (deg)"
    return np.abs(local[:, 1]), "viridis", (0, 1), r"$|n_2|$, normal along $\sigma_2$"


def stereonet(ax, upto=None):
    """Equal-area lower-hemisphere projection of the fault normals, box frame."""
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=0.8)
    ax.plot([-1, 1], [0, 0], color=FAINT, lw=0.4)
    ax.plot([0, 0], [-1, 1], color=FAINT, lw=0.4)

    def project(v):
        v = np.where(v[:, 2:3] > 0, -v, v)
        s = 1.0 / np.sqrt(1.0 - v[:, 2])
        return v[:, 0] * s, v[:, 1] * s

    values, cmap, lim, _label = colour_values()
    n = len(normals) if upto is None else upto + 1
    x, y = project(normals[:n])
    ax.scatter(x, y, c=values[:n], cmap=cmap, vmin=lim[0], vmax=lim[1], s=14,
               edgecolor="none")
    px, py = project(frame.T)
    for k in range(3):
        ax.plot(px[k], py[k], "s", mfc="white", mec=INK, ms=7)
        ax.annotate(f"$\\sigma_{k + 1}$", (px[k], py[k]), textcoords="offset points",
                    xytext=(6, 5), fontsize=9)
    if upto is not None:
        ax.plot(x[-1], y[-1], "o", mfc="none", mec=ACCENT, ms=11, mew=1.8)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("fault normals (lower hemisphere)", fontsize=9)


def box_2d(ax, upto=None):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=0.8))
    values, cmap, lim, _label = colour_values()
    cm = plt.get_cmap(cmap)
    n = len(records) if upto is None else upto + 1
    for k in range(n):
        r = records[k]
        t = np.array([r["normal"][1], -r["normal"][0]])
        ends = np.array([r["centre"] - r["radius"] * t, r["centre"] + r["radius"] * t])
        current = upto is not None and k == upto
        ax.plot(ends[:, 0], ends[:, 1], "-",
                color=ACCENT if current else cm((values[k] - lim[0]) / (lim[1] - lim[0])),
                lw=3.0 if current else 1.2, alpha=1.0 if (current or upto is None) else 0.45,
                zorder=5 if current else 2)
    c = records[0]["centre"]
    for sign in (-1, 1):
        axis = frame[:, 0]
        ax.annotate("", xy=c + sign * 0.3 * axis, xytext=c + sign * 0.47 * axis,
                    arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(r"welded faults in the box ($\sigma_1$ arrows)", fontsize=9)


def disc_3d(ax, k):
    r = records[k]
    for e in np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3):
        for d in range(3):
            if e[d] == 0:
                f = e.copy()
                f[d] = 1
                ax.plot(*zip(e, f), color=FAINT, lw=0.6)
    n = r["normal"]
    helper = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a = np.linspace(0, 2 * np.pi, 60)
    rim = r["centre"] + r["radius"] * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2))
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection([rim], facecolor=ACCENT, alpha=0.55,
                                         edgecolor=ACCENT))
    tip = r["centre"] + 0.3 * n
    ax.plot(*zip(r["centre"], tip), color=INK, lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-60)
    ax.set_title("the disc fault", fontsize=9)


def mohr_points(ax, upto=None):
    values, cmap, lim, label = colour_values()
    n = len(records) if upto is None else upto + 1
    sc = ax.scatter(sigma_n[:n], tau_signed[:n], c=values[:n], cmap=cmap,
                    vmin=lim[0], vmax=lim[1], s=22 if DIM == 2 else 12,
                    edgecolor=INK if DIM == 2 else "none", linewidth=0.3, zorder=5)
    if upto is not None:
        ax.plot(sigma_n[upto], tau_signed[upto], "o", mfc="none", mec=ACCENT,
                ms=12, mew=2.0, zorder=7)
    return sc, label


# ---- the still figure -----------------------------------------------------
fig, (ax_left, ax_mohr) = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained",
                                       gridspec_kw=dict(width_ratios=[1, 1.35]))
if DIM == 2:
    box_2d(ax_left)
else:
    stereonet(ax_left)
mohr_axes(ax_mohr)
sc, label = mohr_points(ax_mohr)
fig.colorbar(sc, ax=ax_mohr, shrink=0.75, label=label)
ax_mohr.set_title(summary, fontsize=9)
fig.suptitle(f"Mohr diagram from randomly tumbling welded faults ({DIM}-D)", fontsize=12)
still = os.path.join(params.uw_output_dir, f"mohr-tumble-{DIM}d.png")
fig.savefig(still, dpi=170)
plt.close(fig)
print("wrote", still)

# ---- the animation --------------------------------------------------------
if params.uw_animate:
    frame_dir = os.path.join(run_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    frames = []
    for k in range(0, len(records), params.uw_frame_stride):
        if DIM == 2:
            fig = plt.figure(figsize=(10, 5.0), layout="constrained")
            ax_a = fig.add_subplot(1, 2, 1)
            ax_m = fig.add_subplot(1, 2, 2)
            box_2d(ax_a, upto=k)
        else:
            fig = plt.figure(figsize=(13, 5.0), layout="constrained")
            ax_a = fig.add_subplot(1, 3, 1, projection="3d")
            ax_s = fig.add_subplot(1, 3, 2)
            ax_m = fig.add_subplot(1, 3, 3)
            disc_3d(ax_a, k)
            stereonet(ax_s, upto=k)
        mohr_axes(ax_m)
        mohr_points(ax_m, upto=k)
        ax_m.set_title(f"fault {k + 1} of {len(records)}", fontsize=10)
        path = os.path.join(frame_dir, f"frame_{k:04d}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        frames.append(path)
    images = [Image.open(f) for f in frames]
    gif = os.path.join(params.uw_output_dir, f"mohr-tumble-{DIM}d.gif")
    images[0].save(gif, save_all=True, append_images=images[1:] + [images[-1]] * 10,
                   duration=160, loop=0)
    print("wrote", gif, f"({len(frames)} frames)")
