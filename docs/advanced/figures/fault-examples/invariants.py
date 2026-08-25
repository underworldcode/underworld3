"""Invariants: the numbers that survive a change of frame.

The pedagogical problem this figure exists to solve: told that a tensor
"has invariants", students reasonably ask what is being varied. Nothing
is. The state of stress at a point is one physical object; sigma_xx,
sigma_xy ... are merely what you read off when you interrogate it with
one particular set of planes. Choose different planes and every number
changes, while the thing described does not.

So this animation does not vary the stress. It varies the OBSERVER. A
square element turns through the same fixed field, and at every angle
the full 2x2 matrix in that element's own frame is displayed. All four
entries churn. Underneath them sit the mean stress and the radius,
computed from those same churning entries, and they do not move.

Two of those angles are the ones the course already knows: the frame
in which the off-diagonal entries vanish is the principal frame. The
Cartesian description and the principal description are the same state
written twice -- which is exactly why the numbers they share are the
ones worth naming.

The counting is the punchline. A symmetric 2x2 tensor holds 3 numbers;
choosing a frame costs 1 angle; 3 - 1 = 2 quantities cannot depend on
the choice. In 3D, 6 - 3 = 3 -- which is why there are exactly three
principal invariants and not four.

Every entry is measured. The rotating-frame matrix is assembled from
the welded-probe sweep cached by mohr_two_fields.py:

    sigma'_nn(theta) = probe normal traction at theta
    sigma'_nt(theta) = probe shear traction at theta
    sigma'_tt(theta) = probe normal traction at theta - 90

A confining pressure is ADDED analytically. The solve is driven purely
deviatorically and, in a closed velocity-driven box, its pressure is
fixed only to an arbitrary constant anyway (see the handoff note), so
imposing a stated confining pressure is more honest than quoting the
solver's own. It also makes the first invariant something other than
zero, which is the point.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

D = os.path.dirname(os.path.abspath(__file__))
CENTRE = np.array([0.5, 0.5])
STEP = 7.5
NGRID = 24                       # 180 / STEP: the sweep is 180-periodic
P_CONF = 2.5                     # stated confining pressure (compression +)

VARY_C = "#b3541e"               # the components: they change
INV_C = "#1b5e20"                # the invariants: they do not
NORM_C = "#8e24aa"
SHEAR_C = "#ef6c00"
FRAME_C = "#1565c0"              # the observer's axes
DRIVE_C = "#6d4c41"

A_RATE, GAMMA, ETA = 0.5, 1.0, 1.0
SIG_DRIVE = ETA * np.array([[2 * A_RATE, GAMMA], [GAMMA, -2 * A_RATE]])
R_AN = float(np.linalg.norm(SIG_DRIVE, 2))


def load_probes():
    cache = os.path.join(D, "_mohr2_probes_A.npz")
    assert os.path.exists(cache), (
        "missing _mohr2_probes_A.npz -- run mohr_two_fields.py first")
    p = np.load(cache)["probes"]
    assert abs(p[1, 0] - p[0, 0] - STEP) < 1e-9, "cache step is not 7.5 deg"
    return p


def sig_at(probes, theta):
    """Measured normal traction for the plane at `theta`, 180-periodic."""
    return probes[int(round((theta % 180.0) / STEP)) % NGRID, 1]


def frame_matrix(probes, k):
    """The stress tensor in the frame of probe k, compression positive.

    sigma_geo = P.I - sigma_solver, so the whole tensor flips sign and
    the confining pressure is added to the diagonal.
    """
    theta, sig_k, tau_k = probes[k]
    snn = P_CONF - sig_k
    stt = P_CONF - sig_at(probes, theta - 90.0)
    snt = -tau_k
    return theta, np.array([[snn, snt], [snt, stt]])


def invariants(M):
    sm = 0.5 * (M[0, 0] + M[1, 1])
    rad = np.hypot(0.5 * (M[0, 0] - M[1, 1]), M[0, 1])
    return sm, rad


def principal_dirs(probes):
    """Measured principal directions: where the probe's shear vanishes."""
    th, tau = probes[:, 0], probes[:, 2]
    out = []
    for i in range(len(tau) - 1):
        if tau[i] == 0.0 or tau[i] * tau[i + 1] < 0.0:
            w = abs(tau[i]) / (abs(tau[i]) + abs(tau[i + 1]))
            ts = th[i] + w * (th[i + 1] - th[i])
            out.append((ts, P_CONF - sig_at(probes, round(ts / STEP) * STEP)))
    return out


def panel_element(ax, probes, pdirs, k, scale):
    """The observer's element turning through a state that does not turn.

    Deliberately spare. The wall tractions that drive the solve are left
    out here -- this panel is about frames, not about what is applied,
    and the green axes already say everything about the state.
    """
    theta, M = frame_matrix(probes, k)
    c = CENTRE
    e1 = np.array([-np.sin(np.radians(theta)), np.cos(np.radians(theta))])
    e2 = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                               edgecolor="0.55"))

    # the state itself: the MEASURED principal directions, drawn across
    # the whole box. They never move, whatever the element does.
    for th_star, s_star in pdirs:
        d = np.array([-np.sin(np.radians(th_star)),
                      np.cos(np.radians(th_star))])
        ax.plot([c[0] - 0.46 * d[0], c[0] + 0.46 * d[0]],
                [c[1] - 0.46 * d[1], c[1] + 0.46 * d[1]],
                "--", color=INV_C, lw=1.5, alpha=0.9, zorder=1)
        for sgn in (+1, -1):
            ax.annotate("", xytext=c + sgn * 0.44 * d, xy=c + sgn * 0.30 * d,
                        arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                        color=INV_C), zorder=2)
        p = c + 0.545 * d
        ax.text(p[0], p[1], f"{s_star:.2f}", fontsize=9.5, color=INV_C,
                ha="center", va="center", zorder=6)

    # the observer: a square element aligned to the rotating frame
    h = 0.13
    corners = np.array([c + a * h * e1 + b * h * e2
                        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    ax.add_patch(plt.Polygon(corners, closed=True, fill=True, fc="white",
                             ec=FRAME_C, lw=1.8, zorder=3))
    for m, tang, snn, snt in ((e1, e2, M[0, 0], M[0, 1]),
                              (-e1, -e2, M[0, 0], M[0, 1]),
                              (e2, -e1, M[1, 1], M[0, 1]),
                              (-e2, e1, M[1, 1], M[0, 1])):
        base = c + h * m
        ax.annotate("", xytext=base + scale * snn * m, xy=base,
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=NORM_C),
                    zorder=4)
        ax.annotate("", xytext=base, xy=base + scale * snt * tang,
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=SHEAR_C),
                    zorder=5)
    for vec, lab in ((e1, r"$\hat n$"), (e2, r"$\hat t$")):
        p = c + 0.215 * vec
        ax.text(p[0], p[1], lab, fontsize=11, color=FRAME_C, zorder=6,
                ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.8))

    ax.text(0.03, 0.95, rf"frame at ${theta:.1f}°$", fontsize=11.5,
            color=FRAME_C, transform=ax.transAxes)
    ax.text(0.03, 0.045, "the state (fixed)", fontsize=9.5, color=INV_C,
            transform=ax.transAxes)
    ax.text(0.97, 0.045, "the element (turning)", fontsize=9.5,
            color=FRAME_C, ha="right", transform=ax.transAxes)
    ax.set_xlim(-0.26, 1.26)
    ax.set_ylim(-0.26, 1.26)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Turning the observer, not the stress", fontsize=10.5)


def panel_numbers(ax, probes, k, spread):
    """The matrix in this frame, and what it is made of."""
    theta, M = frame_matrix(probes, k)
    sm, rad = invariants(M)
    diagonal = abs(M[0, 1]) < 0.04

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, "the same state, written in this frame",
            fontsize=10.5, ha="center", color=VARY_C)

    for x in (0.135, 0.865):
        s = 1 if x < 0.5 else -1
        ax.plot([x + s * 0.035, x, x, x + s * 0.035],
                [0.86, 0.86, 0.60, 0.60], "-", color="0.3", lw=1.6)
    ax.text(0.09, 0.73, r"$\boldsymbol{\sigma} =$", fontsize=13, ha="right",
            va="center")
    for (i, j), (x, y) in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                              ((0.34, 0.795), (0.66, 0.795),
                               (0.34, 0.665), (0.66, 0.665))):
        ax.text(x, y, f"{M[i, j]:+.2f}", fontsize=14, ha="center",
                va="center", color=VARY_C,
                fontweight="bold" if i == j else "normal")
    ax.text(0.34, 0.895, r"$\hat n$", fontsize=10, ha="center", color=FRAME_C)
    ax.text(0.66, 0.895, r"$\hat t$", fontsize=10, ha="center", color=FRAME_C)
    ax.text(0.5, 0.555,
            "all four change with the frame" if not diagonal
            else "the shear has vanished: this is the principal frame",
            fontsize=9.5, ha="center",
            color=VARY_C if not diagonal else INV_C)

    ax.plot([0.06, 0.94], [0.485, 0.485], "-", color="0.8", lw=1.0)
    ax.add_patch(plt.Rectangle((0.05, 0.045), 0.90, 0.395, fill=True,
                               fc="#eef5ee", ec=INV_C, lw=1.2, zorder=0))
    ax.text(0.5, 0.395, "built from those same four numbers —\n"
                        "and they do not move", fontsize=9.5, ha="center",
            va="center", color=INV_C)
    rows = ((r"$\sigma_m = \frac{1}{2}(\sigma_{nn}+\sigma_{tt})$", sm),
            (r"$R = \sqrt{\left(\frac{\sigma_{nn}-\sigma_{tt}}{2}\right)^2"
             r" + \sigma_{nt}^2}$", rad),
            (r"$\sigma_1 = \sigma_m + R$", sm + rad),
            (r"$\sigma_3 = \sigma_m - R$", sm - rad))
    for r, (lab, val) in enumerate(rows):
        y = 0.295 - 0.075 * r
        ax.text(0.11, y, lab, fontsize=10.5, va="center", color=INV_C)
        ax.text(0.90, y, f"{val:.3f}", fontsize=12, va="center", ha="right",
                color=INV_C, fontweight="bold")
    ax.text(0.5, -0.045, "measured spread over 24 independent solves:  "
                         rf"$\sigma_m\,\pm${spread[0]:.0e},  "
                         rf"$R\,\pm${spread[1]:.0e}",
            fontsize=8, ha="center", color="0.45")


def panel_traces(ax, probes, k, sm_all, rad_all):
    """Components wobbling; invariants dead flat. The whole argument."""
    th = probes[:, 0]
    comp = np.array([frame_matrix(probes, i)[1] for i in range(len(probes))])
    ax.axhline(0, color="0.85", lw=0.8)
    series = ((comp[:, 0, 0], NORM_C, "-", r"$\sigma_{nn}$"),
              (comp[:, 1, 1], NORM_C, "--", r"$\sigma_{tt}$"),
              (comp[:, 0, 1], SHEAR_C, "-", r"$\sigma_{nt}$"))
    for y, col, ls, lab in series:
        ax.plot(th[:k + 1], y[:k + 1], ls, lw=1.7, color=col, label=lab)
        ax.plot([th[k]], [y[k]], "o", ms=8, color=col, zorder=5)
    for y, lab in ((sm_all, r"$\sigma_m$"), (rad_all, "$R$")):
        ax.plot(th[:k + 1], y[:k + 1], "-", lw=2.6, color=INV_C, alpha=0.9,
                label=lab)
        ax.plot([th[k]], [y[k]], "o", ms=8, color=INV_C, zorder=5)
    ax.text(172, 0.5 * (sm_all[0] + rad_all[0]), "invariant",
            fontsize=9.5, color=INV_C, ha="right", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    ax.set_xlim(-6, 186)
    ax.set_ylim(-1.6 * R_AN, P_CONF + 2.1 * R_AN)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("orientation of the frame")
    ax.set_ylabel("stress (compression positive)")
    ax.legend(fontsize=8.5, loc="upper left", ncol=5,
              framealpha=0.9, columnspacing=1.1,
              handlelength=1.6)
    ax.set_title("3 components − 1 angle = 2 invariants", fontsize=10.5)


def save_gif(frames, out, colors=None):
    images = [Image.open(f) for f in frames]
    if colors:
        images = [im.convert("RGB") for im in images]
        pal = images[len(images) // 2].quantize(colors=colors,
                                                method=Image.MEDIANCUT)
        images = [im.quantize(palette=pal, dither=Image.NONE)
                  for im in images]
    images[0].save(out, save_all=True,
                   append_images=images[1:] + [images[-1]] * 6,
                   duration=280, loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, "
          f"{os.path.getsize(out) / 1024:.0f} KB)")


probes = load_probes()
pdirs = principal_dirs(probes)
inv = np.array([invariants(frame_matrix(probes, i)[1])
                for i in range(len(probes))])
sm_all, rad_all = inv[:, 0], inv[:, 1]
spread = (float(sm_all.max() - sm_all.min()), float(rad_all.max() - rad_all.min()))
print(f"principal directions (measured): "
      + ", ".join(f"{t:.1f}° ({s:.2f})" for t, s in pdirs))
print(f"sigma_m  {sm_all.mean():.4f}  spread {spread[0]:.4f}   "
      f"(imposed P = {P_CONF})")
print(f"R        {rad_all.mean():.4f}  spread {spread[1]:.4f}   "
      f"(analytic {R_AN:.4f})")

SCALE = 0.13 / (P_CONF + R_AN)
frames = []
for k in range(len(probes)):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7),
                             gridspec_kw=dict(width_ratios=[1.0, 0.92, 1.3]))
    panel_element(axes[0], probes, pdirs, k, SCALE)
    panel_numbers(axes[1], probes, k, spread)
    panel_traces(axes[2], probes, k, sm_all, rad_all)
    fig.suptitle("Two descriptions, one state: what survives the change "
                 "of frame", fontsize=12)
    fig.tight_layout()
    frame = os.path.join(D, f"_inv_frame_{k:03d}.png")
    fig.savefig(frame, dpi=100)
    plt.close(fig)
    frames.append(frame)

save_gif(frames, os.path.join(D, "invariants.gif"), colors=128)

k_star = int(np.argmin(np.abs(probes[:, 2])))
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7),
                         gridspec_kw=dict(width_ratios=[1.0, 0.92, 1.3]))
panel_element(axes[0], probes, pdirs, k_star, SCALE)
panel_numbers(axes[1], probes, k_star, spread)
panel_traces(axes[2], probes, len(probes) - 1, sm_all, rad_all)
fig.suptitle("Two descriptions, one state: what survives the change of frame",
             fontsize=12)
fig.tight_layout()
out = os.path.join(D, "invariants.png")
fig.savefig(out, dpi=190)
plt.close(fig)
print(f"wrote {out}  (principal frame at theta = {probes[k_star, 0]:.1f}°)")
