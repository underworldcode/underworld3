"""What a fault feels: the traction resolved in the fault's own frame.

The companion to mohr_animate.py / mohr_two_fields.py, for students who
are not yet ready for the Mohr plane. Deliberately there is NO Mohr
circle here. A small fault turns inside a fixed stress field and we
watch, at each orientation, the one thing the fault actually cares
about: the traction on its surface, split into the push across it and
the drag along it.

Three panels:

  left    the map. The applied field, the turning fault, and the
          traction vector sigma . n on its plane.
  middle  the SAME instant seen from the fault: the fault held still,
          the traction decomposed onto its own axes (n_hat, t_hat).
          This is the fault reference frame -- the fault does not know
          which way north is, only how hard it is squeezed and how hard
          it is dragged.
  right   sigma_n and tau against orientation, traced out as the sweep
          runs. Where tau crosses zero the plane is a principal plane,
          and the principal directions appear on the map as they are
          found.

Every number is measured: this reads the committed probe cache written
by mohr_two_fields.py (a full welded-fault Stokes solve per angle). No
solver import, so the figure can be restyled anywhere numpy and
matplotlib exist. If the cache is missing, run mohr_two_fields.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

D = os.path.dirname(os.path.abspath(__file__))

# geometry of the solves that produced the cache (common.CENTRE, HALF)
CENTRE = np.array([0.5, 0.5])
HALF = 0.2

# field A of mohr_two_fields: shear_plus_stretch(a=0.5, gamma=1.0) at
# eta = 1. Deviatoric, so the analytic radius is |sigma|_2.
A_RATE, GAMMA, ETA = 0.5, 1.0, 1.0
SIG = ETA * np.array([[2 * A_RATE, GAMMA], [GAMMA, -2 * A_RATE]])
R = float(np.linalg.norm(SIG, 2))

PROBE_C = "#c62828"      # the fault: the thing being interrogated
AXIS_C = "#1b5e20"       # principal axes, once discovered
TRAC_C = "#4a7bf7"       # the full traction vector
NORM_C = "#8e24aa"       # its normal component
SHEAR_C = "#ef6c00"      # its shear component
DRIVE_C = "#6d4c41"      # the applied far-field state


def load_probes():
    cache = os.path.join(D, "_mohr2_probes_A.npz")
    assert os.path.exists(cache), (
        "missing _mohr2_probes_A.npz -- run mohr_two_fields.py first")
    p = np.load(cache)["probes"]
    print(f"loaded {len(p)} measured probes, "
          f"theta {p[0, 0]:.1f}..{p[-1, 0]:.1f} deg")
    return p


def zero_crossings(probes):
    """Orientations where the MEASURED shear traction changes sign.

    tau = 0 means the traction is parallel to the plane's normal, so
    that normal is a principal direction. Found in the data rather than
    taken from the drive, so the animation discovers the axes the same
    way a field geologist would.
    """
    th, sig, tau = probes[:, 0], probes[:, 1], probes[:, 2]
    out = []
    for i in range(len(tau) - 1):
        if tau[i] == 0.0 or tau[i] * tau[i + 1] < 0.0:
            w = abs(tau[i]) / (abs(tau[i]) + abs(tau[i + 1]))
            out.append((th[i] + w * (th[i + 1] - th[i]),
                        sig[i] + w * (sig[i + 1] - sig[i]),
                        th[i + 1]))
    return out


def draw_drive(ax, sigma, scale, centre=CENTRE, box=1.0):
    """Wall tractions of the applied field, pointing INTO the box under
    compression -- what is being applied, before anything is measured."""
    for n in ([1, 0], [-1, 0], [0, 1], [0, -1]):
        n = np.array(n, dtype=float)
        base = centre + 0.5 * box * n
        T = sigma @ n
        ax.annotate("", xytext=base + scale * T, xy=base,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                    color=DRIVE_C, alpha=0.85))


def panel_map(ax, probes, crossings, k):
    """The fault turning inside the fixed applied field."""
    theta, sig_k, tau_k = probes[k]
    c = CENTRE
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                               edgecolor="0.4"))
    draw_drive(ax, SIG, 0.22 / R)

    for (th_star, sig_star, reveal_at) in crossings:
        if theta + 1e-9 < reveal_at:
            continue
        ts = np.array([np.cos(np.radians(th_star)),
                       np.sin(np.radians(th_star))])
        a = np.array([-ts[1], ts[0]])
        ax.plot([c[0] - 0.44 * a[0], c[0] + 0.44 * a[0]],
                [c[1] - 0.44 * a[1], c[1] + 0.44 * a[1]],
                "--", color=AXIS_C, lw=1.4, alpha=0.85, zorder=1)
        lab = c + 0.38 * a
        ax.text(lab[0], lab[1], f"{-sig_star:+.2f}", fontsize=8.5,
                color=AXIS_C, ha="center", va="center", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))

    t = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    n = np.array([-t[1], t[0]])
    ax.plot([c[0] - HALF * t[0], c[0] + HALF * t[0]],
            [c[1] - HALF * t[1], c[1] + HALF * t[1]],
            "-", color=PROBE_C, lw=3.4, solid_capstyle="butt", zorder=3)

    # the fault's own axes travel with it: this is the frame the middle
    # panel is drawn in, so it has to be visible here too
    for vec, lab in ((n, r"$\hat n$"), (t, r"$\hat t$")):
        ax.annotate("", xytext=c, xy=c + 0.16 * vec,
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="0.35"),
                    zorder=4)
        p = c + 0.205 * vec
        ax.text(p[0], p[1], lab, fontsize=9.5, ha="center", va="center",
                color="0.35", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))

    T = sig_k * n + tau_k * t
    ax.annotate("", xytext=c, xy=c + (0.26 / R) * T,
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=TRAC_C),
                zorder=4)

    ax.text(0.04, 0.94, rf"$\theta = {theta:.1f}°$", fontsize=12,
            transform=ax.transAxes)
    ax.text(0.04, 0.055, r"$\vec T = \boldsymbol{\sigma}\cdot\hat n$",
            fontsize=10, color=TRAC_C, transform=ax.transAxes)
    ax.text(0.97, 0.055, "applied field", fontsize=9, color=DRIVE_C,
            ha="right", transform=ax.transAxes)
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(-0.20, 1.20)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("The fault turns in a fixed field", fontsize=10.5)


def panel_fault_frame(ax, probes, k):
    """The same instant, seen by the fault.

    The fault is held horizontal and the world turns instead: that is
    what "the fault reference frame" means. The arrows are the traction
    T = sigma . n_hat, which is the force per area on the face whose
    outward normal is n_hat -- the top of the block BELOW. Drawn as it
    acts, so compression points down into that block, the textbook way.
    The numbers are quoted compression positive, the geological
    convention and the opposite of the solver's; the flip happens here
    and in the trace panel only.
    """
    theta, sig_k, tau_k = probes[k]
    sn = -sig_k                      # geological: squeezing is positive
    sc = 0.42 / R

    ax.add_patch(plt.Rectangle((-0.78, 0.055), 1.56, 0.645, fc="0.955",
                               ec="none", zorder=0))
    ax.add_patch(plt.Rectangle((-0.78, -0.70), 1.56, 0.645, fc="0.895",
                               ec="none", zorder=0))
    ax.text(-0.74, 0.63, "block above", fontsize=8, color="0.5", ha="left")
    ax.text(-0.74, -0.90, "block below", fontsize=8, color="0.5", ha="left",
            va="center")
    ax.plot([-0.78, 0.78], [0, 0], "-", color=PROBE_C, lw=3.6,
            solid_capstyle="butt", zorder=3)

    # the fault's axes ARE the axes of the page here
    a = np.array([0.52, -0.74])
    ax.annotate("", xytext=a, xy=a + [0.0, 0.16],
                arrowprops=dict(arrowstyle="->", lw=1.2, color="0.4"))
    ax.text(a[0] + 0.045, a[1] + 0.14, r"$\hat n$", fontsize=10, color="0.4")
    ax.annotate("", xytext=a, xy=a + [0.16, 0.0],
                arrowprops=dict(arrowstyle="->", lw=1.2, color="0.4"))
    ax.text(a[0] + 0.15, a[1] - 0.12, r"$\hat t$", fontsize=10, color="0.4")

    # --- the decomposition ---------------------------------------------
    O = np.array([-0.30, 0.0])
    Tn = np.array([0.0, sc * sig_k])
    Tt = np.array([sc * tau_k, 0.0])
    ax.plot([(O + Tt)[0], (O + Tn + Tt)[0]], [(O + Tt)[1], (O + Tn + Tt)[1]],
            ":", color="0.5", lw=1.0, zorder=2)
    ax.plot([(O + Tn)[0], (O + Tn + Tt)[0]], [(O + Tn)[1], (O + Tn + Tt)[1]],
            ":", color="0.5", lw=1.0, zorder=2)
    # components heavier than the resultant, so the two stay legible even
    # at the principal orientations where they lie on top of each other
    ax.annotate("", xytext=O, xy=O + Tn,
                arrowprops=dict(arrowstyle="-|>", lw=3.4, color=NORM_C),
                zorder=4)
    ax.annotate("", xytext=O, xy=O + Tt,
                arrowprops=dict(arrowstyle="-|>", lw=3.4, color=SHEAR_C),
                zorder=4)
    ax.annotate("", xytext=O, xy=O + Tn + Tt,
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=TRAC_C),
                zorder=5)
    ax.text(-0.02, -0.90, r"$\vec T = \sigma_n\hat n + \tau\hat t$",
            fontsize=10.5, color=TRAC_C, ha="center", va="center")

    # --- the twin, so the couple is visible ----------------------------
    P = np.array([0.40, 0.0])
    ax.annotate("", xytext=P, xy=P - Tn - Tt,
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=TRAC_C,
                                alpha=0.55), zorder=4)
    ax.text(0.46, 0.60, "equal and opposite\non the other side",
            fontsize=8, color="0.45", ha="center", va="center")

    sense = "compression" if sn >= 0 else "tension"
    ax.text(-0.76, 0.90, rf"$\sigma_n = {sn:+.2f}$  ({sense})", fontsize=11,
            color=NORM_C, ha="left", va="center")
    ax.text(-0.76, 0.78, rf"$\tau = {tau_k:+.2f}$  (drag along)",
            fontsize=11, color=SHEAR_C, ha="left", va="center")

    ax.set_xlim(-0.80, 0.80)
    ax.set_ylim(-1.00, 1.00)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("…in the fault's own frame", fontsize=10.5)


def panel_traces(ax, probes, crossings, k, k_full=None):
    """sigma_n and tau against orientation, drawn as the sweep finds them."""
    kf = k if k_full is None else k_full
    th = probes[:, 0]
    sn, ta = -probes[:, 1], probes[:, 2]
    ax.axhline(0, color="0.85", lw=0.8)
    for (th_star, _s, reveal_at) in crossings:
        if probes[kf, 0] + 1e-9 >= reveal_at:
            ax.axvline(th_star, color=AXIS_C, ls="--", lw=1.0, alpha=0.8)
            ax.text(th_star, 1.30 * R, f"{th_star:.1f}°", fontsize=8,
                    color=AXIS_C, ha="center")
    ax.plot(th[:kf + 1], sn[:kf + 1], "-o", ms=4, lw=1.6, color=NORM_C,
            label=r"$\sigma_n$  (across the fault)")
    ax.plot(th[:kf + 1], ta[:kf + 1], "-o", ms=4, lw=1.6, color=SHEAR_C,
            label=r"$\tau$  (along the fault)")
    ax.plot([th[k]], [sn[k]], "o", ms=9, color=NORM_C, zorder=5)
    ax.plot([th[k]], [ta[k]], "o", ms=9, color=SHEAR_C, zorder=5)
    ax.set_xlim(-6, 186)
    ax.set_ylim(-1.45 * R, 1.45 * R)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(r"fault orientation $\theta$")
    ax.set_ylabel("traction on the fault")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.set_title(r"where $\tau = 0$, the plane is principal", fontsize=10.5)


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
crossings = zero_crossings(probes)
print("principal orientations at "
      + ", ".join(f"{c[0]:.1f}°" for c in crossings)
      + f"   (analytic radius {R:.3f})")

frames = []
for k in range(len(probes)):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5),
                             gridspec_kw=dict(width_ratios=[1.0, 0.9, 1.25]))
    panel_map(axes[0], probes, crossings, k)
    panel_fault_frame(axes[1], probes, k)
    panel_traces(axes[2], probes, crossings, k)
    fig.suptitle("A fault samples the stress field", fontsize=12)
    fig.tight_layout()
    frame = os.path.join(D, f"_ff_frame_{k:03d}.png")
    fig.savefig(frame, dpi=100)
    plt.close(fig)
    frames.append(frame)

save_gif(frames, os.path.join(D, "fault-frame.gif"), colors=128)

# static: the principal orientation, where the shear has gone
# the static frame: both components large (so the decomposition reads),
# the normal one COMPRESSIVE (so it is the geological case), and the
# fault as oblique as those allow, since a fault parallel to the page
# makes the two frames coincide and hides the whole point
sn_all = -probes[:, 1]
score = np.minimum(np.abs(sn_all), np.abs(probes[:, 2]))
obliq = np.minimum(probes[:, 0], 180.0 - probes[:, 0])
cand = np.where(sn_all > 0)[0]
cand = cand[score[cand] > score[cand].max() - 0.02]
k_star = int(cand[np.argmax(obliq[cand])])
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5),
                         gridspec_kw=dict(width_ratios=[1.0, 0.9, 1.25]))
panel_map(axes[0], probes, [(a, b, 0.0) for a, b, _ in crossings], k_star)
panel_fault_frame(axes[1], probes, k_star)
panel_traces(axes[2], probes, crossings, k_star, k_full=len(probes) - 1)
fig.suptitle("A fault samples the stress field", fontsize=12)
fig.tight_layout()
out = os.path.join(D, "fault-frame.png")
fig.savefig(out, dpi=190)
plt.close(fig)
print(f"wrote {out}  (static at theta = {probes[k_star, 0]:.1f}°: "
      f"sigma_n {-probes[k_star, 1]:+.3f}, tau {probes[k_star, 2]:+.3f})")
