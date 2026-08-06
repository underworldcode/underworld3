"""Two stress fields, two circles: the shape depends on the state.

The teaching sequel to mohr_animate.py. The same rotating stress probe
is swept through TWO different regional stress fields, and traces two
visibly different Mohr circles — different size, different principal
orientation. The point is that the 2-theta construction is the same in
both, but the circle it builds is a property of the stress state, not
of the method.

Three things this does that mohr_animate.py does not:

  * the applied field is DRAWN on the box (wall tractions from the
    drive's analytic deviatoric stress), so you can see what is being
    probed before anything moves;
  * the principal orientations appear as LINES struck through the model
    when the probe crosses tau = 0, and stay there — the axes are
    discovered by the sweep rather than flashed as a caption. The
    crossings are found in the MEASURED probes, not from the drive;
  * the probe reads as an instrument (its own colour, its own label),
    not as one more fault in the picture.

Each frame is a full welded-fault solve. 2 x 25 solves, cached.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import common
from underworld3.utilities import fault_contact

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
STEP = 7.5
ANGLES = np.arange(0.0, 180.0 + 1e-9, STEP)

PROBE_C = "#c62828"      # the rotating probe: it is the instrument
AXIS_C = "#1b5e20"       # principal axes, once discovered
TRAC_C = "#4a7bf7"       # the measured traction vector
DRIVE_C = "#6d4c41"      # the applied far-field state


# --- the two fields ---------------------------------------------------------
#
# Chosen so the circles are unmistakably different objects at slide
# scale: B's radius is about half A's, and its principal axes sit 37.5
# degrees away.  Both drives are purely deviatoric, so both circles are
# centred on the same mean stress -- the honest difference is radius and
# orientation, and no confining-pressure shift is implied.

def drive_a(child):
    return common.shear_plus_stretch(child, a=0.5, gamma=1.0)


def drive_b(child):
    return common.pure_shear_drive(child, phi_deg=60.0, tau0=0.8)


A_RATE, GAMMA = 0.5, 1.0
PHI_B, TAU0_B = 60.0, 0.8

# analytic deviatoric stress of each drive (used for the wall arrows and
# the reference circle only -- every probe on the plot is measured)
SIG_A = common.ETA * np.array([[2 * A_RATE, GAMMA], [GAMMA, -2 * A_RATE]])
_c2, _s2 = np.cos(2 * np.radians(PHI_B)), np.sin(2 * np.radians(PHI_B))
SIG_B = TAU0_B * np.array([[-_c2, -_s2], [-_s2, _c2]])

FIELDS = [
    dict(key="A", drive=drive_a, sigma=SIG_A, cache="_mohr2_probes_A.npz",
         gif="mohr-circle-build-A.gif",
         title="Field A: shear with stretch",
         subtitle=r"$\dot\varepsilon$: $a=0.5$, $\gamma=1.0$"),
    dict(key="B", drive=drive_b, sigma=SIG_B, cache="_mohr2_probes_B.npz",
         gif="mohr-circle-build-B.gif",
         title="Field B: pure shear, compression at 60°",
         subtitle=r"$\tau_0=0.8$, $\phi=60°$"),
]

for f in FIELDS:
    f["R"] = float(np.linalg.norm(f["sigma"], 2))   # deviatoric radius

R_MAX = max(f["R"] for f in FIELDS)
print(f"analytic radii: A {FIELDS[0]['R']:.4f}  B {FIELDS[1]['R']:.4f}  "
      f"(ratio {FIELDS[0]['R'] / FIELDS[1]['R']:.2f})")


def probe_under(drive, theta, half_length=HALF, cell_size=0.04):
    """One welded-fault stress probe under an ARBITRARY drive.

    common.mohr_probe is hardwired to shear_plus_stretch; this is the
    same measurement with the drive passed in. Reads the shear traction
    through the weld's own law and the normal traction through the
    no-opening reaction, exactly as mohr_probe does.
    """
    eta_weld = 200.0 * common.ETA / half_length
    child = common.split_with_fault(
        common.base_mesh(cell_size),
        common.fault_segment(theta, half_length))
    stokes = common.stokes_on(child, drive(child))
    stokes.add_fault_bc(eta_weld, boundary="Fault")
    fault_contact.solve_with_fault(stokes, picard=2)
    s, V, _leak = common.slip_profile(stokes)
    s_n, sig = common.normal_traction(stokes)
    tau = eta_weld * float(np.median(V[common.inner(s)]))
    sigma_n = float(np.median(sig[common.inner(s_n)]))
    return sigma_n, tau


def sweep(field):
    """(theta, sigma_n, tau) for the whole rotation, cached."""
    cache = os.path.join(D, field["cache"])
    if os.path.exists(cache):
        probes = np.load(cache)["probes"]
        assert len(probes) == len(ANGLES), "cache is stale — delete it"
        print(f"field {field['key']}: loaded {len(probes)} cached probes")
        return probes
    rows = []
    for theta in ANGLES:
        sigma_n, tau = probe_under(field["drive"], theta)
        rows.append((theta, sigma_n, tau))
        print(f"  {field['key']} theta {theta:6.1f}: "
              f"sigma_n {sigma_n:8.4f}  tau {tau:8.4f}")
    probes = np.array(rows)
    np.savez(cache, probes=probes)
    return probes


def zero_crossings(probes):
    """Fault angles where the measured shear traction changes sign.

    At tau = 0 the fault plane is a principal plane, so its NORMAL is a
    principal direction carrying the measured sigma_n. Linear
    interpolation between the bracketing solves is plenty at 7.5 degree
    steps and keeps the axes measured rather than assumed.
    """
    th, sig, tau = probes[:, 0], probes[:, 1], probes[:, 2]
    out = []
    for i in range(len(tau) - 1):
        if tau[i] == 0.0 or tau[i] * tau[i + 1] < 0.0:
            w = abs(tau[i]) / (abs(tau[i]) + abs(tau[i + 1]))
            out.append((th[i] + w * (th[i + 1] - th[i]),
                        sig[i] + w * (sig[i + 1] - sig[i]),
                        th[i + 1]))          # revealed once the sweep passes
    return out


def draw_drive(ax, sigma, scale):
    """Wall tractions of the applied field: sigma . n at each wall
    midpoint, pointing INTO the box under compression."""
    for n in ([1, 0], [-1, 0], [0, 1], [0, -1]):
        n = np.array(n, dtype=float)
        base = common.CENTRE + 0.5 * n
        T = sigma @ n
        ax.annotate("", xytext=base + scale * T, xy=base,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                    color=DRIVE_C, alpha=0.85))


def render(field, probes):
    crossings = zero_crossings(probes)
    print(f"field {field['key']}: principal orientations at "
          + ", ".join(f"{c[0]:.1f}°" for c in crossings))

    centre = float(np.mean(probes[:, 1]))
    cg = -centre                      # geological convention: compression +
    scg = -probes[:, 1]
    R = field["R"]
    drive_scale = 0.22 / R_MAX

    frames = []
    for k in range(len(probes)):
        theta, sig_k, tau_k = probes[k]
        fig, (axl, axr) = plt.subplots(
            1, 2, figsize=(9.6, 4.6),
            gridspec_kw=dict(width_ratios=[1.0, 1.25]))

        # ---- left: the probe turning inside the applied field ----------
        axl.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                                    edgecolor="0.4"))
        draw_drive(axl, field["sigma"], drive_scale)

        c = common.CENTRE
        # Principal axes, revealed as the sweep discovers them and kept.
        # The FIRST crossing already fixes both directions (they are
        # orthogonal), so both lines appear then; each carries its value
        # only once its own crossing has measured it.
        if crossings and theta + 1e-9 >= crossings[0][2]:
            th0 = crossings[0][0]
            t_s = np.array([np.cos(np.radians(th0)), np.sin(np.radians(th0))])
            n_s = np.array([-t_s[1], t_s[0]])
            for axis in (n_s, t_s):
                axl.plot([c[0] - 0.44 * axis[0], c[0] + 0.44 * axis[0]],
                         [c[1] - 0.44 * axis[1], c[1] + 0.44 * axis[1]],
                         "--", color=AXIS_C, lw=1.4, alpha=0.85, zorder=1)
            for (th_star, sig_star, reveal_at) in crossings:
                if theta + 1e-9 < reveal_at:
                    continue
                ts = np.array([np.cos(np.radians(th_star)),
                               np.sin(np.radians(th_star))])
                ax_dir = np.array([-ts[1], ts[0]])
                lab = c + 0.38 * ax_dir
                axl.text(lab[0], lab[1], f"{-sig_star:+.2f}", fontsize=8.5,
                         color=AXIS_C, ha="center", va="center", zorder=5,
                         bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))

        t = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
        n = np.array([-t[1], t[0]])
        # the probe: heavier than anything else, with gauge ticks
        axl.plot([c[0] - HALF * t[0], c[0] + HALF * t[0]],
                 [c[1] - HALF * t[1], c[1] + HALF * t[1]],
                 "-", color=PROBE_C, lw=3.2, solid_capstyle="butt", zorder=3)
        for f_ in (-0.75, -0.25, 0.25, 0.75):
            p = c + f_ * HALF * t
            axl.plot([p[0] - 0.022 * n[0], p[0] + 0.022 * n[0]],
                     [p[1] - 0.022 * n[1], p[1] + 0.022 * n[1]],
                     "-", color=PROBE_C, lw=1.0, zorder=3)
        axl.annotate("", xytext=c, xy=c + 0.085 * n,
                     arrowprops=dict(arrowstyle="->", lw=0.9, color="0.45"))
        axl.text(*(c + 0.115 * n), r"$\hat n$", fontsize=8.5, ha="center",
                 va="center", color="0.45")

        # the traction the plane actually feels, from the MEASURED probe
        # (one scale for both fields, so B reads as the weaker state)
        T = sig_k * n + tau_k * t
        axl.annotate("", xytext=c, xy=c + (0.26 / R_MAX) * T,
                     arrowprops=dict(arrowstyle="-|>", lw=2.4, color=TRAC_C),
                     zorder=4)

        axl.text(0.04, 0.94, rf"$\theta = {theta:.1f}°$", fontsize=12,
                 transform=axl.transAxes)
        axl.text(0.04, 0.10, "stress probe", fontsize=9, color=PROBE_C,
                 transform=axl.transAxes)
        axl.text(0.04, 0.045, r"$\sigma\cdot\hat n$: traction on its plane",
                 fontsize=9, color=TRAC_C, transform=axl.transAxes)
        axl.text(0.97, 0.045, "applied field", fontsize=9, color=DRIVE_C,
                 ha="right", transform=axl.transAxes)
        axl.set_xlim(-0.20, 1.20)
        axl.set_ylim(-0.20, 1.20)
        axl.set_aspect("equal")
        axl.set_xticks([])
        axl.set_yticks([])
        axl.set_title(field["title"], fontsize=10)

        # ---- right: the circle this field builds -----------------------
        tt = np.linspace(0, 2 * np.pi, 300)
        axr.plot(cg + R * np.cos(tt), R * np.sin(tt), "-",
                 color="0.75", lw=0.9)
        axr.axhline(0, color="0.85", lw=0.6)
        axr.axvline(cg, color="0.85", lw=0.6)
        axr.plot([cg - R, cg + R], [0, 0], "D", ms=5, color=AXIS_C, zorder=4)
        axr.text(cg, -1.20 * R_MAX, "principal stresses", fontsize=7.5,
                 ha="center", va="center", color=AXIS_C)
        axr.plot(scg[:k + 1], probes[:k + 1, 2], "o", ms=5, mfc="none",
                 mec=PROBE_C, mew=1.2)
        axr.plot([cg, -sig_k], [0.0, tau_k], "-", color=TRAC_C, lw=1.2)
        axr.plot([-sig_k], [tau_k], "o", ms=9, color=PROBE_C, zorder=5)
        axr.text(0.04, 0.94, r"the probe sweeps at $2\theta$", fontsize=10,
                 transform=axr.transAxes)
        axr.text(0.04, 0.88, field["subtitle"], fontsize=9, color=DRIVE_C,
                 transform=axr.transAxes)
        axr.set_xlabel(r"normal stress $\sigma$ (compression positive)")
        axr.set_ylabel(r"shear traction $\tau$")
        # identical limits for both fields: the circles must be comparable
        axr.set_xlim(cg - 1.35 * R_MAX, cg + 1.35 * R_MAX)
        axr.set_ylim(-1.35 * R_MAX, 1.35 * R_MAX)
        axr.set_aspect("equal")

        fig.suptitle("A rotating probe measures the stress field",
                     fontsize=11)
        fig.tight_layout()
        frame = os.path.join(D, f"_m2{field['key']}_frame_{k:03d}.png")
        fig.savefig(frame, dpi=110)
        plt.close(fig)
        frames.append(frame)

    images = [Image.open(f) for f in frames]
    out = os.path.join(D, field["gif"])
    images[0].save(out, save_all=True,
                   append_images=images[1:] + [images[-1]] * 6,
                   duration=280, loop=0)
    size_kb = os.path.getsize(out) / 1024
    print(f"wrote {out} ({len(frames)} frames, {size_kb:.0f} KB)")
    return centre


# --- run --------------------------------------------------------------------
results = []
for field in FIELDS:
    probes = sweep(field)
    centre = render(field, probes)
    results.append((field, probes, centre))

# --- the payoff: both circles on one plane ----------------------------------
fig, ax = plt.subplots(figsize=(6.2, 5.6))
tt = np.linspace(0, 2 * np.pi, 400)
for (field, probes, centre), col, ls in zip(results,
                                            ["#c62828", "#1565c0"],
                                            ["-", "-"]):
    cg = -centre
    R = field["R"]
    ax.plot(cg + R * np.cos(tt), R * np.sin(tt), ls, color=col, lw=1.4,
            alpha=0.55)
    ax.plot(-probes[:, 1], probes[:, 2], "o", ms=5, mfc="none", mec=col,
            mew=1.2, label=f"{field['title']}  ($R={R:.2f}$)")
    ax.plot([cg - R, cg + R], [0, 0], "D", ms=5, color=col, zorder=4)
ax.axhline(0, color="0.85", lw=0.6)
ax.set_xlabel(r"normal stress $\sigma$ (compression positive)")
ax.set_ylabel(r"shear traction $\tau$")
ax.set_title("Same construction, different stress state")
ax.legend(fontsize=8, loc="upper right")
ax.set_aspect("equal")
fig.tight_layout()
out = os.path.join(D, "mohr-two-circles.png")
fig.savefig(out, dpi=200)
print("wrote", out)
