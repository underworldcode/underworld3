"""Southern California, read by rotating stress gauges.

The animated sequel to california.py. Instead of plotting each
neighbour's fixed-orientation probe cloud, a small WELDED GAUGE is
placed at each of the three neighbour locations and ROTATED through 180
degrees. Each gauge traces out its own local Mohr circle — a clock face
per site — and the whole sweep is run twice:

    ambient   (San Andreas welded)  -> the grey circle
    post-slip (San Andreas free)    -> the red circle

so Delta CFF appears as MOTION of the circle toward or away from the
failure envelope, rather than as a number. The sites do not see the
same stress state, which is the point: a single regional drive plus one
curved fault produces a different local circle at each place.

Cost: the three gauges share one assembler, so all three rotate inside a
SINGLE solve per angle -- 2 x 25 solves, not 2 x 25 x 3. Cached.

Gauge discipline: a closed velocity-driven box fixes pressure only to a
per-solve constant. A fourth, FIXED gauge sits far from the San Andreas
as a pressure reference; its ambient normal stress is analytic
(ambient_sigma_n_simple), so the constant is measured and removed per
solve, and printed. The far-field reference is the same principle as
common.far_field_anchor: a slip event changes nothing far away.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import underworld3 as uw
from underworld3.utilities import fault_contact

import common
from california import (SAF_PTS, MINORS, TREND, TAU0, MU_P, P0, COH,
                        ETA_WELD, saf_normal, _t, _n)

D = os.path.dirname(os.path.abspath(__file__))
H = 0.012
LIG = 2 * H
L_GAUGE = 0.05                      # gauge half-length (max safe 0.0617)
STEP = 7.5
ANGLES = np.arange(0.0, 180.0 + 1e-9, STEP)

# the three neighbour locations of california.py, as single sites
SITES = [
    ("Garlock", np.array([0.66, 0.485]), "#6a1b9a"),
    ("ECSZ", np.mean([MINORS[k].mean(axis=0) for k in ("E1", "E2", "E3")],
                     axis=0), "#1a6b1a"),
    ("SJF", MINORS["SJF"].mean(axis=0), "#e65100"),
]
LABEL = {"Garlock": "Garlock", "ECSZ": "ECSZ",
         "SJF": "San Jacinto"}
# label offsets from the gauge centre — the San Jacinto site sits close
# to the trace (and SW of it, on the Pacific side), so its label goes
# south, away from the trace
LAB_OFF = {"Garlock": (0.0, -0.085, "center", "top"),
           "ECSZ": (0.0, -0.085, "center", "top"),
           "SJF": (0.0, -0.085, "center", "top")}

# pressure reference: fixed, far from the trace, in the quiet corner
REF_C = np.array([0.16, 0.20])
REF_ANG = 0.0
REF_T = np.array([np.cos(np.radians(REF_ANG)), np.sin(np.radians(REF_ANG))])


def _seg(centre, angle_deg, half=L_GAUGE):
    t = np.array([np.cos(np.radians(angle_deg)),
                  np.sin(np.radians(angle_deg))])
    return np.array([centre - half * t, centre + half * t])


def _d_point_polyline(p, pts):
    best = np.inf
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        u = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        best = min(best, np.linalg.norm(p - (a + u * ab)))
    return best


def check_clearances():
    """A rotating gauge sweeps a disc of radius L_GAUGE. Every disc must
    clear the trace, the walls, the reference gauge and each other by at
    least one ligament. A gauge that clips the trunk produces a
    plausible figure rather than an error, so this is an assertion."""
    centres = [(n, c) for n, c, _ in SITES] + [("ref", REF_C)]
    for name, c in centres:
        d_saf = _d_point_polyline(c, SAF_PTS)
        d_wall = min(c[0], 1 - c[0], c[1], 1 - c[1])
        print(f"  {name:8s} d(SAF)={d_saf:.4f}  d(wall)={d_wall:.4f}")
        assert d_saf - L_GAUGE >= LIG, f"{name}: gauge disc reaches the SAF"
        assert d_wall - L_GAUGE >= LIG, f"{name}: gauge disc reaches a wall"
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            d = np.linalg.norm(centres[i][1] - centres[j][1])
            assert d - 2 * L_GAUGE >= LIG, \
                f"{centres[i][0]}-{centres[j][0]} discs overlap"
    n_nodes = 2 * L_GAUGE / H
    print(f"  clearances OK; ~{n_nodes:.0f} nodes along each gauge")


def solve_at(angle, trunk_free):
    """One solve with all three gauges at `angle`, plus the fixed
    reference. Returns {site: (sigma_n, tau)} medians, gauge-anchored."""
    faults = [("SAF", SAF_PTS)]
    faults += [(name, _seg(c, angle)) for name, c, _ in SITES]
    faults += [("REF", _seg(REF_C, REF_ANG))]

    child = common.base_mesh(H).add_fault(faults)
    stokes = common.stokes_on(
        child, common.boundary_simple_shear(child, TREND, TAU0))
    stokes.add_fault_bc(0 if trunk_free else ETA_WELD, boundary="SAF",
                        normal=saf_normal(child))
    for name, _c, _col in SITES:
        stokes.add_fault_bc(ETA_WELD, boundary=name)
    stokes.add_fault_bc(ETA_WELD, boundary="REF")
    fault_contact.solve_with_fault(stokes, picard=2)

    # the per-solve pressure constant, from the far reference gauge
    s_r, _xy, sig_r, _tau_r = common.probe_nodes(stokes, "REF", REF_T,
                                                 ETA_WELD)
    c_gauge = (float(np.median(sig_r[common.inner(s_r)]))
               - common.ambient_sigma_n_simple(TREND, REF_T, TAU0))

    out = {}
    t_hat = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    for name, _c, _col in SITES:
        s, _xy, sig, tau = common.probe_nodes(stokes, name, t_hat, ETA_WELD)
        m = common.inner(s)
        out[name] = (float(np.median(sig[m])) - c_gauge,
                     float(np.median(tau[m])))
    return out, c_gauge


def sweep():
    cache = os.path.join(D, "_california_clocks.npz")
    if os.path.exists(cache):
        d = dict(np.load(cache))
        assert len(d["angles"]) == len(ANGLES), "cache is stale — delete it"
        print(f"loaded cached sweep ({len(ANGLES)} angles x 2 states)")
        return d
    check_clearances()
    d = {"angles": ANGLES}
    for state, trunk_free in (("amb", False), ("slip", True)):
        sig = {n: [] for n, _c, _col in SITES}
        tau = {n: [] for n, _c, _col in SITES}
        t_wall = time.perf_counter()
        for angle in ANGLES:
            res, c_gauge = solve_at(angle, trunk_free)
            for name, _c, _col in SITES:
                sig[name].append(res[name][0])
                tau[name].append(res[name][1])
            print(f"  [{state}] theta {angle:6.1f}  gauge const "
                  f"{c_gauge:+9.4f}  "
                  + "  ".join(f"{n}: sig {res[n][0]:+.3f} tau {res[n][1]:+.3f}"
                              for n, _c, _col in SITES))
        print(f"[timing] {state}: {time.perf_counter() - t_wall:.1f} s "
              f"for {len(ANGLES)} solves")
        for name, _c, _col in SITES:
            d[f"{name}_{state}_sig"] = np.array(sig[name])
            d[f"{name}_{state}_tau"] = np.array(tau[name])
    np.savez(cache, **d)
    return d


data = sweep()

# --- Delta CFF per site, from the measured circles --------------------------
#
# The slip-tendency direction is PER ORIENTATION. california.py can take
# sign(median(tau_ambient)) because its probes sit on one fixed fault, so
# that median is a real, well-signed quantity; across a full 180 degree
# gauge sweep the ambient tau traces a whole circle and its median is
# ~zero, so the same expression would pick its sign out of rounding
# noise. Resolve each orientation in the direction that orientation is
# already being sheared.
DCFF = {}
for name, _c, _col in SITES:
    s0, t0 = data[f"{name}_amb_sig"], data[f"{name}_amb_tau"]
    s1, t1 = data[f"{name}_slip_sig"], data[f"{name}_slip_tau"]
    tau_dir = np.sign(t0)
    DCFF[name] = tau_dir * (t1 - t0) + MU_P * (s1 - s0)
    print(f"{name:8s} median dCFF {np.median(DCFF[name]):+.3f}  "
          f"(max {np.max(DCFF[name]):+.3f} at "
          f"theta {ANGLES[np.argmax(DCFF[name])]:.1f}°)")


# --- principal orientations, found by the sweep -----------------------------
#
# At tau = 0 the gauge plane is a principal plane, so its NORMAL is a
# principal direction carrying the measured sigma_n. Reading them off the
# measured sweep keeps the axes measured rather than assumed -- and the
# ambient and post-slip crossings sit at DIFFERENT gauge angles, which is
# how much the earthquake rotated the local stress.

def crossings(sig, tau):
    """[(theta*, sigma_n*, reveal_after)] where tau changes sign."""
    out = []
    for i in range(len(tau) - 1):
        if tau[i] == 0.0 or tau[i] * tau[i + 1] < 0.0:
            w = abs(tau[i]) / (abs(tau[i]) + abs(tau[i + 1]))
            out.append((ANGLES[i] + w * (ANGLES[i + 1] - ANGLES[i]),
                        sig[i] + w * (sig[i + 1] - sig[i]),
                        ANGLES[i + 1]))
    return out


def sigma1_axis(cr):
    """Unit vector along the most-COMPRESSIVE principal direction.

    Of the two crossings, the one with the larger compression-positive
    normal stress is sigma_1; its axis is the gauge NORMAL there.
    """
    if not cr:
        return None, None, None
    th, sig, _rev = max(cr, key=lambda c: -c[1])
    t = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
    return np.array([-t[1], t[0]]), th, -sig


PRINC = {}
for name, _c, _col in SITES:
    for state in ("amb", "slip"):
        cr = crossings(data[f"{name}_{state}_sig"],
                       data[f"{name}_{state}_tau"])
        axis, th, s1 = sigma1_axis(cr)
        PRINC[(name, state)] = dict(cross=cr, axis=axis, theta=th, s1=s1)
    a, b = PRINC[(name, "amb")], PRINC[(name, "slip")]
    if a["axis"] is not None and b["axis"] is not None:
        rot = (b["theta"] - a["theta"] + 90) % 180 - 90
        print(f"{name:8s} sigma1 axis {a['theta']:6.1f}° -> {b['theta']:6.1f}°"
              f"  (rotated {rot:+.1f}°),  sigma1 {a['s1']:+.2f} -> "
              f"{b['s1']:+.2f}")

# --- each site's REAL fault, for context under the gauge --------------------
# These traces are NOT in the solve (a gauge sitting on its own fault
# could not clear it): they are drawn to show where the mapped fault
# actually is, and at which single orientation it samples the sweep.
REAL = {"Garlock": [MINORS["Garlock"]],
        "ECSZ": [MINORS["E1"], MINORS["E2"], MINORS["E3"]],
        "SJF": [MINORS["SJF"]]}
STRIKE = {}
for name, _c, _col in SITES:
    seg0 = REAL[name][len(REAL[name]) // 2]
    d = seg0[1] - seg0[0]
    STRIKE[name] = np.degrees(np.arctan2(d[1], d[0])) % 180.0
    at_strike = np.interp(STRIKE[name], ANGLES, DCFF[name])
    print(f"{name:8s} mapped strike {STRIKE[name]:5.1f}°  ->  "
          f"dCFF at that orientation {at_strike:+.3f}")

# --- the animation ----------------------------------------------------------
XLIM = (-0.1, P0 + 1.3)
YLIM = (-1.45, 1.45)
ss = np.linspace(*XLIM, 80)
strength = np.maximum(COH + MU_P * ss, 0.0)

frames = []
for k in range(len(ANGLES)):
    angle = ANGLES[k]
    # 4:3 with width_ratios ~1.9:1 lets a square map and three square
    # clock faces both fill their cells without a sea of white
    fig = plt.figure(figsize=(12.4, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.9, 1.0])

    # ---- map -------------------------------------------------------------
    axm = fig.add_subplot(gs[:, 0])
    axm.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                                edgecolor="0.75"))
    axm.plot(SAF_PTS[:, 0], SAF_PTS[:, 1], "-", color="black", lw=3.0,
             zorder=3)
    axm.text(0.20, 0.86, "San Andreas", fontsize=10, rotation=-48,
             color="black", ha="center", va="center")
    axm.text(0.30, 0.47, "Transverse Ranges\n(restraining bend)",
             fontsize=8, color="0.35", ha="center", va="center")
    # the plate-boundary drive
    for pm in (+1.0, -1.0):
        base = common.CENTRE + pm * 0.40 * _n - pm * 0.16 * _t
        axm.annotate("", xytext=base, xy=base + pm * 0.30 * _t,
                     arrowprops=dict(arrowstyle="-|>", lw=1.8,
                                     color="#6d4c41", alpha=0.8))
    axm.text(0.06, 0.06, "right-lateral plate-boundary shear", fontsize=8,
             color="#6d4c41")

    for name, c, col in SITES:
        # the MAPPED fault traces, underneath: where the real faults are,
        # and the single orientation each of them actually samples
        for tr in REAL[name]:
            axm.plot(tr[:, 0], tr[:, 1], "-", color=col, lw=2.2, alpha=0.4,
                     zorder=2, solid_capstyle="round")
        # sigma_1 direction, revealed as the sweep discovers it and kept:
        # grey before the earthquake, coloured after
        for state, pc, ls, lw in (("amb", "0.45", "--", 1.2),
                                  ("slip", col, "-", 1.6)):
            pr = PRINC[(name, state)]
            if pr["axis"] is None or angle + 1e-9 < min(x[2] for x in
                                                        pr["cross"]):
                continue
            u = pr["axis"]
            axm.plot([c[0] - 0.085 * u[0], c[0] + 0.085 * u[0]],
                     [c[1] - 0.085 * u[1], c[1] + 0.085 * u[1]],
                     ls, color=pc, lw=lw, alpha=0.9, zorder=3)
        circ = plt.Circle(c, L_GAUGE, fill=False, ec=col, ls=":", lw=0.8,
                          alpha=0.5)
        axm.add_patch(circ)
        seg = _seg(c, angle)
        axm.plot(seg[:, 0], seg[:, 1], "-", color=col, lw=3.4, zorder=4,
                 solid_capstyle="butt")
        dx, dy, ha, va = LAB_OFF[name]
        axm.text(c[0] + dx, c[1] + dy, LABEL[name], fontsize=9,
                 color=col, ha=ha, va=va, zorder=5,
                 bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    axm.text(0.985, 0.955,
             "pale lines: mapped faults\nheavy bar: the rotating gauge\n"
             r"$\sigma_1$ found by the sweep — grey before" "\n"
             "the earthquake, coloured after",
             fontsize=7.5, color="0.3", transform=axm.transAxes,
             ha="right", va="top", linespacing=1.5,
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=2.0), zorder=6)
    seg = _seg(REF_C, REF_ANG)
    axm.plot(seg[:, 0], seg[:, 1], "-", color="0.6", lw=2.0)
    axm.text(REF_C[0], REF_C[1] - 0.045, "pressure reference", fontsize=7,
             color="0.6", ha="center")

    axm.text(0.035, 0.925, rf"gauge orientation $\theta = {angle:.1f}°$",
             fontsize=11, transform=axm.transAxes, va="center",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0),
             zorder=6)
    axm.set_xlim(-0.04, 1.04)
    axm.set_ylim(-0.04, 1.04)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for spine in axm.spines.values():        # the drawn box IS the frame
        spine.set_visible(False)
    axm.text(0.5, 1.005, "Three stress gauges, turning together in one "
             "field  (schematic geometry, not to scale)", fontsize=8.5,
             color="0.35", ha="center", va="bottom",
             transform=axm.transAxes)

    # ---- one clock per site ----------------------------------------------
    for row, (name, _c, col) in enumerate(SITES):
        ax = fig.add_subplot(gs[row, 1])
        for sgn in (+1, -1):
            ax.plot(ss, sgn * strength, "-", color="0.4", lw=0.9)
        ax.fill_between(ss, strength, 2.6, color="#c62828", alpha=0.06, lw=0)
        ax.fill_between(ss, -strength, -2.6, color="#c62828", alpha=0.06,
                        lw=0)
        ax.axhline(0, color="0.92", lw=0.5)

        s0 = P0 - data[f"{name}_amb_sig"][:k + 1]
        t0 = data[f"{name}_amb_tau"][:k + 1]
        s1 = P0 - data[f"{name}_slip_sig"][:k + 1]
        t1 = data[f"{name}_slip_tau"][:k + 1]
        ax.plot(s0, t0, "o", ms=4, mfc="none", mec="0.55", mew=0.9,
                label="ambient")
        ax.plot(s1, t1, "o", ms=4.5, color="#c62828", alpha=0.85,
                label="after slip")

        # the persistent bar: once BOTH principal points have been found,
        # the horizontal diameter joining them IS the principal pair, and
        # it stays on the diagram for the rest of the sweep
        for state, pc, dy in (("amb", "0.45", 0.0), ("slip", "#c62828", 0.0)):
            cr = PRINC[(name, state)]["cross"]
            found = [x for x in cr if angle + 1e-9 >= x[2]]
            if not found:
                continue
            xs = [P0 - x[1] for x in found]
            ax.plot(xs, [dy] * len(xs), "D", ms=5, color=pc, zorder=7)
            if len(found) == 2:
                ax.plot(xs, [dy, dy], "-", color=pc, lw=2.2, alpha=0.9,
                        zorder=6)
        ax.annotate("", xytext=(s0[-1], t0[-1]), xy=(s1[-1], t1[-1]),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="0.3"))
        ax.plot([s0[-1]], [t0[-1]], "o", ms=7, mfc="none", mec="0.2",
                mew=1.3, zorder=6)
        ax.plot([s1[-1]], [t1[-1]], "o", ms=8, color="#c62828",
                mec="0.2", mew=0.8, zorder=6)

        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(col)
            spine.set_linewidth(1.8)
        # the RANGE over orientations, not a single number: whether the
        # event helps or hinders depends on which way the receiver faces
        lo, hi = DCFF[name].min(), DCFF[name].max()
        at_strike = np.interp(STRIKE[name], ANGLES, DCFF[name])
        ax.set_title(f"{LABEL[name]}:  $\\Delta$CFF {lo:+.2f} to {hi:+.2f}"
                     f"   —   at its own strike {at_strike:+.2f}",
                     fontsize=8.5, color=col)
        if row == 0:
            ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
        if row == 2:
            ax.set_xlabel(r"$\sigma$ (compression positive, $P_0=1$)",
                          fontsize=8)
        ax.set_ylabel(r"$\tau$", fontsize=8)

    fig.suptitle("One regional stress — until the San Andreas slips",
                 fontsize=11.5)
    fig.tight_layout()
    frame = os.path.join(D, f"_cal_clock_{k:03d}.png")
    fig.savefig(frame, dpi=100)
    plt.close(fig)
    frames.append(frame)

images = [Image.open(f) for f in frames]
out = os.path.join(D, "california-clocks.gif")
images[0].save(out, save_all=True,
               append_images=images[1:] + [images[-1]] * 8,
               duration=280, loop=0)
print(f"wrote {out} ({len(frames)} frames, "
      f"{os.path.getsize(out) / 1024:.0f} KB)")

# the completed state, for print and PDF
import shutil
shutil.copyfile(frames[-1], os.path.join(D, "california-clocks.png"))
print("wrote", os.path.join(D, "california-clocks.png"))
