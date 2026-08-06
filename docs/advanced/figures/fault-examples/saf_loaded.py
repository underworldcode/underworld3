"""The reverse experiment: what the neighbours do to the San Andreas.

california.py and california_clocks.py let the master fault slip and
read its neighbours. This turns the question round: let ONE neighbour
slip -- the San Jacinto, then the Garlock -- and measure what it does
along the length of the San Andreas.

The San Andreas is WELDED in every state here, so it is a receiver, not
a source. That is what makes the measurement cheap: a welded fault is a
per-node stress probe along its whole length, so one solve per source
event gives Delta CFF everywhere along the trace, with no gauge sweep at
all.

The trace is curved, so the shear traction must be resolved on the LOCAL
tangent at each node, not on the mean trend: fault_pair_jumps returns the
per-node normals (the analytic ones, since the fault carries
add_fault_bc(normal=...)), and the tangent is their quarter turn.
common.probe_nodes takes a single global tangent and is right for the
straight neighbours, not for this.

Gauge discipline as everywhere else: a closed velocity-driven box fixes
pressure only to a per-solve constant, so the far end of the trace --
which a local event cannot reach -- anchors each solve.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import underworld3 as uw
from underworld3.utilities import fault_contact

import common
from california import (SAF_PTS, MINORS, TREND, TAU0, MU_P, ETA_WELD,
                        saf_normal, _A, _t, _n)

D = os.path.dirname(os.path.abspath(__file__))
H = 0.012

# the source events, in the order they appear on the figure
SOURCES = [("SJF", "San Jacinto", "#e65100"),
           ("Garlock", "Garlock", "#6a1b9a")]


def saf_probe(stokes):
    """(s, xy, sigma_n, tau) along the San Andreas, on LOCAL tangents.

    sigma_n is the no-opening reaction (tension positive, as measured);
    tau is the weld's own law, eta_f V, resolved on each node's own
    tangent so the bend is handled honestly.
    """
    from underworld3.utilities.rotated_bc import _point_coord

    info = stokes._rotated_freeslip_info
    coords, jumps, normals = fault_contact.fault_pair_jumps(stokes, "SAF",
                                                            info)
    nrm = np.asarray(normals, dtype=float)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    tng = np.column_stack([-nrm[:, 1], nrm[:, 0]])
    # orient every tangent the same way along the trace
    flip = (tng @ _t) < 0
    tng[flip] *= -1.0
    tau = ETA_WELD * np.einsum("ij,ij->i", jumps, tng)

    assembler = fault_contact._InterfaceAssembler(stokes, include=("SAF",))
    sig_all = assembler.nodal_normal_traction(stokes, info["reaction"])
    plus = set(stokes.mesh._fault_point_pairs["SAF"].values())
    dm = stokes.dm
    dim = stokes.mesh.dim
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    rows = [(np.asarray(_point_coord(dm, dim, cvec, csec, v0, v1, q)),
             sig_all[k]) for q, k in assembler._points.items() if q in plus]
    xy_sig = np.array([r[0] for r in rows])
    sig = np.array([r[1] for r in rows])

    s_sig = (xy_sig - _A) @ _t
    o = np.argsort(s_sig)
    xy_sig, sig = xy_sig[o], sig[o]
    s = (coords - _A) @ _t
    o2 = np.argsort(s)
    assert np.allclose(coords[o2], xy_sig), "node ordering disagrees"
    return s[o2], coords[o2], sig, tau[o2]


def solve(free=None):
    """One solve; `free` names the neighbour allowed to slip (None = all
    welded, the ambient state). The San Andreas is welded throughout."""
    faults = [("SAF", SAF_PTS)] + [(k, v) for k, v in MINORS.items()]
    child = common.base_mesh(H).add_fault(faults)
    stokes = common.stokes_on(
        child, common.boundary_simple_shear(child, TREND, TAU0))
    stokes.add_fault_bc(ETA_WELD, boundary="SAF", normal=saf_normal(child))
    for k in MINORS:
        stokes.add_fault_bc(0 if k == free else ETA_WELD, boundary=k)
    fault_contact.solve_with_fault(stokes, picard=2)
    out = saf_probe(stokes)
    v_med = 0.0
    if free is not None:
        _sj, Vj = common.slip_vs_position(
            stokes, MINORS[free][1] - MINORS[free][0],
            centre=MINORS[free].mean(axis=0), name=free)
        v_med = float(np.median(Vj))
        sense = "dextral" if v_med > 0 else "sinistral"
        print(f"    {free} slip: median jump {v_med:+.3f} ({sense} "
              f"along its own tangent)")
    return out + (v_med,)


cache = os.path.join(D, "_saf_loaded.npz")
if os.path.exists(cache):
    data = dict(np.load(cache))
    print("loaded cached run")
else:
    t_wall = time.perf_counter()
    store = {}
    print("  ambient (everything welded)")
    s, xy, sig0, tau0, _v = solve(None)
    store["s"], store["xy"] = s, xy
    store["sig_amb"], store["tau_amb"] = sig0, tau0
    for key, _lab, _col in SOURCES:
        print(f"  source: {key}")
        s1, xy1, sig1, tau1, v1 = solve(key)
        assert np.allclose(xy1, xy), "node set moved between solves"
        store[f"sig_{key}"], store[f"tau_{key}"] = sig1, tau1
        store[f"slip_{key}"] = np.array([v1])
    print(f"[timing] {1 + len(SOURCES)} solves: "
          f"{time.perf_counter() - t_wall:.1f} s")
    np.savez(cache, **store)
    data = dict(np.load(cache))

s = data["s"]
xy = data["xy"]
sig0, tau0 = data["sig_amb"], data["tau_amb"]

# Delta CFF along the trace, per source event. The far end of the trace
# is the pressure gauge: a local event cannot reach it, so whatever
# constant appears there is the solve's, not the physics.
DC = {}
for key, lab, col in SOURCES:
    sig1, tau1 = data[f"sig_{key}"], data[f"tau_{key}"]
    src = MINORS[key].mean(axis=0)
    far = np.hypot(xy[:, 0] - src[0], xy[:, 1] - src[1]) > 0.55
    if far.sum() < 5:
        far = np.hypot(xy[:, 0] - src[0], xy[:, 1] - src[1]) > 0.45
    c_gauge = float(np.median((sig1 - sig0)[far]))
    dc = (np.sign(tau0) * (tau1 - tau0)
          + MU_P * (sig1 - sig0 - c_gauge))
    DC[key] = dc
    j = int(np.argmax(np.abs(dc)))
    print(f"{key:8s} gauge {c_gauge:+8.4f} (from {int(far.sum())} far nodes)"
          f"   dCFF range {dc.min():+.3f} .. {dc.max():+.3f}"
          f"   |max| {dc[j]:+.3f} at s = {s[j]:.3f}, xy = "
          f"({xy[j, 0]:.2f}, {xy[j, 1]:.2f})")

# --- the figure -------------------------------------------------------------
from matplotlib.collections import LineCollection

CLIM = 0.4
BEND_S = 0.45                      # the Big Bend, in the arc parameter

fig = plt.figure(figsize=(13.6, 4.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35])

for col_i, (key, lab, col) in enumerate(SOURCES):
    ax = fig.add_subplot(gs[0, col_i])
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                               edgecolor="0.8"))
    # every neighbour, pale; the one that slipped, solid
    for k2, pts in MINORS.items():
        is_src = (k2 == key)
        ax.plot(pts[:, 0], pts[:, 1], "-",
                color=col if is_src else "0.72",
                lw=3.4 if is_src else 1.8,
                zorder=4 if is_src else 2, solid_capstyle="round")
    # the San Andreas, coloured by what the event did to it
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    val = 0.5 * (DC[key][:-1] + DC[key][1:])
    lc = LineCollection(seg, array=val, cmap="RdBu_r", clim=(-CLIM, CLIM),
                        linewidths=5.0, zorder=3, capstyle="round")
    ax.add_collection(lc)

    # measured slip sense on the source
    v_med = float(data[f"slip_{key}"][0])
    src_pts = MINORS[key]
    tt = src_pts[1] - src_pts[0]
    tt = tt / np.linalg.norm(tt)
    nn = np.array([-tt[1], tt[0]])
    mid = src_pts.mean(axis=0)
    sgn = np.sign(v_med)
    for pm in (+1.0, -1.0):
        base = mid + pm * 0.028 * nn - pm * sgn * 0.055 * tt
        ax.annotate("", xytext=base, xy=base + pm * sgn * 0.11 * tt,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=col),
                    zorder=5)
    # label on the far side of the source from the trace, so it cannot
    # collide with the marker on the San Andreas
    away = nn if np.dot(nn, mid - xy[np.argmin(
        np.hypot(xy[:, 0] - mid[0], xy[:, 1] - mid[1]))]) > 0 else -nn
    lp = mid + 0.105 * away
    ax.text(lp[0], lp[1], f"{lab} slips", fontsize=9.5, color=col,
            ha="center", va="center", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    # the strongest effect, not the median: most of the trace is far from
    # the source and barely moves, so a median over the whole fault says
    # "nothing happened" for both events -- the same trap as the medians
    # in california_clocks.py
    j = int(np.argmax(np.abs(DC[key])))
    ax.plot([xy[j, 0]], [xy[j, 1]], "o", ms=10, mfc="none", mec="black",
            mew=1.5, zorder=7)
    lab_xy = xy[j] + np.array([-0.16, -0.10])
    ax.annotate(f"{DC[key][j]:+.2f}", xy=xy[j], xytext=lab_xy,
                fontsize=10, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="0.3"),
                bbox=dict(fc="white", ec="0.7", alpha=0.95, pad=2.0),
                zorder=8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    verb = "loads" if DC[key][j] > 0 else "relaxes"
    ax.set_title(f"{lab} slips\n→ {verb} the bend", fontsize=10.5,
                 color=col)
    if col_i == 1:
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"$\Delta$CFF on the San Andreas", fontsize=9)
        cb.ax.tick_params(labelsize=8)

axp = fig.add_subplot(gs[0, 2])
axp.axhline(0, color="0.6", lw=0.8)
axp.axvline(BEND_S, color="0.85", lw=6.0, zorder=0)
axp.text(BEND_S, 0.42, "Big Bend", fontsize=8.5, rotation=90, color="0.45",
         ha="right", va="top")
for key, lab, col in SOURCES:
    axp.plot(s, DC[key], "-", color=col, lw=2.0, label=f"{lab} slips")
axp.set_xlabel("distance along the San Andreas  (SE $\\rightarrow$ NW)")
axp.set_ylabel(r"$\Delta$CFF")
axp.set_title("Both events focus on the bend — with opposite sign",
              fontsize=10)
axp.legend(fontsize=8.5, loc="lower right")
axp.set_xlim(s.min(), s.max())

fig.suptitle("The reverse experiment: what the neighbours do to the "
             "San Andreas", fontsize=12.5)
fig.tight_layout()
out = os.path.join(D, "saf-loaded.png")
fig.savefig(out, dpi=200)
print("wrote", out)

# --- the animation: wind the stress drop up from zero -----------------------
#
# Delta CFF is LINEAR in the stress the source drops (the mechanics are
# linear and the slipping fault is completely weak during its event), so
# a partial drop is exactly a scaled full drop. Every frame below is
# therefore an exact solution, not an interpolation between two states --
# and none of it costs another solve.
from PIL import Image

FRAC = np.linspace(0.0, 1.0, 21)
frames = []
for fi, f in enumerate(FRAC):
    fig = plt.figure(figsize=(10.6, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3])
    for row, (key, lab, col) in enumerate(SOURCES):
        axm = fig.add_subplot(gs[row, 0])
        axm.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                                    edgecolor="0.85"))
        for k2, pts in MINORS.items():
            is_src = (k2 == key)
            axm.plot(pts[:, 0], pts[:, 1], "-",
                     color=col if is_src else "0.75",
                     lw=3.4 if is_src else 1.6,
                     zorder=4 if is_src else 2, solid_capstyle="round")
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        val = f * 0.5 * (DC[key][:-1] + DC[key][1:])
        lc = LineCollection(seg, array=val, cmap="RdBu_r",
                            clim=(-CLIM, CLIM), linewidths=5.0, zorder=3,
                            capstyle="round")
        axm.add_collection(lc)

        v_med = float(data[f"slip_{key}"][0])
        src_pts = MINORS[key]
        tt = src_pts[1] - src_pts[0]
        tt = tt / np.linalg.norm(tt)
        nn = np.array([-tt[1], tt[0]])
        mid = src_pts.mean(axis=0)
        sgn = np.sign(v_med)
        for pm in (+1.0, -1.0):          # arrows grow with the event
            base = mid + pm * 0.028 * nn
            tip = base + pm * sgn * (0.02 + 0.10 * f) * tt
            axm.annotate("", xytext=base, xy=tip,
                         arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                         color=col, alpha=0.35 + 0.65 * f),
                         zorder=5)
        axm.text(mid[0], mid[1] + 0.105, f"{lab} slips", fontsize=9.5,
                 color=col, ha="center", va="center", zorder=6,
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        axm.set_xlim(-0.03, 1.03)
        axm.set_ylim(-0.03, 1.03)
        axm.set_aspect("equal")
        axm.set_xticks([])
        axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(lc, ax=axm, fraction=0.046, pad=0.02)
        cb.set_label(r"$\Delta$CFF", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        axp = fig.add_subplot(gs[row, 1])
        axp.axhline(0, color="0.6", lw=0.8)
        axp.axvline(BEND_S, color="0.88", lw=6.0, zorder=0)
        axp.text(BEND_S, 0.43, "Big Bend", fontsize=8, rotation=90,
                 color="0.45", ha="right", va="top")
        axp.plot(s, f * DC[key], "-", color=col, lw=2.2)
        j = int(np.argmax(np.abs(DC[key])))
        axp.plot([s[j]], [f * DC[key][j]], "o", ms=7, color=col, zorder=5)
        axp.annotate(f"{f * DC[key][j]:+.2f}", xy=(s[j], f * DC[key][j]),
                     xytext=(11, -4 if DC[key][j] < 0 else 4),
                     textcoords="offset points", fontsize=9.5, color=col,
                     va="top" if DC[key][j] < 0 else "bottom")
        axp.set_ylim(-0.34, 0.44)       # fixed: both grow on one scale
        axp.set_xlim(s.min(), s.max())
        axp.set_ylabel(r"$\Delta$CFF", fontsize=9)
        axp.tick_params(labelsize=8)
        if row == 1:
            axp.set_xlabel("distance along the San Andreas  "
                           "(SE $\\rightarrow$ NW)", fontsize=9)
        axp.set_title(f"{lab} slips", fontsize=9.5, color=col)

    fig.suptitle("The neighbours load the San Andreas   —   "
                 f"stress drop on the source: {100 * f:3.0f}%",
                 fontsize=12)
    fig.tight_layout()
    frame = os.path.join(D, f"_safl_frame_{fi:03d}.png")
    fig.savefig(frame, dpi=100)
    plt.close(fig)
    frames.append(frame)

images = [Image.open(fp).convert("RGB") for fp in frames]
pal = images[-1].quantize(colors=96, method=Image.MEDIANCUT)
images = [im.quantize(palette=pal, dither=Image.NONE) for im in images]
out = os.path.join(D, "saf-loaded.gif")
images[0].save(out, save_all=True,
               append_images=images[1:] + [images[-1]] * 10,
               duration=160, loop=0, optimize=True)
print(f"wrote {out} ({len(frames)} frames, "
      f"{os.path.getsize(out) / 1024:.0f} KB)")
