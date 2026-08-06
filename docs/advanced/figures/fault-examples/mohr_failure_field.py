"""What failure does to the stress field around it.

The field-level sequel to mohr_friction.py. The same rotating Coulomb
fault, but the left panel is no longer a cartoon: it is the MEASURED
stress field, shown as the change in the local Mohr radius

    d(tau_max) = tau_max(x) - tau_infinity,     tau_max = |sigma'| / 2

with short ticks along the most-compressive principal direction. So the
left panel is a map of "how big is the Mohr circle HERE, and which way
does it point" — the same object the right panel draws for the fault
itself.

The point of the animation: while the fault is STUCK it is invisible —
it transmits the full stress and the field stays uniform. The moment it
SLIDES it drops its shear traction, and the field reorganises: lobes of
raised and lowered tau_max at the tips, principal axes swinging round to
meet the weak surface. Failure is not a dot moving on a diagram; it
rewrites the stress around it.

The fault carries the COHESIVE Mohr-Coulomb law of mohr_cohesion.py
(same mu and C), not bare friction: a purely deviatoric drive puts half
the sweep in tension, where bare friction has no strength at all and
every orientation would be held shut by the no-opening constraint.
Cohesion keeps the fault physical down to sigma = -C/mu, so the sweep
spends most of its time in the regimes worth watching.

The ambient state is uniform and analytic (the drive is a constant
deviatoric stress), so tau_infinity needs no reference solve.

25 solves at h = 0.025, fields cached per angle.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.collections import PolyCollection
from PIL import Image

import underworld3 as uw
from underworld3.utilities import fault_contact

import common

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
A_RATE, GAMMA = 0.5, 1.0
# Cohesive Mohr-Coulomb, as in mohr_cohesion.py. Bare friction has no
# strength at all in tension, so half of a deviatoric sweep lands in the
# unphysical held-shut regime; cohesion keeps the fault physical until
# sigma = -C/mu = -1.0, leaving a short tensile arc instead of half the
# circle. The values match mohr_cohesion.py so the two figures agree.
MU, COH, V0 = 0.6, 0.6, 1e-4
H = 0.025
R_INF = common.ETA * np.sqrt(4 * A_RATE**2 + GAMMA**2)   # ambient tau_max
STEP = 7.5
ANGLES = np.arange(0.0, 180.0 + 1e-9, STEP)

CLIM = 0.6                       # dtau_max colour range
FAULT_C = "#c62828"
SLIDE_C = "#d9960a"


def register_cohesive_law(stokes):
    """Mohr-Coulomb with cohesion as a symbolic law (four lines, per the
    harness design). ``normal_stress`` is the SIGNED effective normal
    stress, reaction-fed and negative in tension; the strength clamp is
    the law's own."""
    V = fault_contact.slip_rate
    S = fault_contact.normal_stress
    law = fault_contact.SymbolicFaultLaw(
        sympy.Max(COH + MU * S, 0) * (2 / sympy.pi) * sympy.atan(V / V0))
    fault_contact.add_frictionless_fault_bc(stokes, "Fault")
    fault_contact._register_law(stokes, "Fault", law)


def stress_cells(child, stokes, tag):
    """P0 (cell) stress components on the split mesh's true connectivity.

    Cellwise averages, never continuous-P1: projecting the rough
    near-fault stress to P1 rings at the node scale (measured residual
    rms 0.26 at half-wavelength h/2).
    """
    x, y = child.X
    v = stokes.Unknowns.u
    p = stokes.Unknowns.p
    exprs = dict(sxx=-p.sym[0] + 2 * common.ETA * v.sym[0].diff(x),
                 syy=-p.sym[0] + 2 * common.ETA * v.sym[1].diff(y),
                 sxy=common.ETA * (v.sym[0].diff(y) + v.sym[1].diff(x)))
    out = {}
    for name, expr in exprs.items():
        s_var = uw.discretisation.MeshVariable(f"{name}_{tag}", child, 1,
                                               degree=0, continuous=False)
        proj = uw.systems.Projection(child, s_var)
        proj.uw_function = expr
        proj.smoothing = 0.0
        proj.solve()
        row = common.split_mesh_cell_rows(child, s_var)
        out[name] = np.asarray(s_var.data[:, 0])[row].copy()
    return out


def sweep():
    cache = os.path.join(D, "_mohr_failure_field.npz")
    if os.path.exists(cache):
        d = dict(np.load(cache, allow_pickle=True))
        assert len(d["probes"]) == len(ANGLES), "cache is stale — delete it"
        print(f"loaded cached sweep ({len(ANGLES)} angles)")
        return d
    rows, store = [], {}
    t_wall = time.perf_counter()
    for k, theta in enumerate(ANGLES):
        child = common.split_with_fault(
            common.base_mesh(H), common.fault_segment(theta, HALF))
        stokes = common.stokes_on(
            child, common.shear_plus_stretch(child, A_RATE, GAMMA))
        register_cohesive_law(stokes)
        fault_contact.solve_with_fault(stokes, picard=3)

        s, V, _leak = common.slip_profile(stokes)
        s_n, sig = common.normal_traction(stokes)
        v_med = float(np.median(V[common.inner(s)]))
        sigma_n = float(np.median(sig[common.inner(s_n)]))
        # the law's own strength clamp, evaluated at the measured state
        strength = max(COH + MU * (-sigma_n), 0.0)
        tau = strength * (2 / np.pi) * np.arctan(v_med / V0)
        rows.append((theta, sigma_n, tau, v_med))

        comp = stress_cells(child, stokes, f"f{k}")
        pts, faces = common.split_mesh_cell_render(child)
        store[f"pts_{k}"] = np.asarray(pts, dtype=float)
        store[f"fac_{k}"] = np.asarray(faces, dtype=np.int64)
        for nm in ("sxx", "syy", "sxy"):
            store[f"{nm}_{k}"] = comp[nm]
        print(f"theta {theta:6.1f}: sigma_n {sigma_n:8.4f}  tau {tau:8.4f}  "
              f"V {v_med:9.5f}  cells {len(comp['sxx'])}")
    print(f"[timing] {len(ANGLES)} solves + projections: "
          f"{time.perf_counter() - t_wall:.1f} s")
    store["probes"] = np.array(rows)
    np.savez_compressed(cache, **store)
    print(f"cache {os.path.getsize(cache) / 1e6:.1f} MB")
    return dict(np.load(cache, allow_pickle=True))


data = sweep()
probes = data["probes"]
scg = -probes[:, 1]                      # compression positive
S_ZERO = -COH / MU                       # strength vanishes here
held_shut = scg < S_ZERO
sliding = (np.abs(probes[:, 3]) > 5 * V0) & ~held_shut
print(f"stuck {int((~sliding & ~held_shut).sum())}, "
      f"sliding {int(sliding.sum())}, held-shut {int(held_shut.sum())} "
      f"of {len(probes)} orientations")


def draw_stress_plane(ax):
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_INF * np.cos(tt), R_INF * np.sin(tt), "-", color="0.8",
            lw=0.9, label="ambient circle")
    ss = np.linspace(S_ZERO, 1.5 * R_INF, 80)
    strength = np.maximum(COH + MU * ss, 0.0)
    for sgn in (+1, -1):
        ax.plot(ss, sgn * strength, "--", color="0.35", lw=1.0,
                label=(r"envelope $\tau = \pm(C + \mu\sigma)$" if sgn > 0
                       else None))
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)
    ax.axvspan(-1.6 * R_INF, S_ZERO, color="0.92", zorder=0)
    ax.text(0.5 * (-1.25 * R_INF + S_ZERO), -0.55 * R_INF,
            "no strength\nleft here", fontsize=7, ha="center", va="center",
            color="0.45", rotation=90)
    ax.set_xlabel(r"normal stress $\sigma$ (compression positive)")
    ax.set_ylabel(r"shear traction $\tau$")
    ax.set_xlim(-1.25 * R_INF, 1.25 * R_INF)
    ax.set_ylim(-1.35 * R_INF, 1.35 * R_INF)
    ax.set_aspect("equal")


frames = []
for k in range(len(ANGLES)):
    theta, sig_k, tau_k, v_k = probes[k]
    shut_k = bool(held_shut[k])
    slide_k = bool(sliding[k])

    pts = data[f"pts_{k}"][:, :2]
    tri = data[f"fac_{k}"].reshape(-1, 4)[:, 1:]
    sxx, syy, sxy = (data[f"sxx_{k}"], data[f"syy_{k}"], data[f"sxy_{k}"])
    half_diff = 0.5 * (sxx - syy)
    tau_max = np.sqrt(half_diff**2 + sxy**2)
    dtau = tau_max - R_INF

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(10.6, 4.8),
        gridspec_kw=dict(width_ratios=[1.12, 1.15]))

    # ---- left: the field the fault is making -----------------------------
    pc = PolyCollection(pts[tri], array=dtau, cmap="RdBu_r",
                        clim=(-CLIM, CLIM), edgecolors="none", lw=0)
    axl.add_collection(pc)

    # principal-compression ticks on a coarse grid, away from the tips
    cent = pts[tri].mean(axis=1)
    ang_t = 0.5 * np.arctan2(2 * sxy, sxx - syy)      # most TENSILE dir
    ang_c = ang_t + np.pi / 2                          # most compressive
    t_hat = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    # Mask only the two TIPS, where the crack singularity makes the
    # principal direction meaningless. The near-fault region is exactly
    # where the axes swing, so it must stay in.
    rel = cent - common.CENTRE
    tips = [common.CENTRE + s * HALF * t_hat for s in (+1.0, -1.0)]
    near = np.zeros(len(cent), dtype=bool)
    for tip in tips:
        near |= np.hypot(cent[:, 0] - tip[0], cent[:, 1] - tip[1]) < 0.045
    g = np.linspace(0.05, 0.95, 15)
    picked = []
    for gx in g:
        for gy in g:
            d = np.hypot(cent[:, 0] - gx, cent[:, 1] - gy)
            d[near] = np.inf
            j = int(np.argmin(d))
            if d[j] < 0.04:
                picked.append(j)
    picked = np.unique(picked)
    seg = 0.024
    for j in picked:
        u = np.array([np.cos(ang_c[j]), np.sin(ang_c[j])])
        axl.plot([cent[j, 0] - seg * u[0], cent[j, 0] + seg * u[0]],
                 [cent[j, 1] - seg * u[1], cent[j, 1] + seg * u[1]],
                 "-", color="0.25", lw=1.0, alpha=0.8, zorder=3)

    c = common.CENTRE
    n = np.array([-t_hat[1], t_hat[0]])
    axl.plot([c[0] - HALF * t_hat[0], c[0] + HALF * t_hat[0]],
             [c[1] - HALF * t_hat[1], c[1] + HALF * t_hat[1]],
             "-", color=FAULT_C, lw=3.0, zorder=5, solid_capstyle="butt")
    if slide_k:
        off = 0.030 * n
        sgn = np.sign(v_k)
        for pm in (+1, -1):
            axl.annotate("", xytext=c + pm * off - pm * sgn * 0.085 * t_hat,
                         xy=c + pm * off + pm * sgn * 0.085 * t_hat,
                         arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                         color=SLIDE_C), zorder=6)
    status, scol = (("HELD SHUT (unphysical)", "0.35") if shut_k
                    else ("SLIDING", SLIDE_C) if slide_k
                    else ("stuck — the field never notices", FAULT_C))
    bbox = dict(fc="white", ec="none", alpha=0.82, pad=1.6)
    axl.text(0.03, 0.965, rf"$\theta = {theta:.1f}°$", fontsize=12,
             transform=axl.transAxes, va="top", bbox=bbox, zorder=8)
    axl.text(0.03, 0.885, status, fontsize=10.5, color=scol,
             transform=axl.transAxes, va="top", bbox=bbox, zorder=8)
    axl.text(0.03, 0.025, "ticks: most-compressive principal direction",
             fontsize=7.5, color="0.25", transform=axl.transAxes,
             bbox=bbox, zorder=8)
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.set_aspect("equal")
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_title(r"change in local Mohr radius $\Delta\tau_{max}$",
                  fontsize=10)

    # ---- right: the fault's own probe ------------------------------------
    draw_stress_plane(axr)
    axr.legend(fontsize=7, loc="upper left")
    stuck_prev = (~sliding & ~held_shut)[:k + 1]
    axr.plot(scg[:k + 1][stuck_prev], probes[:k + 1][stuck_prev, 2], "o",
             ms=5, mfc="none", mec=FAULT_C, mew=1.2)
    axr.plot(scg[:k + 1][sliding[:k + 1]], probes[:k + 1][sliding[:k + 1], 2],
             "s", ms=5, mfc="none", mec=SLIDE_C, mew=1.2)
    axr.plot(scg[:k + 1][held_shut[:k + 1]],
             probes[:k + 1][held_shut[:k + 1], 2], "x", ms=6, mew=1.6,
             color="0.45")
    mark, mcol = (("x", "0.45") if shut_k else
                  ("s", SLIDE_C) if slide_k else ("o", FAULT_C))
    axr.plot([-sig_k], [tau_k], mark, ms=10, mew=2.4, color=mcol, zorder=6)
    axr.set_title("the fault's traction, against its strength", fontsize=10)

    cb = fig.colorbar(pc, ax=axl, fraction=0.046, pad=0.02)
    cb.set_label(r"$\Delta\tau_{max}$", fontsize=9)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("When the fault slips, the stress field rearranges",
                 fontsize=11.5)
    fig.tight_layout()
    frame = os.path.join(D, f"_mff_frame_{k:03d}.png")
    fig.savefig(frame, dpi=90)
    plt.close(fig)
    frames.append(frame)

# A continuous field render carries far more colours than the line plots
# of the other animations, so quantise to a shared adaptive palette --
# otherwise the GIF lands near 1 MB, well over the docs sizing.
images = [Image.open(f).convert("RGB") for f in frames]
# NB dithering must be OFF: it scatters pixel noise that defeats GIF's
# run-length compression and made the file three times LARGER.
pal = images[len(images) // 2].quantize(colors=64, method=Image.MEDIANCUT)
images = [im.quantize(palette=pal, dither=Image.NONE) for im in images]
out = os.path.join(D, "mohr-failure-field.gif")
images[0].save(out, save_all=True,
               append_images=images[1:] + [images[-1]] * 6,
               duration=280, loop=0, optimize=True)
print(f"wrote {out} ({len(frames)} frames, "
      f"{os.path.getsize(out) / 1024:.0f} KB)")

# the sliding state, for print and PDF
k_still = int(np.argmax(np.where(sliding, np.abs(probes[:, 3]), -1.0)))
import shutil
shutil.copyfile(frames[k_still], os.path.join(D, "mohr-failure-field.png"))
print(f"wrote mohr-failure-field.png (theta {ANGLES[k_still]:.1f}°, sliding)")
