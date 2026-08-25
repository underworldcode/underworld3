"""Stress trajectories in the schematic southern California model.

A *stress trajectory* is a curve everywhere tangent to one principal
stress direction. Two orthogonal families fill the plane, and away from
any structure they are the straight, parallel lines of the regional
field. Near a fault that is slipping they are not: the fault cannot
carry shear, so the principal directions must meet it at 45 degrees,
and the trajectories refract into the restraining bend.

This is the field-scale reading of the principal directions the Mohr
figures introduce at a point, on the geometry the CFF figures already
use, so the students meet it on a map they have seen before.

Geometry MIRRORS california.py deliberately (same trace, same
neighbours, same drive) so the two figures are the same experiment.
Kept as a copy rather than an import because california.py builds its
figures at module level.

Method notes, both of which matter:

* The principal direction is a DIRECTOR, not a vector: theta and
  theta+180 are the same state. Averaging or interpolating theta wraps
  catastrophically at the branch cut, so everything here is carried as
  the double-angle vector (cos 2t, sin 2t), which is single valued, and
  halved only at the point of use.
* Integration picks, at each step, the branch continuing the previous
  step (sign continuity). Without it a trajectory flips 90 degrees
  whenever the eigenvector solver reorders its output.

Trajectories are stopped before they reach a fault: the field is
genuinely discontinuous across a slipping surface, so drawing through
it would be a lie.

underworld3 gained a general version of this after these figures were
made -- visualisation/glyphs.py: direction_trajectories (evenly spaced,
Jobard & Lehmann placement) and principal_stress_glyphs. It is on
feature/stress-glyphs and NOT yet merged to main, so this script keeps
its own integrator; the colours and terminology here follow it, and
this should be switched over once that branch lands.

Run:  python california_trajectories.py     (~1 solve; then cached)
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.interpolate import griddata

import underworld3 as uw
from underworld3.utilities import fault_contact

import common

D = os.path.dirname(os.path.abspath(__file__))
MU_P = 0.4
TAU0 = 1.0
TREND = 132.0
ETA_WELD = 200.0 * common.ETA / 0.2

_A = np.array([0.88, 0.06])
_t = np.array([np.cos(np.radians(TREND)), np.sin(np.radians(TREND))])
_n = np.array([-_t[1], _t[0]])
S_END = 0.94
BEND_W = 0.07
BEND_S0 = 0.45
LAM = 0.06


def _w(s):
    return 0.5 * BEND_W * (1.0 + np.tanh((s - BEND_S0) / LAM))


def saf_trace(n_seg=47):
    s = np.linspace(0.0, S_END, n_seg + 1)
    return _A + np.outer(s, _t) + np.outer(_w(s), _n)


def saf_normal(child):
    x, y = child.X
    s = (x - _A[0]) * _t[0] + (y - _A[1]) * _t[1]
    wp = (0.5 * BEND_W / LAM) * (1 - sympy.tanh((s - BEND_S0) / LAM) ** 2)
    return sympy.Matrix([[_n[0] - wp * _t[0], _n[1] - wp * _t[1]]])


SAF_PTS = saf_trace()
MINORS = {
    "Garlock": np.array([[0.52, 0.44], [0.80, 0.53]]),
    "E1": np.array([[0.62, 0.64], [0.56, 0.78]]),
    "E2": np.array([[0.72, 0.62], [0.66, 0.76]]),
    "E3": np.array([[0.81, 0.68], [0.75, 0.82]]),
    "SJF": np.array([[0.84, 0.24], [0.70, 0.38]]),
}
COLOUR = {"Garlock": "#6a1b9a", "E1": "#1a6b1a", "E2": "#1a6b1a",
          "E3": "#1a6b1a", "SJF": "#e65100"}

C_COMP = "#2166ac"    # most-compressive direction  (blue)
C_EXT = "#b2182b"     # most-extensional direction (red)
# Colours follow underworld3 visualisation/glyphs.py: "blue compressive,
# red tensile, matching the RdBu_r field convention".


# --------------------------------------------------------------- solve
def solve_slipping():
    faults = [("SAF", SAF_PTS)] + [(k, v) for k, v in MINORS.items()]
    child = common.base_mesh(0.012).add_fault(faults)
    stokes = common.stokes_on(
        child, common.boundary_simple_shear(child, TREND, TAU0))
    stokes.add_fault_bc(0, boundary="SAF", normal=saf_normal(child))
    for k in MINORS:
        stokes.add_fault_bc(ETA_WELD, boundary=k)
    fault_contact.solve_with_fault(stokes, picard=2)
    return child, stokes


def stress_cells(child, stokes):
    """P0 (cell) stress on the split mesh's true connectivity — the same
    discipline california.py uses; a continuous-P1 projection would ring
    at the node scale near the fault."""
    x, y = child.X
    v, p = stokes.Unknowns.u, stokes.Unknowns.p
    exprs = dict(
        sxx=-p.sym[0] + 2 * common.ETA * v.sym[0].diff(x),
        syy=-p.sym[0] + 2 * common.ETA * v.sym[1].diff(y),
        sxy=common.ETA * (v.sym[0].diff(y) + v.sym[1].diff(x)))
    out = {}
    for name, expr in exprs.items():
        s_var = uw.discretisation.MeshVariable(f"{name}_tr", child, 1,
                                               degree=0, continuous=False)
        proj = uw.systems.Projection(child, s_var)
        proj.uw_function = expr
        proj.smoothing = 0.0
        proj.solve()
        row = common.split_mesh_cell_rows(child, s_var)
        out[name] = np.asarray(s_var.data[:, 0])[row].copy()
    return out


cache = os.path.join(D, "_california_traj.npz")
if os.path.exists(cache):
    data = dict(np.load(cache))
    print("loaded cached run")
else:
    t0 = time.perf_counter()
    child, stokes = solve_slipping()
    comp = stress_cells(child, stokes)
    pts, faces = common.split_mesh_cell_render(child)
    fc = np.asarray(faces).reshape(-1, 4)[:, 1:]
    cent = np.asarray(pts)[fc].mean(axis=1)
    data = dict(cent=cent, **comp)
    np.savez(cache, **data)
    print(f"[timing] slipping solve + stress projections: "
          f"{time.perf_counter() - t0:.1f} s")

cent = data["cent"][:, :2]
sxx, syy, sxy = data["sxx"], data["syy"], data["sxy"]

# Solver convention is tension-positive; the geological convention on
# every other figure in this set is compression-positive, so flip. The
# principal DIRECTIONS are unchanged by the flip (it negates the whole
# tensor) — sigma_1 and sigma_3 simply swap names, which is exactly why
# the labels have to be applied after the flip, not before.
sxx, syy, sxy = -sxx, -syy, -sxy

# double angle of the most-compressive direction, per cell
two_theta = np.arctan2(2.0 * sxy, sxx - syy)
c2, s2 = np.cos(two_theta), np.sin(two_theta)
diff_stress = 2.0 * np.hypot(0.5 * (sxx - syy), sxy)   # sigma_1 - sigma_3

# ------------------------------------------------------- director grid
NG = 320
gx = np.linspace(0.02, 0.98, NG)
gy = np.linspace(0.02, 0.98, NG)
GX, GY = np.meshgrid(gx, gy)
grid = np.column_stack([GX.ravel(), GY.ravel()])
C2 = griddata(cent, c2, grid, method="linear").reshape(NG, NG)
S2 = griddata(cent, s2, grid, method="linear").reshape(NG, NG)
TAU = griddata(cent, diff_stress, grid, method="linear").reshape(NG, NG)


from scipy.ndimage import gaussian_filter

# The P0 stress is piecewise constant, so the raw director field is
# bumpy at the cell scale and trajectories integrated through it wander
# and cross — which trajectories of one family never do. Smooth the
# DOUBLE-ANGLE vector (legitimate: it is a genuine vector, unlike the
# angle) and renormalise.
C2 = gaussian_filter(np.nan_to_num(C2, nan=0.0), 3.0)
S2 = gaussian_filter(np.nan_to_num(S2, nan=0.0), 3.0)
nrm = np.hypot(C2, S2)
C2, S2 = C2 / np.maximum(nrm, 1e-12), S2 / np.maximum(nrm, 1e-12)
TAU = gaussian_filter(np.nan_to_num(TAU, nan=0.0), 3.0)

THETA = 0.5 * np.arctan2(S2, C2)          # most-compressive direction

# Distance to the nearest fault, on the grid, so lines can be stopped
# before they cross a surface the field is discontinuous across.
# ONLY the San Andreas slips; the neighbours are welded probes, so the
# field is continuous across them and blanking them out would claim a
# discontinuity that is not there.
SEGS = [SAF_PTS]


def dist_field(GX, GY):
    d = np.full(GX.shape, 1e9)
    P = np.column_stack([GX.ravel(), GY.ravel()])
    for seg in SEGS:
        a, b = seg[:-1], seg[1:]
        ab = b - a
        for k in range(len(a)):
            t = np.clip(((P - a[k]) @ ab[k]) / (ab[k] @ ab[k]), 0, 1)
            q = a[k] + t[:, None] * ab[k]
            d = np.minimum(d, np.hypot(*(P - q).T).reshape(GX.shape))
    return d


DIST = dist_field(GX, GY)
# The blank band is NOT a damage zone: it is the smoothing length. The
# double-angle field is Gaussian-smoothed over ~3 grid cells, which
# blurs across the fault, so nothing is drawn within ~2 sigma of the
# one surface that is actually slipping.
KEEPOUT = 0.016
MASK = DIST < KEEPOUT

# A director field has no global sign, which is why streamplot cannot be
# used on it directly. Here the field is a perturbation of uniform simple
# shear, so every direction can be put in the half-plane nearest the
# far-field one — that makes it a true vector field, and lets matplotlib
# do the even-spacing work it is good at.
REF1 = np.array([np.cos(np.radians(TREND + 45.0)),
                 np.sin(np.radians(TREND + 45.0))])


def vector_field(theta, ref):
    vx, vy = np.cos(theta), np.sin(theta)
    flip = (vx * ref[0] + vy * ref[1]) < 0
    vx, vy = np.where(flip, -vx, vx), np.where(flip, -vy, vy)
    return (np.ma.array(vx, mask=MASK), np.ma.array(vy, mask=MASK))


# ------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.6, 8.8))
ax.set_aspect("equal")
ax.set_xlim(0.03, 0.97)
ax.set_ylim(0.03, 0.97)
ax.set_xticks([])
ax.set_yticks([])

ax.contourf(GX, GY, np.ma.array(TAU, mask=MASK), levels=18,
            cmap="Greys", alpha=0.16, zorder=0)

for theta, ref, col in ((THETA, REF1, C_COMP),
                        (THETA + 0.5 * np.pi,
                         np.array([-REF1[1], REF1[0]]), C_EXT)):
    vx, vy = vector_field(theta, ref)
    ax.streamplot(gx, gy, vx, vy, density=0.85, color=col, linewidth=0.9,
                  arrowsize=0, minlength=0.18, zorder=2)

ax.plot(SAF_PTS[:, 0], SAF_PTS[:, 1], "-", color="black", lw=3.4,
        zorder=4, solid_capstyle="round")
for k, pts in MINORS.items():
    ax.plot(pts[:, 0], pts[:, 1], "-", color=COLOUR[k], lw=2.6, zorder=4,
            solid_capstyle="round")


def direction_at(x, y, family):
    i = int(np.clip((x - gx[0]) / (gx[1] - gx[0]), 0, NG - 1))
    j = int(np.clip((y - gy[0]) / (gy[1] - gy[0]), 0, NG - 1))
    th = THETA[j, i] + (0.0 if family == 0 else 0.5 * np.pi)
    return np.array([np.cos(th), np.sin(th)])


# equal arms, so a cross reads as ORIENTATION and not as magnitude
ANCHORS = [(0.13, 0.20), (0.17, 0.83), (0.90, 0.90), (0.90, 0.55),
           (0.45, 0.33), (0.47, 0.63), (0.72, 0.44), (0.30, 0.10)]
for a in ANCHORS:
    for family, col in ((0, C_COMP), (1, C_EXT)):
        d = direction_at(*a, family)
        L = 0.052
        ax.plot([a[0] - L * d[0], a[0] + L * d[0]],
                [a[1] - L * d[1], a[1] + L * d[1]], "-", color=col,
                lw=3.4, zorder=6, solid_capstyle="round")
    ax.plot(*a, "o", ms=3.4, color="black", zorder=7)

for xy, s, ha in (((0.245, 0.455), "San Andreas (N)", "center"),
                  ((0.895, 0.115), "San Andreas (S)", "center"),
                  ((0.655, 0.492), "Garlock", "center"),
                  ((0.60, 0.855), "ECSZ", "center"),
                  ((0.795, 0.245), "San Jacinto", "center"),
                  ((0.325, 0.235), "Transverse Ranges\n(restraining bend)",
                   "center")):
    ax.text(*xy, s, fontsize=9, color="black", ha=ha, va="center",
            zorder=8,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

ax.plot([], [], "-", color=C_COMP, lw=2.6, label="most compressive")
ax.plot([], [], "-", color=C_EXT, lw=2.6, label="most extensional")
ax.plot([], [], "-", color="black", lw=2.6, label="San Andreas (slipping)")
ax.legend(loc="lower left", fontsize=9, framealpha=0.93)

# No figure title: the slide it lives on carries one, and the space is
# better given to the map.
ax.text(0.5, -0.030,
        "trajectories: curves everywhere tangent to a principal direction   "
        "\u00b7   crosses: both principal directions at a point, equal arms",
        transform=ax.transAxes, fontsize=8.2, color="0.3", ha="center")
ax.text(0.5, -0.062,
        "grey: differential stress   \u00b7   trajectories stop at the "
        "San Andreas because it is slipping: the field is discontinuous there",
        transform=ax.transAxes, fontsize=8.2, color="0.3", ha="center")

fig.tight_layout()
out = os.path.join(D, "california-trajectories.png")
fig.savefig(out, dpi=155, facecolor="white", bbox_inches="tight")
print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
