"""Surface lateral velocity above the flow: from a zg95 checkpoint, the surface
tangential velocity against theta, aligned above the unrolled annulus (theta,
depth) with temperature and streamlines, and the annulus itself with streamlines
beside it. The trench is drawn when the step wrote its surface file.

python zg95_stream.py <run dir> <index> [out.png]
"""
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import LinearNDInterpolator

run, idx = sys.argv[1], int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(run, f"stream_{idx:04d}.png")
RO, RI = 1.0, 0.55
with h5py.File(os.path.join(run, f"zg95.mesh.T_b.{idx:05d}.h5")) as h:
    T = np.asarray(h["fields/T_b"]).ravel()
    X = np.asarray(h["fields/coordinates"])
    C = np.asarray(h["fields/cells"]).astype(int)
with h5py.File(os.path.join(run, f"zg95.mesh.V_b.{idx:05d}.h5")) as h:
    V = np.asarray(h["fields/V_b"]).reshape(-1, 2)
    XV = np.asarray(h["fields/coordinates"])
state = {}
sf = os.path.join(run, f"state_{idx:04d}.json")
if os.path.exists(sf):
    state = json.load(open(sf))
rec = {}
rf = os.path.join(run, "records.json")
if os.path.exists(rf):
    rr = [r for r in json.load(open(rf)) if r.get("step") == idx]
    rec = rr[0] if rr else {}
trace = None
surf = os.path.join(run, f"surface_{idx:04d}.npz")
if not os.path.exists(surf):
    # a checkpoint without its own surface file: the nearest earlier one (the trench
    # moves about half a degree per step, so a few steps old is still the fault's place)
    import glob
    cands = sorted(glob.glob(os.path.join(run, "surface_*.npz")))
    cands = [c for c in cands if int(os.path.basename(c)[8:12]) <= idx]
    surf = cands[-1] if cands else surf
if os.path.exists(surf):
    trace = np.load(surf)["trace"]

r = np.hypot(X[:, 0], X[:, 1])
th = np.arctan2(X[:, 1], X[:, 0])
rV = np.hypot(XV[:, 0], XV[:, 1])
thV = np.arctan2(XV[:, 1], XV[:, 0])
vt = -V[:, 0] * np.sin(thV) + V[:, 1] * np.cos(thV)
vr = V[:, 0] * np.cos(thV) + V[:, 1] * np.sin(thV)

# unrolled grid (theta in degrees, depth), periodic in theta: pad the nodes by one period
def padded(th_, vals, *more):
    th3 = np.concatenate([th_ - 2 * np.pi, th_, th_ + 2 * np.pi])
    return th3, [np.concatenate([v, v, v]) for v in (vals, *more)]
th3, (T3, r3) = padded(th, T, r)
th3V, (vt3, vr3, rV3) = padded(thV, vt, vr, rV)
nth, nd = 720, 120
thg = np.linspace(-np.pi, np.pi, nth)
dg = np.linspace(0.0, RO - RI, nd)
TH, D = np.meshgrid(thg, dg)
Rg = RO - D
Tg = LinearNDInterpolator(np.column_stack([th3, r3]), T3)(TH, Rg)
vtg = LinearNDInterpolator(np.column_stack([th3V, rV3]), vt3)(TH, Rg)
vrg = LinearNDInterpolator(np.column_stack([th3V, rV3]), vr3)(TH, Rg)
# streamlines in the unrolled plane: d(theta_deg)/dt = v_t / r (in degrees), d(depth)/dt = -v_r
U = np.degrees(vtg / Rg)
W = -vrg

# annulus grid for the second streamline view
ng = 500
xg = np.linspace(-RO, RO, ng)
XG, YG = np.meshgrid(xg, xg)
RG = np.hypot(XG, YG)
inside = (RG > RI + 1e-3) & (RG < RO - 1e-3)
vx_i = LinearNDInterpolator(XV, V[:, 0])(XG, YG)
vy_i = LinearNDInterpolator(XV, V[:, 1])(XG, YG)
vx_i[~inside] = np.nan
vy_i[~inside] = np.nan

o = np.argsort(thV[rV > RO - 1e-6])
th_s = thV[rV > RO - 1e-6][o]
vt_s = vt[rV > RO - 1e-6][o]

VEXAG = 2.0           # vertical exaggeration of the unrolled section (1 = true aspect at mid-depth)
_deg_per_len = 360.0 / (np.pi * (RO + RI))     # degrees of arc per length unit at mid-radius
fig = plt.figure(figsize=(19, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[1, 1.15], hspace=0.08, wspace=0.12)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
ax2 = fig.add_subplot(gs[:, 1])
ax0.plot(np.degrees(th_s), vt_s, lw=1.2)
ax0.axhline(0, color="k", lw=0.5)
if trace is not None:
    tt = np.degrees(np.arctan2(trace[:, 1], trace[:, 0]))
    ax0.axvline(tt[0], color="#c0392b", lw=0.8, ls="--")
ax0.set_ylabel("surface tangential velocity")
ax0.grid(alpha=0.3)
plt.setp(ax0.get_xticklabels(), visible=False)
t = state.get("t", float("nan"))
title = f"{os.path.basename(os.path.normpath(run))}: step {idx}, t = {t:.4f}"
for k, nm in (("mobility", "mobility"), ("lid_depth", "lid"), ("yield_frac_lid", "yielded in lid")):
    if rec.get(k) is not None and not (isinstance(rec[k], float) and np.isnan(rec[k])):
        title += f"; {nm} {rec[k]:.2f}"
ax0.set_title(title, fontsize=10, loc="left")
pc = ax1.pcolormesh(np.degrees(TH), D, Tg, cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
ax1.streamplot(np.degrees(thg), dg, U, W, density=(2.2, 0.9), color="k", linewidth=0.6, arrowsize=0.8)
if trace is not None:
    ax1.plot(np.degrees(np.arctan2(trace[:, 1], trace[:, 0])), RO - np.hypot(trace[:, 0], trace[:, 1]),
             color="#c0392b", lw=2)
ax1.set_ylim(RO - RI, 0)
ax1.set_xlim(-180, 180)
ax1.set_aspect(VEXAG * _deg_per_len, adjustable="box", anchor="N")
ax1.set_xlabel("theta (deg)")
ax1.set_ylabel("depth")
fig.colorbar(pc, ax=ax1, orientation="horizontal", pad=0.42, shrink=0.4, aspect=40, label="T")
v0, v1, v2, m01, m12, m20 = (C[:, k] for k in range(6))
tris = np.vstack([np.column_stack([v0, m01, m20]), np.column_stack([m01, v1, m12]),
                  np.column_stack([m20, m12, v2]), np.column_stack([m01, m12, m20])])
ax2.tripcolor(mtri.Triangulation(X[:, 0], X[:, 1], tris), T, shading="gouraud", cmap="RdBu_r", vmin=0, vmax=1)
ax2.streamplot(xg, xg, vx_i, vy_i, density=1.6, color="k", linewidth=0.5, arrowsize=0.7)
if trace is not None:
    ax2.plot(trace[:, 0], trace[:, 1], color="#c0392b", lw=2)
ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_xlim(-1.02, 1.02); ax2.set_ylim(-1.02, 1.02)
ax2.set_title("streamlines on the annulus", fontsize=10, loc="left")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
