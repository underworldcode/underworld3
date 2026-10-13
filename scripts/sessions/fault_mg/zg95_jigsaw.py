"""The pieces side by side: for each (label, run dir, checkpoint index) a row with
T on the base, the surface tangential velocity from the checkpointed velocity,
and the horizontally averaged T profile, titled with the step's mobility and
yielded fraction from the run's records.

python zg95_jigsaw.py <out.png> label=dir:index [label=dir:index ...]
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

out = sys.argv[1]
items = []
for a in sys.argv[2:]:
    label, rest = a.split("=", 1)
    d, idx = rest.rsplit(":", 1)
    items.append((label, os.path.expanduser(d), int(idx)))
RO, RI = 1.0, 0.55
fig, axes = plt.subplots(len(items), 3, figsize=(16, 4.3 * len(items)), squeeze=False,
                         gridspec_kw={"width_ratios": [1.1, 1, 0.8]})
for row, (label, d, idx) in enumerate(items):
    with h5py.File(os.path.join(d, f"zg95.mesh.T_b.{idx:05d}.h5")) as h:
        T = np.asarray(h["fields/T_b"]).ravel()
        X = np.asarray(h["fields/coordinates"])
        C = np.asarray(h["fields/cells"]).astype(int)
    vfile = os.path.join(d, f"zg95.mesh.V_b.{idx:05d}.h5")
    V = None
    if os.path.exists(vfile):
        with h5py.File(vfile) as h:
            V = np.asarray(h["fields/V_b"]).reshape(-1, 2)
            XV = np.asarray(h["fields/coordinates"])
    v0, v1, v2, m01, m12, m20 = (C[:, k] for k in range(6))
    tris = np.vstack([np.column_stack([v0, m01, m20]), np.column_stack([m01, v1, m12]),
                      np.column_stack([m20, m12, v2]), np.column_stack([m01, m12, m20])])
    rec = {}
    rf = os.path.join(d, "records.json")
    if os.path.exists(rf):
        recs = [r for r in json.load(open(rf)) if r.get("step") == idx]
        rec = recs[0] if recs else {}
    t = rec.get("t", float("nan"))
    ax = axes[row, 0]
    tc = ax.tripcolor(mtri.Triangulation(X[:, 0], X[:, 1], tris), T, shading="gouraud", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    title = f"{label}: step {idx}, t = {t:.3f}"
    if "mobility" in rec:
        title += f"; mobility {rec['mobility']:.2f}, lid {rec.get('lid_depth', float('nan')):.3f}"
    if rec.get("yield_frac_lid") is not None and not (isinstance(rec.get("yield_frac_lid"), float) and np.isnan(rec["yield_frac_lid"])):
        title += f", yielded in lid {100 * rec['yield_frac_lid']:.0f}%"
    ax.set_title(title, fontsize=9, loc="left")
    if row == 0:
        fig.colorbar(tc, ax=ax, shrink=0.8)
    ax = axes[row, 1]
    if V is not None:
        r = np.hypot(XV[:, 0], XV[:, 1])
        on = r > RO - 1e-6
        th = np.arctan2(XV[on, 1], XV[on, 0])
        vt = -V[on, 0] * np.sin(th) + V[on, 1] * np.cos(th)
        o = np.argsort(th)
        ax.plot(np.degrees(th[o]), vt[o], lw=1)
        ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("surface tangential velocity"); ax.grid(alpha=0.3)
    ax.set_title("surface velocity", fontsize=9, loc="left")
    ax = axes[row, 2]
    depth = RO - np.hypot(X[:, 0], X[:, 1])
    nb = 45
    edges = np.linspace(0, RO - RI, nb + 1)
    k = np.clip(np.digitize(depth, edges) - 1, 0, nb - 1)
    prof = np.bincount(k, weights=T, minlength=nb) / np.maximum(np.bincount(k, minlength=nb), 1)
    ax.plot(prof, 0.5 * (edges[1:] + edges[:-1]), lw=1.2)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("mean T"); ax.set_ylabel("depth"); ax.grid(alpha=0.3)
    ax.set_title("horizontally averaged T", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(out, dpi=120)
print("wrote", out)
