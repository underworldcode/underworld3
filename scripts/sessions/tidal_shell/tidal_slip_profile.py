"""Slip along the stripes over the last orbit, Berne et al. Fig. 4 style: the
cycle-integrated lateral slip |ds| (activity) coloured along each trace in the
polar view and as 1-D profiles against distance along the stripe, relative to the
maximum over all stripes; and the same for the signed net offset of the orbit.
Usage: <sim> <run>"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim, run = sys.argv[1:3]
f = sorted(glob.glob(os.path.join(sim, run, "slipprof_*.npz")))[-1]
d = np.load(f)
names = sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_coords")})
pol = np.load(sorted(glob.glob(os.path.join(sim, run, "polar_*.npz")))[-1])
c, e1, e2 = pol["centre"], pol["e1"], pol["e2"]


def to_polar(pts):
    pts = pts / np.linalg.norm(pts, axis=1)[:, None]
    cosr = np.clip(pts @ c, -1, 1); rho = np.arccos(cosr)
    dd = pts - cosr[:, None] * c[None]; dd /= np.maximum(np.linalg.norm(dd, axis=1), 1e-30)[:, None]
    return rho * (dd @ e1), rho * (dd @ e2)


act_max = max(d[f"{n}_act_cycle"].max() for n in names)
net_max = max(np.abs(d[f"{n}_net_cycle"]).max() for n in names)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
for ax, key, vmax, cmap, ttl in ((axes[0], "act_cycle", act_max, "inferno", "cycle-integrated lateral slip along the stripes (relative to the maximum)"),
                                 (axes[1], "net_cycle", net_max, "RdBu_r", "net offset over the orbit (right-lateral positive)")):
    for n in names:
        X = d[f"{n}_coords"]; a, b = to_polar(X)
        s = X @ e1; o = np.argsort(s)
        v = d[f"{n}_{key}"]
        sc = ax.scatter(a[o], b[o], c=(v[o] / vmax if key == "act_cycle" else v[o]), cmap=cmap,
                        vmin=(0 if key == "act_cycle" else -net_max), vmax=(1 if key == "act_cycle" else net_max), s=60, zorder=3)
        ax.plot(a[o], b[o], color="gray", lw=4, zorder=2, alpha=0.4)
    for r in (0.1, 0.2, 0.3):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="gray", lw=0.5, alpha=0.6))
    ax.set_aspect("equal"); ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=9, loc="left"); fig.colorbar(sc, ax=ax, shrink=0.8)
ax = axes[2]
for n in names:
    X = d[f"{n}_coords"]; s = X @ e1; o = np.argsort(s); s = s[o] - s[o][0]
    ax.plot(s, d[f"{n}_act_cycle"][o] / act_max, "-", label=f"{n} activity")
    ax.plot(s, d[f"{n}_net_cycle"][o] / net_max, "--", label=f"{n} net (scaled)")
ax.set_xlabel("distance along the stripe"); ax.set_ylabel("relative to the maximum"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
ax.set_title("1-D profiles along each stripe", fontsize=9, loc="left")
fig.suptitle(f"{run}: slip along the tiger stripes over the last orbit (t = {float(d['t']):.2f}); max activity {act_max:.4f}, max net {net_max:.4f}", fontsize=10)
out = os.path.join(sim, f"slipprofile_{run}.png"); fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)
for n in names:
    X = d[f"{n}_coords"]; s = X @ e1; o = np.argsort(s); v = d[f"{n}_act_cycle"][o]; w = d[f"{n}_net_cycle"][o]
    print(f"  {n}: {len(v)} pairs, activity max {v.max():.4f} at {np.argmax(v)/(len(v)-1):.2f} of the length, "
          f"ends {v[0]/v.max():.2f} / {v[-1]/v.max():.2f} of max; net offset max {w.max():+.4f} min {w.min():+.4f}")
