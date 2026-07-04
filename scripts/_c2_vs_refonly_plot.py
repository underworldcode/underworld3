"""Focused, readable c2 check: coarsen_cap=2 (equidistribution)
vs a16z (refine-only, percentile-push) — the two directly
comparable runs that use the SAME proper Nusselt definition.
Zoomed to the settled window; settled values annotated. The
old-Nu cached runs (ref24/u16/a16p/a16s) are deliberately
excluded — different (old, unreliable) Nu definition."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/tmp/metric_mesh/sat"
RUNS = [("a16e",  "#2ca02c",
         "resolution_ratio=2 (single-knob equidist., FINAL)"),
        ("a16c2", "#1f77b4",
         "coarsen_cap=2 (legacy hand-tuned 2-knob)"),
        ("a16z",  "#d62728",
         "refine-only (a16z, no de-resolution)")]

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for tag, col, lab in RUNS:
    z = np.load(f"{D}/sat_{tag}_hist.npz")
    t, Nu, vr = z["t"], z["Nu"], z["vrms"]
    ax[0].plot(t, Nu, "-", color=col, lw=1.8, label=lab)
    ax[1].plot(t, vr, "-", color=col, lw=1.8, label=lab)
    # settled value = mean over last 15% of the trajectory
    k = max(1, len(t) // 7)
    nu_s, vr_s = Nu[-k:].mean(), vr[-k:].mean()
    ax[0].axhline(nu_s, color=col, ls=":", lw=1.0, alpha=0.7)
    ax[0].annotate(f"{tag} settled Nu≈{nu_s:.2f}",
                   xy=(t[-1], nu_s), xytext=(-4, 6),
                   textcoords="offset points", color=col,
                   fontsize=10, ha="right", fontweight="bold")
    print(f"{tag:>6}: {len(t):4d} steps  t_end={t[-1]:.4f}  "
          f"settled Nu≈{nu_s:.3f}  vrms≈{vr_s:.1f}")

tmax = 0.045
for a, ttl, yl in ((ax[0], "Nusselt(t) — proper surface-flux def",
                    "Nu"), (ax[1], "vrms(t)", "vrms")):
    a.set_xlim(0, tmax)
    a.set_xlabel("dimensionless time")
    a.set_ylabel(yl)
    a.set_title(ttl)
    a.legend(fontsize=10, loc="best")
    a.grid(alpha=0.3)
ax[0].set_ylim(0, 18)
fig.suptitle("Final equidistribution (1 knob) vs hand-tuned 2-knob "
             "vs refine-only — same Ra=1e5, same proper Nu",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{D}/equidist_final_compare.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
