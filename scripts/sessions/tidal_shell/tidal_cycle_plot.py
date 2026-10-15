"""Slip cycle of each stripe against orbital phase, cycle by cycle (the
asymmetry that develops under cyclic loading): signed slip rate along the
stripe, cumulative slip, and the normal traction (clamping) on the stripe.
Usage: <sim> <run> [<run2> ...] -> <sim>/cycle_<run>.png and a per-cycle table."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim = sys.argv[1]
for run in sys.argv[2:]:
    rs = json.load(open(os.path.join(sim, run, "records.json")))
    t = np.array([r["t"] for r in rs])
    names = sorted(rs[0]["slip"].keys())
    ncyc = int(np.ceil(t.max() - 1e-9))
    fig, axes = plt.subplots(len(names), 3, figsize=(16, 3.0 * len(names)), squeeze=False)
    print(f"{run}: net slip per cycle (signed, along the stripe) and clamping range")
    for i, nm in enumerate(names):
        rate = np.array([r["slip"][nm]["signed_mean"] for r in rs])
        cum = np.array([r["slip"][nm]["cumulative"] for r in rs])
        sn = np.array([r["slip"][nm].get("sigma_n_mean", np.nan) for r in rs])
        for c in range(ncyc):
            m = (t > c) & (t <= c + 1 + 1e-9)
            ph = (t[m] - c) * 360.0
            axes[i, 0].plot(ph, rate[m], label=f"cycle {c + 1}")
            axes[i, 1].plot(ph, cum[m] - (cum[m][0] - rate[m][0] * (t[m][0] - c)), label=f"cycle {c + 1}")
            axes[i, 2].plot(ph, sn[m], label=f"cycle {c + 1}")
            net = cum[m][-1] - (cum[m][0] - rate[m][0] * (t[1] - t[0]))
            print(f"  {nm} cycle {c + 1}: net slip {net:+.5f}, |rate| max {np.abs(rate[m]).max():.4f}, "
                  f"sigma_n mean range {np.nanmin(sn[m]):+.4f} .. {np.nanmax(sn[m]):+.4f}")
        axes[i, 0].set_ylabel(nm); axes[i, 0].axhline(0, color="k", lw=0.5)
        axes[i, 1].axhline(0, color="k", lw=0.5); axes[i, 2].axhline(0, color="k", lw=0.5)
        for ax in axes[i]:
            ax.grid(alpha=0.3); ax.set_xlim(0, 360)
    axes[0, 0].set_title("signed slip rate along the stripe"); axes[0, 1].set_title("slip within the cycle (from the cycle start)")
    axes[0, 2].set_title("normal traction on the stripe (negative = clamped)")
    for ax in axes[-1]: ax.set_xlabel("orbital phase (deg)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(run, fontsize=11); fig.tight_layout()
    fig.savefig(os.path.join(sim, f"cycle_{run}.png"), dpi=110); plt.close(fig)
