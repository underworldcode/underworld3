"""Compare the topography small-scale fluctuations for the etd_topo
estimator using: P2 velocity (baseline), P1-projected velocity, and
P1 + screened-Poisson smoothing.  Shows h(θ) and its angular spectrum.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/+Simulations/freesurface_stress_equilibrium")
IN = "output"
cases = [
    ("dtf1.00_n14_upper_res16",       "P2 velocity (baseline)", "C3"),
    ("dtf1.00_n14_upper_res16_sm0",   "P1 velocity",            "C0"),
    ("dtf1.00_n14_upper_res16_sm0.1", "P1 + smooth(0.1)",       "C2"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
for tagsuffix, lbl, c in cases:
    fn = os.path.join(IN, f"phase_i2d_fs_upper_{tagsuffix}.npz")
    if not os.path.exists(fn):
        print("missing", fn); continue
    d = dict(np.load(fn, allow_pickle=True))
    lab = [k.rsplit("_", 1)[0] for k in d if k.endswith("_finalDr")][0]
    th = d[f"{lab}_finalTh"]; dr = d[f"{lab}_finalDr"]
    o = np.argsort(th); th, dr = th[o], dr[o]
    ax1.plot(th, dr, color=c, lw=1.5, label=lbl)
    # angular power spectrum (deviation from mean)
    sig = dr - dr.mean()
    fft = np.abs(np.fft.rfft(sig)) / len(sig)
    k = np.arange(len(fft))
    ax2.semilogy(k, fft + 1e-12, color=c, lw=1.5, label=lbl, marker="o", ms=3)

ax1.set_xlabel("θ (rad)"); ax1.set_ylabel("final h(θ)")
ax1.set_title("Final topography profile"); ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax2.set_xlabel("angular wavenumber k"); ax2.set_ylabel("|ĥ(k)|")
ax2.set_title("Angular spectrum (high-k = small-scale fluctuations)")
ax2.set_xlim(0, 30); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.suptitle("etd_topo: effect of P1 velocity projection / smoothing on "
             "topography noise", fontsize=12)
fig.tight_layout()
p = os.path.join(OUT, "etd_topo_velocity_smoothing.png")
fig.savefig(p, dpi=130); print(f"Wrote {p}")
