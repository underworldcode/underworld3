"""Compare convection history (vrms, Nu) between a uniform fixed mesh and an
adapting mesh AT EQUIVALENT PHYSICAL TIME t. If adaptation/remap preserves the
solution, the vrms(t) and Nu(t) curves must overlay — the mesh is just a
discretisation of the same PDE.

dt differs between the runs (adapted fine cells → smaller dt), so step number
does NOT align; everything is interpolated onto a common t grid.

Usage:
  python adapt_vs_uniform_compare.py --sim-dir ~/+Simulations/StagnantLid \
      --uniform cmp_uniform --adapt cmp_adapt
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--sim-dir", default="~/+Simulations/StagnantLid")
ap.add_argument("--uniform", default="cmp_uniform")
ap.add_argument("--adapt", default="cmp_adapt")
ap.add_argument("--out", default="adapt_vs_uniform.png")
args = ap.parse_args()
SIM = os.path.expanduser(args.sim_dir)


def load(tag):
    h = np.load(os.path.join(SIM, tag, "history.npz"))
    return h["t"], h["vrms"], h["Nu"], h["dt"]


tu, vu, nu_u, dtu = load(args.uniform)
ta, va, nu_a, dta = load(args.adapt)

# Common t window = overlap of the two runs.
t0 = max(tu.min(), ta.min())
t1 = min(tu.max(), ta.max())
tg = np.linspace(t0, t1, 40)
vu_i = np.interp(tg, tu, vu); va_i = np.interp(tg, ta, va)
nu_i = np.interp(tg, tu, nu_u); na_i = np.interp(tg, ta, nu_a)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].plot(tu, vu, "-o", ms=3, label=f"uniform ({len(tu)} steps)")
ax[0].plot(ta, va, "-s", ms=3, label=f"adapt ({len(ta)} steps)")
ax[0].set_title("vrms vs t"); ax[0].set_xlabel("t"); ax[0].set_ylabel("vrms"); ax[0].legend()
ax[1].plot(tu, nu_u, "-o", ms=3, label="uniform")
ax[1].plot(ta, nu_a, "-s", ms=3, label="adapt")
ax[1].set_title("Nu vs t"); ax[1].set_xlabel("t"); ax[1].set_ylabel("Nu"); ax[1].legend()
# Relative difference on the common grid.
rel_v = np.abs(va_i - vu_i) / np.maximum(np.abs(vu_i), 1e-30)
rel_n = np.abs(na_i - nu_i) / np.maximum(np.abs(nu_i), 1e-30)
ax[2].plot(tg, 100 * rel_v, "-", label="|Δvrms|/vrms %")
ax[2].plot(tg, 100 * rel_n, "-", label="|ΔNu|/Nu %")
ax[2].set_title("relative difference (adapt vs uniform)")
ax[2].set_xlabel("t"); ax[2].set_ylabel("%"); ax[2].legend()
fig.tight_layout()
outp = os.path.join(SIM, args.out)
fig.savefig(outp, dpi=130)
print("wrote", outp)

print(f"\noverlap t in [{t0:.5f}, {t1:.5f}]")
print(f"  uniform: {len(tu)} steps, t_max={tu.max():.5f}, dt~{np.median(dtu):.2e}")
print(f"  adapt:   {len(ta)} steps, t_max={ta.max():.5f}, dt~{np.median(dta):.2e}")
print(f"\n{'t':>9} {'vrms_unif':>11} {'vrms_adapt':>11} {'dv%':>7} "
      f"{'Nu_unif':>9} {'Nu_adapt':>9} {'dNu%':>7}")
for i in range(0, len(tg), 4):
    print(f"{tg[i]:>9.5f} {vu_i[i]:>11.3e} {va_i[i]:>11.3e} {100*rel_v[i]:>7.2f} "
          f"{nu_i[i]:>+9.3f} {na_i[i]:>+9.3f} {100*rel_n[i]:>7.2f}")
print(f"\nmedian |Δvrms|/vrms = {100*np.median(rel_v):.2f}%   "
      f"median |ΔNu|/Nu = {100*np.median(rel_n):.2f}%")
print(f"max    |Δvrms|/vrms = {100*np.max(rel_v):.2f}%   "
      f"max    |ΔNu|/Nu = {100*np.max(rel_n):.2f}%")
