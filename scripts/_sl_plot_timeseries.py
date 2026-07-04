"""Plot the current state of the adapt_loop_movie_ref3 run —
vrms, Nu, misalignment over time, with the Stokes/AdvDiff
order swap annotated."""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.expanduser(
    os.environ.get('SL_MOVIE_OUT',
                   '~/+Simulations/StagnantLid/adapt_loop_movie_ref5'))
h = np.load(os.path.join(OUT, "history.npz"))
t = h['t']
vrms = h['vrms']
Nu = h['Nu']
mis = h['misalignment'] if 'misalignment' in h.files else None
ad = h['adapted'].astype(bool)
n = len(t)
print(f"history has {n} entries; "
      f"latest t={t[-1]:.5f}, vrms={vrms[-1]:.2f}, "
      f"Nu={Nu[-1]:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(t, vrms, 'b-', lw=1.5)
axes[0].set_ylabel(r'$v_\mathrm{rms}$')
axes[0].grid(alpha=0.3)

axes[1].plot(t, Nu, 'r-', lw=1.5)
axes[1].axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.5)
axes[1].set_ylabel('Nu (surface)')
axes[1].grid(alpha=0.3)

t_swap = None  # ref5 run started post-swap (no annotation needed)

if mis is not None:
    mv = ~np.isnan(mis)
    axes[2].plot(t[mv], mis[mv], 'g-', lw=1.5)
    axes[2].set_ylabel('misalignment')
axes[2].set_xlabel('t')
axes[2].grid(alpha=0.3)

axes[0].set_title(
    'Stagnant lid Ra=1e7 dEta=1e4 from perturbation\n'
    'ref=5, dt-mult=3, adapt every 5, no skip')
plt.tight_layout()
out = os.path.join(OUT, "plot_full_timeseries.png")
plt.savefig(out, dpi=130)
plt.close()
print(f"wrote {out}")
