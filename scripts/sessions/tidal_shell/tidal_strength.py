"""How far the tidal shear overcomes the fault strength: the driving shear on the
stripes (locked shell) against the Coulomb strength mu (T_b - sigma_n) read on the
frictional shell, the slip rate, and the build-up released at each onset.
Usage: <sim> <locked run> <frictional run> <mu> <T_b>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim, locked, fric = sys.argv[1:4]
mu, Tb = float(sys.argv[4]), float(sys.argv[5])
L = json.load(open(os.path.join(sim, locked, "records.json")))
F = json.load(open(os.path.join(sim, fric, "records.json")))
tL = np.array([r["t"] for r in L]); tF = np.array([r["t"] for r in F])
lastL = tL > tL.max() - 1 + 1e-9; lastF = tF > tF.max() - 1 + 1e-9
names = sorted(F[0]["slip"].keys())
tau = np.mean([[r["drive"][n]["tau"] for r in L] for n in names], axis=0)[lastL]
ph = (tL[lastL] - np.floor(tL[lastL] - 1e-9)) * 360
sn = np.mean([[r["slip"][n]["sigma_n_mean"] for r in F] for n in names], axis=0)[lastF]
rate = np.mean([[r["slip"][n]["signed_mean"] for r in F] for n in names], axis=0)[lastF]
phF = (tF[lastF] - np.floor(tF[lastF] - 1e-9)) * 360
strength = mu * np.maximum(Tb - sn, 0)
tau_i = np.interp(phF, ph, tau)
excess = np.abs(tau_i) - strength
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
ax = axes[0]
ax.plot(phF, np.abs(tau_i), "k", lw=2, label="|driving shear| (locked shell)")
ax.plot(phF, strength, "r", lw=2, label="strength mu (T_b - sigma_n) (frictional shell)")
ax.fill_between(phF, strength, np.abs(tau_i), where=np.abs(tau_i) > strength, color="orange", alpha=0.35, label="excess: shear above strength")
ax.set_xlabel("mean anomaly (deg)"); ax.set_xlim(0, 360); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("driving shear against fault strength (stripe mean, last cycle)")
ax2 = axes[1]
ax2.plot(phF, excess, "orange", lw=2, label="excess |tau| - strength")
ax2.plot(phF, rate / np.abs(rate).max() * np.abs(excess).max(), "b--", lw=1.5, label="slip rate (scaled)")
ax2.axhline(0, color="k", lw=0.6); ax2.set_xlim(0, 360); ax2.grid(alpha=0.3); ax2.legend(fontsize=8); ax2.set_xlabel("mean anomaly (deg)")
ax2.set_title("Coulomb excess and the slip it drives")
fig.tight_layout(); fig.savefig(os.path.join(sim, f"strength_{fric}.png"), dpi=120)
# numbers: onsets (excess crossing zero upward), the strength and shear there, peak excess per window
sgn = np.sign(excess); on = np.where((sgn[1:] > 0) & (sgn[:-1] <= 0))[0] + 1
print(f"{fric}: peak |driving shear| {np.abs(tau_i).max():.4f}, strength range {strength.min():.4f} .. {strength.max():.4f} "
      f"(fully unclamped at sigma_n = T_b: {mu*Tb:.4f} at sigma_n 0)")
print(f"  fraction of the cycle with |tau| > strength: {(excess > 0).mean():.2f}")
for k in on:
    w = np.arange(k, len(excess)); w = w[: np.argmax(excess[w] <= 0) or len(w)]
    print(f"  onset at {phF[k]:5.0f} deg: |tau| {abs(tau_i[k]):.4f}, strength {strength[k]:.4f}, strength falling at "
          f"{(strength[k]-strength[k-1])/(phF[k]-phF[k-1])*360:+.4f}/period vs |tau| changing at {(abs(tau_i[k])-abs(tau_i[k-1]))/(phF[k]-phF[k-1])*360:+.4f}/period; "
          f"window peak excess {excess[w].max():.4f} ({excess[w].max()/max(strength[w].mean(),1e-12):.0%} of the strength), slip in window {np.trapezoid(rate[w], phF[w]/360):+.4f}")
