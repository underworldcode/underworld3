"""Berne et al. (2024) Fig. 2 analogue: the locked shell's driving tractions on
the stripe planes (shear, normal), the Coulomb traction tau_c = |tau| - mu (T_b - sigma_n)
(sigma_n tension positive; positive tau_c = failure), and the frictional shell's
accumulated lateral slip, all against mean anomaly, for the last cycle; plus the
spin-up parameter Xi(t) = (S(t) - S(t + T)) / S(t) on the mean slip (their Eq. 21).
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
names = sorted(L[0]["drive"].keys())
last_L = tL > tL.max() - 1.0 + 1e-9
last_F = tF > tF.max() - 1.0 + 1e-9
phL = (tL[last_L] - np.floor(tL[last_L] - 1e-9)) * 360.0
phF = (tF[last_F] - np.floor(tF[last_F] - 1e-9)) * 360.0
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
tau_all, sn_all, tc_all, s_all = [], [], [], []
for nm in names:
    tau = np.array([r["drive"][nm]["tau"] for r in L])[last_L]
    sn = np.array([r["drive"][nm]["sigma_n"] for r in L])[last_L]
    tc = np.abs(tau) - mu * (Tb - sn)
    axes[0, 0].plot(phL, tau, label=nm); axes[0, 1].plot(phL, sn, label=nm); axes[1, 0].plot(phL, tc, label=nm)
    tau_all.append(tau); sn_all.append(sn); tc_all.append(tc)
    if nm in F[0]["slip"]:
        cum = np.array([r["slip"][nm]["cumulative"] for r in F])[last_F]
        s = cum - cum[0]
        axes[1, 1].plot(phF, s, label=nm); s_all.append(s)
for ax, arrs in ((axes[0, 0], tau_all), (axes[0, 1], sn_all), (axes[1, 0], tc_all), (axes[1, 1], s_all)):
    if arrs:
        ax.plot(phL if ax is not axes[1, 1] else phF, np.mean(arrs, axis=0), "k", lw=2.2, label="combined")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.set_xlim(0, 360); ax.set_xlabel("mean anomaly (deg)")
axes[0, 0].set_title("driving shear traction on the stripe planes (locked shell)")
axes[0, 1].set_title("driving normal traction (tension positive)")
axes[1, 0].set_title(f"Coulomb traction |tau| - mu (T_b - sigma_n), mu = {mu}, T_b = {Tb}  (> 0: failure)")
axes[1, 1].set_title(f"accumulated lateral slip, frictional shell ({fric})")
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Berne et al. Fig. 2 analogue: last cycle of {locked} / {fric}", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(sim, f"berne_fig2_{fric}.png"), dpi=120)

# spin-up parameter on the mean slip rate magnitude over the stripes
fig2, ax = plt.subplots(figsize=(7, 4))
S = np.array([np.mean([abs(r["slip"][nm]["signed_mean"]) for nm in r["slip"]]) for r in F])
dt = tF[1] - tF[0]; nper = int(round(1.0 / dt))
xi = (S[:-nper] - S[nper:]) / np.maximum(S[:-nper], 1e-12)
ax.plot(tF[:-nper], xi); ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
ax.set_xlabel("time (periods)"); ax.set_ylabel("spin-up parameter Xi"); ax.set_title(f"cycle-to-cycle mismatch of the mean slip rate, {fric}")
fig2.tight_layout(); fig2.savefig(os.path.join(sim, f"berne_spinup_{fric}.png"), dpi=120)
print("net slip over the last cycle per stripe:", {nm: round(float(s[-1]), 5) for nm, s in zip(names, s_all)})
print("Xi at the last available cycle:", round(float(xi[-1]), 4) if len(xi) else None)
