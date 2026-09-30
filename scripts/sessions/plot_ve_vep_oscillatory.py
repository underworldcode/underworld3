"""Plot VE vs VEP oscillatory shear from saved checkpoint."""

import numpy as np
import matplotlib.pyplot as plt

data = np.load("output/ve_vep_oscillatory.npz")
t_ve, s_ve = data["t_ve"], data["s_ve"]
t_vep, s_vep = data["t_vep"], data["s_vep"]
t_ana, s_ana = data["t_ana"], data["s_ana"]
De = float(data["De"])
tau_y = float(data["tau_y"])
ve_amp = float(data["ve_amp"])
omega = float(data["omega"])

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Analytical VE (smooth curve)
ax.plot(t_ana, s_ana, "k-", linewidth=0.8, alpha=0.5, label="VE analytical")

# Numerical VE
ax.plot(t_ve, s_ve, "b-o", markersize=3, linewidth=1.2, label="VE numerical (order 1)")

# Numerical VEP
ax.plot(t_vep, s_vep, "r-s", markersize=3, linewidth=1.2, label=f"VEP numerical ($\\tau_y$={tau_y})")

# Yield stress lines
ax.axhline(tau_y, color="r", linestyle="--", alpha=0.4, linewidth=0.8)
ax.axhline(-tau_y, color="r", linestyle="--", alpha=0.4, linewidth=0.8)
ax.text(0.3, tau_y + 0.02, f"$\\tau_y$ = {tau_y}", color="r", fontsize=9, alpha=0.6)

ax.set_xlabel("Time ($t / t_r$)")
ax.set_ylabel("$\\sigma_{xy}$")
ax.set_title(f"Oscillatory Maxwell VE vs VEP shear (De = {De})")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(0, t_ve[-1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/ve_vep_oscillatory.png", dpi=150)
plt.savefig("output/ve_vep_oscillatory.pdf")
print("Saved output/ve_vep_oscillatory.png and .pdf")
