"""Plot baseline vrms / Nu / Tmin / Tmax evolution from
~/+Simulations/StagnantLid/ot_test/history.npz."""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.expanduser("~/+Simulations/StagnantLid/ot_test")
h = np.load(os.path.join(SRC, "history.npz"))
step = h["step"]
t = h["t"]
vrms = h["vrms"]
Nu = h["Nu"]
Tmin = h["Tmin"]
Tmax = h["Tmax"]
adapted = h["adapted"]

fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                         constrained_layout=True)

ax = axes[0, 0]
ax.plot(step, vrms, "-", color="tab:blue")
adapt_steps = step[adapted > 0.5]
for s in adapt_steps:
    ax.axvline(s, color="0.7", lw=0.5, zorder=-1)
ax.set_xlabel("step")
ax.set_ylabel("vrms")
ax.set_title("vrms (vertical lines = adapt events)")

ax = axes[0, 1]
ax.plot(step, Nu, "-", color="tab:orange")
ax.set_xlabel("step")
ax.set_ylabel("Nu (surface BdIntegral)")
ax.set_title("Nu_surface")

ax = axes[1, 0]
ax.plot(step, Tmin, "-", color="tab:green", label="T_min")
ax.plot(step, Tmax, "-", color="tab:red", label="T_max")
ax.axhline(0.0, color="0.5", lw=0.5)
ax.axhline(1.0, color="0.5", lw=0.5)
ax.set_xlabel("step")
ax.set_ylabel("T extents")
ax.set_title("T_min / T_max (abort if outside [-0.1, 1.1])")
ax.legend()

ax = axes[1, 1]
dt = h["dt"]
ax.semilogy(step, dt, "-", color="tab:purple")
ax.set_xlabel("step")
ax.set_ylabel("dt")
ax.set_title("Δt (CFL-limited)")

fig.suptitle(
    f"baseline: Ra=1e7, Δη=1e2, mode 5  (steps {step[0]:.0f}..{step[-1]:.0f}, "
    f"{int(adapted.sum())} adapts)")
out = "/tmp/ot_test_logs/baseline_history.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"wrote {out}")
print(f"final: step={step[-1]:.0f}  t={t[-1]:.4g}  "
      f"vrms={vrms[-1]:.3e}  Nu={Nu[-1]:.3f}  "
      f"T=[{Tmin[-1]:+.4f},{Tmax[-1]:+.4f}]")
