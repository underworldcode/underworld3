"""Fault orientation vs slip: the Mohr circle's other face.

The same orientation sweep as the Mohr demo, but with FRICTIONLESS
faults under pure simple shear: each fault drops the shear stress
resolved on its own plane, so the peak slip rate follows the resolved
shear tau(theta) = tau_infty cos(2 theta) — the slip-rate version of
reading the Mohr circle. Faults near 45 degrees to the shear plane
barely slip; the fault aligned with it slips fully.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import underworld3 as uw

import common

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2

angles = np.arange(0.0, 90.0 + 1e-9, 15.0)
peaks, profiles = [], []
for theta in angles:
    child = common.split_with_fault(
        common.base_mesh(0.04), common.fault_segment(theta, HALF))
    stokes = common.stokes_on(child, common.simple_shear(child))
    stokes.add_fault_bc(0, boundary="Fault")
    stokes.solve(verbose=False)
    s, V, leak = common.slip_profile(stokes)
    assert np.abs(leak).max() < 1e-10
    peaks.append(np.abs(V).max())
    profiles.append((theta, s - s.min() - HALF, np.abs(V)))
    print(f"theta {theta:5.1f}: peak slip {peaks[-1]:.4f}")

peaks = np.array(peaks)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
cmap = plt.cm.magma
for (theta, s, V) in profiles:
    ax1.plot(s, V, ".-", ms=3, lw=0.9, color=cmap(theta / 120.0),
             label=f"{theta:.0f}°")
ax1.set_xlabel("position along the fault $s$")
ax1.set_ylabel("slip rate $|V(s)|$")
ax1.set_title("frictionless slip profiles by orientation")
ax1.legend(fontsize=8, title="fault angle", ncol=2)

tt = np.linspace(0, 90, 200)
ax2.plot(tt, np.abs(np.cos(np.radians(2 * tt))) * peaks[0], "k-",
         lw=1.0, label=r"$|\cos 2\theta|$ (resolved shear)")
ax2.plot(angles, peaks, "o", ms=6, color="#c62828",
         label="measured peak slip")
ax2.set_xlabel(r"fault angle $\theta$ to the shear plane")
ax2.set_ylabel("peak slip rate")
ax2.set_title("peak slip follows the resolved shear stress")
ax2.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(D, "orientations.png")
fig.savefig(out, dpi=200)
print("wrote", out)
