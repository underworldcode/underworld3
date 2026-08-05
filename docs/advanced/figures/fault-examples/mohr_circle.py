"""The Mohr circle, measured by faults.

A WELDED split fault transmits the full stress across itself, so it is
a passive stress probe: the no-opening constraint's reaction gives the
normal traction sigma_n, and the stiff interface dashpot reads the
shear traction off its own law, tau = eta_f V. Sweeping the fault
orientation theta and plotting (−sigma_n, |tau|) traces the Mohr
circle of the ambient stress state — measured by the machinery itself,
against the analytic circle of the imposed flow.

Drive: v = (a(x−c) + gamma(y−c), −a(y−c)), deviatoric stress
[[2 eta a, eta gamma], [eta gamma, −2 eta a]]; Mohr radius
R = eta sqrt(4 a^2 + gamma^2). The centre sits at the (gauge-fixed)
mean pressure, which the fit reports rather than assumes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import underworld3 as uw
from underworld3.utilities import fault_contact

import common

D = os.path.dirname(os.path.abspath(__file__))
HALF = 0.2
A_RATE, GAMMA = 0.5, 1.0
ETA_WELD = 200.0 * common.ETA / HALF        # welded: slip ~ tau/eta_f -> 0
R_ANALYTIC = common.ETA * np.sqrt(4 * A_RATE**2 + GAMMA**2)

angles = np.arange(0.0, 180.0 - 1e-9, 22.5)
points = []
for theta in angles:
    sigma_n, tau = common.mohr_probe(theta, A_RATE, GAMMA,
                                     eta_weld=ETA_WELD, half_length=HALF)
    points.append((theta, sigma_n, tau))
    print(f"theta {theta:5.1f}: sigma_n {sigma_n:8.4f}  tau {tau:7.4f}")

points = np.array(points)

# circle fit: centre c on the sigma axis, radius R, from the probes
sig, tau = points[:, 1], points[:, 2]
c0 = float(np.mean(sig))
for _ in range(60):                          # two-parameter Gauss fit
    r = np.sqrt((sig - c0) ** 2 + tau ** 2)
    R0 = r.mean()
    c0 -= float(np.mean((c0 - sig) * (1 - R0 / np.maximum(r, 1e-30))))
print(f"fit: centre {c0:.4f}, radius {R0:.4f} "
      f"(analytic radius {R_ANALYTIC:.4f})")

fig, ax = plt.subplots(figsize=(6.8, 4.8))
tt = np.linspace(0, 2 * np.pi, 300)
ax.plot(c0 + R_ANALYTIC * np.cos(tt), R_ANALYTIC * np.sin(tt), "k-",
        lw=1.0, label=f"analytic circle, $R = {R_ANALYTIC:.3f}$")
ax.plot(c0 + R0 * np.cos(tt), R0 * np.sin(tt), "--", lw=0.9,
        color="#4a7bf7", label=f"fit through the probes, $R = {R0:.3f}$")
ax.plot(sig, tau, "o", ms=6, color="#c62828", zorder=5,
        label="welded-fault probes")
ax.axvline(c0, color="0.6", lw=0.6)
for theta, sg, tu in points:
    ax.annotate(f"{theta:.0f}°", (sg, tu), textcoords="offset points",
                xytext=(6, 5), fontsize=8)
ax.axhline(0, color="0.6", lw=0.6)
ax.set_xlabel(r"normal traction $\sigma_n$")
ax.set_ylabel(r"shear traction $\tau$")
ax.set_title("The Mohr circle, measured by welded split-node faults")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="center")
fig.tight_layout()
out = os.path.join(D, "mohr-circle.png")
fig.savefig(out, dpi=200)
print("wrote", out)

assert abs(R0 - R_ANALYTIC) < 0.06 * R_ANALYTIC, "radius off by > 6%"
