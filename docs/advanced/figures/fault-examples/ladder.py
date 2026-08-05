"""The fault-strength ladder: one fault, one drive, four laws.

A horizontal fault under far-field simple shear (resolved shear stress
tau_infty = eta * rate = 1 on the fault plane), solved with each rung of
the constitutive ladder. The slip profile V(s) tells the whole story:

- frictionless: the free crack, elliptical profile, full stress drop;
- viscous eta_f = eta/a: the half-slip dashpot;
- Coulomb, weak (mu sigma_n < tau_infty): slides at reduced stress drop
  tau_infty - mu sigma_n, still elliptical;
- Coulomb, strong (mu sigma_n > tau_infty): sticks — creep at the
  regularisation scale V0, invisible at plot scale;
- rate-and-state (steady state at its own strength): sits between.
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
SIGMA_N = 2.0                       # prescribed normal stress for friction

mesh = common.base_mesh(cell_size=0.04)
segment = common.fault_segment(0.0, half_length=HALF)

runs = []


def solve_case(label, register):
    child = common.split_with_fault(common.base_mesh(0.04), segment)
    stokes = common.stokes_on(child, common.simple_shear(child))
    register(stokes)
    fault_contact.solve_with_fault(stokes, picard=2)
    s, V, leak = common.slip_profile(stokes)
    assert np.abs(leak).max() < 1e-10
    runs.append((label, s, V))
    print(f"{label:32s} peak slip {np.abs(V).max():.4f}")


solve_case("frictionless",
           lambda st: st.add_fault_bc(0, boundary="Fault"))
solve_case(r"viscous, $\eta_f = \eta/a$",
           lambda st: st.add_fault_bc(common.ETA / HALF, boundary="Fault"))
solve_case(r"Coulomb, $\mu\sigma_n = 0.6 < \tau_\infty$",
           lambda st: fault_contact.add_coulomb_fault_bc(
               st, 0.3, "Fault", sigma_n=SIGMA_N, V0=1e-4))
solve_case(r"rate-state (steady, $f_{ss}\sigma_n \approx 0.86$)",
           lambda st: fault_contact.add_rate_state_fault_bc(
               st, 0.42, "Fault", a=0.02, b=0.01, V0=1e-3, Dc=1e-2,
               sigma_n=SIGMA_N))
solve_case(r"Coulomb, $\mu\sigma_n = 1.2 > \tau_\infty$ (stuck)",
           lambda st: fault_contact.add_coulomb_fault_bc(
               st, 0.6, "Fault", sigma_n=SIGMA_N, V0=1e-4))

fig, ax = plt.subplots(figsize=(7.2, 4.6))
colors = ["#c62828", "#e57373", "#d9960a", "#4a7bf7", "#555555"]
for (label, s, V), col in zip(runs, colors):
    ax.plot(s - s.min() - HALF, np.abs(V), ".-", ms=3.5, lw=1.0,
            color=col, label=label)

ss = np.linspace(-HALF, HALF, 200)
ax.plot(ss, np.sqrt(np.maximum(HALF**2 - ss**2, 0)) / HALF
        * np.abs(runs[0][2]).max(), "k--", lw=0.8,
        label="elliptical profile (shape)")

ax.set_xlabel("position along the fault $s$")
ax.set_ylabel("slip rate $|V(s)|$")
ax.set_title("The fault-strength ladder: one fault, four laws "
             r"($\tau_\infty = 1$)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(-HALF * 1.15, HALF * 1.6)
fig.tight_layout()
out = os.path.join(D, "ladder.png")
fig.savefig(out, dpi=200)
print("wrote", out)
