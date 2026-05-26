"""Ground-truth: the EXACT radial equidistribution map for the
undeformed Annulus, computed as a 1D quadrature (no FE, no UW3).

If the exact optimal-transport / equidistribution map itself only
grades deep/near by ~1.07, then "move once by grad(phi)" cannot
reach strong grading and the plan premise needs revisiting. If it
grades strongly (deep/near >> 1.1), the weak FE result is a solver
accuracy problem and the coupled (phi,H) SNES escalation is
justified.

Radial 2D equidistribution: place node radii r_k so equal target
"mass" m(r) = \int rho_tgt(s) s ds sits between consecutive shells
(area element 2*pi*r dr). Radial spacing dr/dxi ∝ 1/(rho_tgt * r).
This monotone rearrangement IS the OT map under radial symmetry.
"""
import numpy as np

R_I, R_O = 0.5, 1.0
WIDTH = 0.12
N = 200_000          # quadrature resolution
N_SHELLS = 16        # ~ matches RES=16 radial resolution

s = np.linspace(R_I, R_O, N)
ds = s[1] - s[0]

for amp in (0.0, 2.0, 8.0, 20.0):
    rho = 1.0 + amp * np.exp(-(((s - R_O) / WIDTH) ** 2))
    dens = rho * s                       # 2D area weighting
    m = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * ds)])
    m /= m[-1]                           # normalised cumulative mass

    targets = np.linspace(0.0, 1.0, N_SHELLS + 1)
    r_nodes = np.interp(targets, m, s)   # equidistributed shell radii
    dr = np.diff(r_nodes)                # radial spacing per shell
    r_mid = 0.5 * (r_nodes[1:] + r_nodes[:-1])

    near = r_mid > (R_O - WIDTH)
    deep = r_mid < (R_O - 0.30)
    dr_near = dr[near].mean()
    dr_deep = dr[deep].mean()
    # uniform start spacing for reference
    dr0 = (R_O - R_I) / N_SHELLS
    print(
        f"AMP={amp:5.1f}  ideal radial spacing  near={dr_near:.4f} "
        f"deep={dr_deep:.4f}  deep/near={dr_deep/dr_near:6.2f}  "
        f"(uniform dr0={dr0:.4f}; near/dr0={dr_near/dr0:.3f})")
