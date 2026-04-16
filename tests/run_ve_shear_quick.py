"""Quick order-1 VE shear box validation (10 steps)."""

import time as timer
import numpy as np
import sympy
import underworld3 as uw

ETA, MU, V0, H, W = 1.0, 1.0, 0.5, 1.0, 2.0
dt = 0.1
gamma_dot = 2.0 * V0 / H

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 8), minCoords=(-W/2, -H/2), maxCoords=(W/2, H/2),
)
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=1)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
stokes.constitutive_model.Parameters.shear_modulus = MU
stokes.constitutive_model.Parameters.dt_elastic = dt

stokes.add_dirichlet_bc((V0, 0.0), "Top")
stokes.add_dirichlet_bc((-V0, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
stokes.tolerance = 1.0e-6

centre = np.array([[0.0, 0.0]])
ddt = stokes.DFDt

time_phys = 0.0
for step in range(10):
    t0 = timer.time()
    stokes.solve(zero_init_guess=False, evalf=False)
    solve_t = timer.time() - t0
    time_phys += dt

    # Read stress from solver.tau (the projected actual stress)
    val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
    sigma_xy = float(val.flatten()[0])
    ana = ETA * gamma_dot * (1.0 - np.exp(-time_phys * MU / ETA))
    print(f"step {step}  t={time_phys:.2f}  solve={solve_t:.1f}s  "
          f"sigma_xy={sigma_xy:.6f}  analytical={ana:.6f}  "
          f"rel_err={abs(sigma_xy - ana) / ana:.3e}")

print("Done")
