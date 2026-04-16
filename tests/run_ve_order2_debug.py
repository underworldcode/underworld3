"""Debug order-2 VE: trace psi_star values at each step."""

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

stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=2)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
stokes.constitutive_model.Parameters.shear_modulus = MU
stokes.constitutive_model.Parameters.dt_elastic = dt

stokes.add_dirichlet_bc((V0, 0.0), "Top")
stokes.add_dirichlet_bc((-V0, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
stokes.tolerance = 1.0e-6

Stress = uw.discretisation.MeshVariable(
    "Stress", mesh, (2, 2), vtype=uw.VarType.SYM_TENSOR, degree=2, continuous=True,
)
work = uw.discretisation.MeshVariable("W", mesh, 1, degree=2)
sigma_proj = uw.systems.Tensor_Projection(mesh, tensor_Field=Stress, scalar_Field=work)

ddt = stokes.DFDt
centre = np.array([[0.0, 0.0]])

time_phys = 0.0
for step in range(5):
    eff_order = stokes.constitutive_model.effective_order

    # Read psi_star values at centre BEFORE solve
    psi0_pre = uw.function.evaluate(ddt.psi_star[0].sym[0, 1], centre)
    psi1_pre = uw.function.evaluate(ddt.psi_star[1].sym[0, 1], centre)
    psi0_val = float(psi0_pre.flatten()[0])
    psi1_val = float(psi1_pre.flatten()[0])

    # Check psi_fn formula
    psi_fn_01 = str(stokes.DFDt.psi_fn[0, 1])
    has_2star = "**" in psi_fn_01 or "psi_star" in psi_fn_01

    t0 = timer.time()
    stokes.solve(zero_init_guess=False, evalf=False)
    solve_t = timer.time() - t0

    psi_fn_01_post = str(stokes.DFDt.psi_fn[0, 1])
    print(f"  psi_fn[0,1] BEFORE solve: {psi_fn_01[:100]}...")
    print(f"  psi_fn[0,1] AFTER  solve: {psi_fn_01_post[:100]}...")
    time_phys += dt

    # Read psi_star AFTER solve (after update_post_solve)
    psi0_post = uw.function.evaluate(ddt.psi_star[0].sym[0, 1], centre)
    psi1_post = uw.function.evaluate(ddt.psi_star[1].sym[0, 1], centre)
    psi0_post_val = float(psi0_post.flatten()[0])
    psi1_post_val = float(psi1_post.flatten()[0])

    sigma_proj.uw_function = stokes.stress_deviator
    sigma_proj.solve()
    val = uw.function.evaluate(Stress.sym[0, 1], centre)
    sigma_xy = float(val.flatten()[0])
    ana = ETA * gamma_dot * (1.0 - np.exp(-time_phys * MU / ETA))

    print(f"step {step}  eff_order={eff_order}  t={time_phys:.2f}  solve={solve_t:.1f}s")
    print(f"  PRE:  psi_star[0]_xy={psi0_val:.6f}  psi_star[1]_xy={psi1_val:.6f}")
    print(f"  POST: psi_star[0]_xy={psi0_post_val:.6f}  psi_star[1]_xy={psi1_post_val:.6f}")
    print(f"  sigma_xy={sigma_xy:.6f}  analytical={ana:.6f}  "
          f"rel_err={abs(sigma_xy - ana) / ana:.3e}")
    print()

print("Done")
