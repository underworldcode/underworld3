"""Run W2 + W3 + W4b for both azimuths at a given resolution.
   Designed for parallel execution: mpirun -np 8 python _run_w2_w3_w4b.py
"""

import numpy as np
import sympy
import time
from pathlib import Path
import underworld3 as uw
from underworld3.workflows import WorkflowProducts

RES = 28
AZIMUTHS = [0, 30]

uw.reset_default_model()
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

shared = WorkflowProducts(products_dir=Path(f"output/h2ex/res_{RES}/shared/products"))
mesh = shared.load("adapted_mesh")
faults = shared.load("fault_surfaces", mesh=mesh)
uw.pprint(0, f"Mesh: {mesh.X.coords.shape[0]} nodes, {len(faults.surfaces)} faults")

eta_0 = uw.expression(r"\eta_0", uw.quantity(1e21, "Pa.s"), "reference viscosity")
eta_1_weak = uw.expression(r"\eta_1", uw.quantity(0.1e21, "Pa.s"), "weak fault viscosity")
fields = faults.compute_nearest_fields(mesh, fault_width=uw.quantity(10.0, "km"))
fault_normal, fault_weight = fields["normal"], fields["weight"]
eta_1_expr = eta_0 - (eta_0 - eta_1_weak) * fault_weight.sym[0]

geo = mesh.CoordinateSystem.geo
V0 = uw.expression(r"V_0", uw.quantity(1, "cm/yr"), "driving velocity")
V0_nd = uw.non_dimensionalise(V0)
penalty = 1.0e6 * uw.non_dimensionalise(eta_0)

for AZIMUTH in AZIMUTHS:
    run = WorkflowProducts(
        products_dir=Path(f"output/h2ex/res_{RES}/azimuth_{AZIMUTH:03d}/products")
    )
    uw.pprint(0, f"\n{'='*60}")
    uw.pprint(0, f"AZIMUTH = {AZIMUTH}, RES = {RES}")
    uw.pprint(0, f"{'='*60}")

    # --- W2: Stokes ---
    uw.pprint(0, "\n--- W2: Stokes TI ---")
    v = uw.discretisation.MeshVariable(f"v_{AZIMUTH}", mesh, mesh.dim, degree=2, units="cm/yr")
    p = uw.discretisation.MeshVariable(f"p_{AZIMUTH}", mesh, 1, degree=1, units="MPa")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")

    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
    stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1_expr
    stokes.constitutive_model.Parameters.director = fault_normal.sym
    stokes.saddle_preconditioner = 1.0 / uw.non_dimensionalise(eta_0)

    theta = np.radians(AZIMUTH)
    drive_dir = np.cos(theta) * geo.unit_east + np.sin(theta) * geo.unit_north

    stokes.add_natural_bc(penalty * geo.unit_down.dot(v.sym) * geo.unit_down, "Bottom")
    stokes.add_natural_bc(penalty * (drive_dir.dot(v.sym) + V0_nd) * drive_dir, "East")
    stokes.add_natural_bc(penalty * (drive_dir.dot(v.sym) - V0_nd) * drive_dir, "West")
    stokes.add_natural_bc(penalty * geo.unit_north.dot(v.sym) * geo.unit_north, "North")
    stokes.add_natural_bc(penalty * geo.unit_north.dot(v.sym) * geo.unit_north, "South")

    t0 = time.time()
    stokes.solve(verbose=False)
    uw.pprint(0, f"  Stokes TI: {time.time()-t0:.0f}s, v DOFs={v.coords.shape[0]*mesh.dim}")

    strain_rate = uw.discretisation.MeshVariable(f"eps_{AZIMUTH}", mesh, 1, degree=1)
    proj = uw.systems.Projection(mesh, strain_rate)
    proj.uw_function = stokes.constitutive_model.Unknowns.Einv2
    proj.smoothing = 1e-6
    proj.solve()

    uw.pprint(0, "--- W2: Stokes Reference ---")
    v_ref = uw.discretisation.MeshVariable(f"v_ref_{AZIMUTH}", mesh, mesh.dim, degree=2, units="cm/yr")
    p_ref = uw.discretisation.MeshVariable(f"p_ref_{AZIMUTH}", mesh, 1, degree=1, units="MPa")
    stokes_ref = uw.systems.Stokes(mesh, velocityField=v_ref, pressureField=p_ref)
    stokes_ref.petsc_options["snes_type"] = "newtonls"
    stokes_ref.petsc_options["ksp_type"] = "fgmres"
    stokes_ref.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes_ref.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
    stokes_ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes_ref.constitutive_model.Parameters.shear_viscosity_0 = eta_0
    stokes_ref.saddle_preconditioner = 1.0 / uw.non_dimensionalise(eta_0)
    stokes_ref.add_natural_bc(penalty * geo.unit_down.dot(v_ref.sym) * geo.unit_down, "Bottom")
    stokes_ref.add_natural_bc(penalty * (drive_dir.dot(v_ref.sym) + V0_nd) * drive_dir, "East")
    stokes_ref.add_natural_bc(penalty * (drive_dir.dot(v_ref.sym) - V0_nd) * drive_dir, "West")
    stokes_ref.add_natural_bc(penalty * geo.unit_north.dot(v_ref.sym) * geo.unit_north, "North")
    stokes_ref.add_natural_bc(penalty * geo.unit_north.dot(v_ref.sym) * geo.unit_north, "South")

    t0 = time.time()
    stokes_ref.solve(verbose=False)
    uw.pprint(0, f"  Stokes ref: {time.time()-t0:.0f}s")

    strain_rate_ref = uw.discretisation.MeshVariable(f"eps_ref_{AZIMUTH}", mesh, 1, degree=1)
    proj_ref = uw.systems.Projection(mesh, strain_rate_ref)
    proj_ref.uw_function = stokes_ref.constitutive_model.Unknowns.Einv2
    proj_ref.smoothing = 1e-6
    proj_ref.solve()

    # Permeability
    delta = strain_rate.data[:, 0] - strain_rate_ref.data[:, 0]
    w = fault_weight.data[:, 0]
    near_fault = w > 0.1
    delta_near = delta[near_fault]
    q25, q75 = np.percentile(delta_near, 25), np.percentile(delta_near, 75)
    permeability = uw.discretisation.MeshVariable(f"perm_{AZIMUTH}", mesh, 1, degree=1)
    log_perm = np.zeros_like(delta)
    log_perm[near_fault & (delta > q75)] = 1
    log_perm[near_fault & (delta < q25)] = -1
    permeability.data[:, 0] = np.power(10.0, log_perm * w)

    run.save("stokes_velocity", v)
    run.save("stokes_pressure", p)
    run.save("strain_rate", strain_rate)
    run.save("strain_rate_ref", strain_rate_ref)
    run.save("permeability", permeability)
    uw.pprint(0, "  W2 checkpointed")

    # --- W3: Darcy ---
    uw.pprint(0, "\n--- W3: Darcy ---")
    h = uw.discretisation.MeshVariable(f"h_darcy_{AZIMUTH}", mesh, 1, degree=2, units="m")
    vd = uw.discretisation.MeshVariable(f"v_darcy_{AZIMUTH}", mesh, mesh.dim, degree=1,
                                         continuous=True, units="m/s")
    darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=h, v_Field=vd)
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
    darcy.constitutive_model.Parameters.permeability = permeability.sym[0]
    darcy.add_essential_bc(0.0, "Surface")

    depth_sym = geo[2]
    source_depth = uw.expression(f"d_s_{AZIMUTH}", uw.quantity(30, "km"), "source depth")
    darcy.f = sympy.Piecewise((10.0, depth_sym > source_depth), (0.0, True))
    darcy.constitutive_model.Parameters.s = 0.01 * geo.unit_down
    darcy.tolerance = 1e-3
    darcy._v_projector.tolerance = 1e-3
    darcy._v_projector.smoothing = 0.0

    t0 = time.time()
    darcy.solve()
    uw.pprint(0, f"  Darcy: {time.time()-t0:.1f}s")

    run.save("darcy_head", h)
    run.save("darcy_velocity", vd)
    uw.pprint(0, "  W3 checkpointed")

    # --- W4b: Concentration ---
    uw.pprint(0, "\n--- W4b: Concentration ---")
    kappa = uw.expression(f"k_C_{AZIMUTH}", uw.quantity(1e-11, "m**2/s"), "tracer diffusivity")
    C = uw.discretisation.MeshVariable(f"C_{AZIMUTH}", mesh, 1, degree=1)
    adv = uw.systems.AdvDiffusion(mesh, u_Field=C, V_fn=vd)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = kappa
    adv.add_essential_bc(1.0, "Bottom")
    adv.add_essential_bc(0.0, "Surface")

    dt = uw.quantity(200, "Myr")
    dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
    r_dm = np.sqrt(np.sum(dm_coords**2, axis=1))
    near_surf = (r_dm > r_dm.max() - 0.12 * (r_dm.max() - r_dm.min())) & \
                (r_dm < r_dm.max() - 0.05 * (r_dm.max() - r_dm.min()))

    t0 = time.time()
    for step in range(100):
        adv.solve(timestep=dt, zero_init_guess=(step == 0))
        if step % 25 == 0:
            C_near = np.asarray(
                uw.function.evaluate(C.sym, dm_coords[near_surf], mode="fast")
            ).ravel()
            uw.pprint(0, f"  step {step}: C_near=[{C_near.min():.3f},{C_near.max():.3f}], "
                         f"{time.time()-t0:.0f}s")

    uw.pprint(0, f"  Concentration done: {time.time()-t0:.0f}s")
    run.save("concentration", C)
    uw.pprint(0, "  W4b checkpointed")

uw.pprint(0, "\n=== All done ===")
