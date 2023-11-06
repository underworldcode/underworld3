import pytest
import sympy
import underworld3 as uw


r_o = 1.0
r_i = 0.6
res = 0.33

annulus = uw.meshing.Annulus(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=0.1,
    qdegree=2,
)

spherical_shell = uw.meshing.SphericalShell(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=0.4,
    qdegree=2,
)

cubed_sphere = uw.meshing.CubedSphere(
    radiusOuter=r_o,
    radiusInner=r_i,
    numElements=3,
    qdegree=2,
    refinement=0,
)


@pytest.mark.parametrize("mesh", [annulus, cubed_sphere, spherical_shell])
def test_stokes_sphere(mesh):
    if mesh.dim == 2:
        x, y = mesh.X
        z = 0
    else:
        x, y, z = mesh.X

    ra = mesh.CoordinateSystem.R[0]

    u = uw.discretisation.MeshVariable(
        "u",
        mesh,
        mesh.dim,
        vtype=uw.VarType.VECTOR,
        degree=2,
        varsymbol=r"\mathbf{u}",
    )
    p = uw.discretisation.MeshVariable(
        "p",
        mesh,
        1,
        vtype=uw.VarType.SCALAR,
        degree=1,
        continuous=True,
        varsymbol=r"\mathbf{p}",
    )

    # Create a density structure / buoyancy force

    radius_fn = sympy.sqrt(
        mesh.rvec.dot(mesh.rvec)
    )  # normalise by outer radius if not 1.0
    unit_rvec = mesh.X / (radius_fn)

    # Some useful coordinate stuff

    hw = 1000.0 / res
    surface_fn = sympy.exp(-(((ra - r_o) / r_o) ** 2) * hw)
    base_fn = sympy.exp(-(((ra - r_i) / r_o) ** 2) * hw)

    ## Buoyancy (T) field

    t_forcing_fn = 1.0 * (
        sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
        + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
        + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p, verbose=False)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(u)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

    stokes.tolerance = 1.0e-2
    stokes.petsc_options["ksp_monitor"] = None
    # stokes.petsc_options["snes_max_it"] = 1 # for timing cases only - force 1 snes iteration for all examples

    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    # stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "multiplicative")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "v")

    stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "fgmres"
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
    stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

    buoyancy_force = 1.0e6 * t_forcing_fn * (1 - surface_fn) * (1 - base_fn)

    # Free slip condition by penalizing radial velocity at the surface (non-linear term)
    free_slip_penalty_upper = u.sym.dot(unit_rvec) * unit_rvec * surface_fn
    free_slip_penalty_lower = u.sym.dot(unit_rvec) * unit_rvec * base_fn

    stokes.bodyforce = unit_rvec * buoyancy_force
    stokes.bodyforce -= 1000000 * (free_slip_penalty_upper + free_slip_penalty_lower)

    stokes.solve()

    stokes.dm.ds.view()

    assert stokes.snes.getConvergedReason() > 0

    del stokes
    del mesh

    return


test_stokes_sphere(cubed_sphere)


del annulus
del spherical_shell
