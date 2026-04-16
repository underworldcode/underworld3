import pytest

pytestmark = pytest.mark.level_3

import sympy
import underworld3 as uw


def _configure_shell_stokes(mesh):
    u = uw.discretisation.MeshVariable(
        "u_shell_nullspace",
        mesh,
        mesh.dim,
        vtype=uw.VarType.VECTOR,
        degree=2,
    )
    p = uw.discretisation.MeshVariable(
        "p_shell_nullspace",
        mesh,
        1,
        vtype=uw.VarType.SCALAR,
        degree=1,
        continuous=True,
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0] * mesh.dim)

    stokes.tolerance = 1.0e-4
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["ksp_rtol"] = 1.0e-4
    stokes.petsc_options["ksp_atol"] = 0.0

    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
    stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

    gamma = mesh.Gamma
    stokes.add_natural_bc(1.0e4 * gamma.dot(u) * gamma, "Upper")
    stokes.add_natural_bc(1.0e4 * gamma.dot(u) * gamma, "Lower")

    stokes.petsc_use_pressure_nullspace = True

    return stokes


@pytest.mark.parametrize(
    ("mesh", "rotation_modes", "expected_mode_count"),
    [
        (
            uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.6, cellSize=0.25, qdegree=2),
            [sympy.Matrix([-sympy.Symbol("y"), sympy.Symbol("x")])],
            2,
        ),
        (
            uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.6, cellSize=0.5, qdegree=2),
            [
                sympy.Matrix([0, -sympy.Symbol("z"), sympy.Symbol("y")]),
                sympy.Matrix([sympy.Symbol("z"), 0, -sympy.Symbol("x")]),
                sympy.Matrix([-sympy.Symbol("y"), sympy.Symbol("x"), 0]),
            ],
            4,
        ),
    ],
)
def test_stokes_shell_rotation_nullspace(mesh, rotation_modes, expected_mode_count):
    if mesh.dim == 2:
        x, y = mesh.X
        coordinate_subs = {
            sympy.Symbol("x"): x,
            sympy.Symbol("y"): y,
        }
    else:
        x, y, z = mesh.X
        coordinate_subs = {
            sympy.Symbol("x"): x,
            sympy.Symbol("y"): y,
            sympy.Symbol("z"): z,
        }

    resolved_modes = [mode.subs(coordinate_subs) for mode in rotation_modes]

    stokes = _configure_shell_stokes(mesh)
    stokes.petsc_velocity_nullspace_basis = resolved_modes
    stokes.solve()

    assert stokes.snes.getConvergedReason() > 0
    assert len(stokes._stokes_nullspace_basis) == expected_mode_count

    jacobian = stokes.snes.getJacobian()
    nullspace = jacobian[0].getNullSpace()

    assert nullspace is not None

    velocity_is = stokes._subdict["velocity"][0]
    pressure_is = stokes._subdict["pressure"][0]
    pressure_modes = 0
    velocity_modes = 0

    for basis_vec in stokes._stokes_nullspace_basis:
        velocity_subvec = basis_vec.getSubVector(velocity_is)
        pressure_subvec = basis_vec.getSubVector(pressure_is)

        try:
            velocity_norm = velocity_subvec.norm()
            pressure_norm = pressure_subvec.norm()
        finally:
            basis_vec.restoreSubVector(velocity_is, velocity_subvec)
            basis_vec.restoreSubVector(pressure_is, pressure_subvec)

        if pressure_norm > 1.0e-10:
            pressure_modes += 1
            assert velocity_norm == pytest.approx(0.0, abs=1.0e-12)
        else:
            velocity_modes += 1
            assert velocity_norm > 0.0

    assert pressure_modes == 1
    assert velocity_modes == expected_mode_count - 1
