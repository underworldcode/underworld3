import pytest

pytestmark = pytest.mark.level_3

import sympy
import underworld3 as uw


def test_stokes_pressure_nullspace_solves_without_pressure_bc():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
    x, y = mesh.X

    u = uw.discretisation.MeshVariable(
        "u_nullspace",
        mesh,
        mesh.dim,
        vtype=uw.VarType.VECTOR,
        degree=2,
    )
    p = uw.discretisation.MeshVariable(
        "p_nullspace",
        mesh,
        1,
        vtype=uw.VarType.SCALAR,
        degree=1,
        continuous=True,
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, x])
    stokes.petsc_use_pressure_nullspace = True

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

    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, None), "Top")
    stokes.add_dirichlet_bc((0.0, None), "Left")
    stokes.add_condition(conds=(0.0, None), label="Right", f_id=0, c_type="dirichlet")

    stokes.solve()

    assert stokes.snes.getConvergedReason() > 0
    assert len(stokes._stokes_nullspace_basis) == 1

    jacobian = stokes.snes.getJacobian()
    nullspace = jacobian[0].getNullSpace()
    assert nullspace is not None

    basis_vec = stokes._stokes_nullspace_basis[0]
    velocity_is = stokes._subdict["velocity"][0]
    pressure_is = stokes._subdict["pressure"][0]

    velocity_subvec = basis_vec.getSubVector(velocity_is)
    pressure_subvec = basis_vec.getSubVector(pressure_is)

    try:
        assert velocity_subvec.norm() == pytest.approx(0.0, abs=1.0e-12)
        assert pressure_subvec.norm() > 0.0
    finally:
        basis_vec.restoreSubVector(velocity_is, velocity_subvec)
        basis_vec.restoreSubVector(pressure_is, pressure_subvec)
