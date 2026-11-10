"""Nonzero-flow LU/quotient-PC field agreement, not a scalability proof."""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

from underworld3.utilities._spherical_stokes_gap import _solve_spherical_gap, _solve_spherical_gap_zero_slip

pytestmark = pytest.mark.level_2


@pytest.fixture(scope="module")
def shell():
    return uw.meshing.SphericalShell(radiusInner=.55, radiusOuter=1,
                                    cellSize=1/8, degree=1, qdegree=4)


@pytest.mark.tier_b
@pytest.mark.parametrize("wall", ["free", "zero"])
def test_norm_pc_repeated_nonzero_flow_agrees_with_lu_fields(shell, wall):
    velocity = uw.discretisation.MeshVariable(f"NormFlowU_{wall}", shell, 3, degree=2)
    pressure = uw.discretisation.MeshVariable(f"NormFlowP_{wall}", shell, 1, degree=1)
    reference_velocity = uw.discretisation.MeshVariable(f"NormFlowReferenceU_{wall}", shell, 3, degree=2)
    reference_pressure = uw.discretisation.MeshVariable(f"NormFlowReferenceP_{wall}", shell, 1, degree=1)
    reference_normal = uw.discretisation.MeshVariable(f"NormFlowReferenceNormal_{wall}", shell, 1, degree=2)
    reference_tangent = uw.discretisation.MeshVariable(f"NormFlowReferenceTangent_{wall}", shell, 3, degree=2)
    stokes = uw.systems.Stokes(shell, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    x, y, z = shell.X
    # Smooth nonconservative forcing has zero continuum torque by odd symmetry.
    # Quadrature 4 resolves its degree-four native load integrand.
    stokes.bodyforce = sp.Matrix([[y*z, -x*z, 0]])
    solve = _solve_spherical_gap if wall == "free" else _solve_spherical_gap_zero_slip
    args = (stokes, "Lower", "Upper", .55, 1., 1., sp.Integer(0), 0., stokes.bodyforce, 6)
    direct = solve(*args, {"ksp_type": "preonly", "pc_type": "lu",
                          "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 100}, 128)
    reference_velocity.array[:] = velocity.array
    reference_pressure.array[:] = pressure.array
    reference_normal.array[:] = direct.normal_reaction.array
    reference_tangent.array[:] = direct.tangential_reaction.array
    force_norm = np.sqrt(abs(uw.maths.Integral(shell, stokes.bodyforce.dot(stokes.bodyforce)).evaluate()))
    velocity_norm = np.sqrt(abs(uw.maths.Integral(shell, reference_velocity.sym.dot(reference_velocity.sym)).evaluate()))
    assert velocity_norm > 1e-7*force_norm
    # Match the production work budget: the Gadi zero-slip mesh needs 2010
    # iterations. Residual tolerances and the independent LU checks stay fixed.
    options = {"ksp_type": "fgmres", "pc_type": "python", "ksp_rtol": 1e-9,
               "ksp_atol": 1e-50, "ksp_max_it": 4000, "ksp_gmres_restart": 100}
    for repeat in range(2):
        result = solve(*args, options, 128, None, 1/8)
        checks = result.checks
        assert checks["ksp_reason"] > 0 and checks["relative_residual"] <= 1e-8
        assert checks["norm_preconditioner_length"] == 1/8
        assert checks["norm_context"] == "_SphericalGapNormPreconditioner"
        assert checks["norm_velocity_block_size"] == 3
        assert checks["rhs_removed_null_relative"] <= 1e-10
        errors = []
        for computed, reference in ((velocity, reference_velocity), (pressure, reference_pressure),
                                    (result.normal_reaction, reference_normal), (result.tangential_reaction, reference_tangent)):
            difference = computed.sym-reference.sym
            error = np.sqrt(abs(uw.maths.Integral(shell, difference.dot(difference)).evaluate()))
            errors.append(float(error))
        # This is fixed-operator linear-solve agreement. Traction/pressure
        # errors use the independent characteristic scale L*||f||, L=radius1.
        assert errors[0] <= 1e-5*velocity_norm
        assert max(errors[1:]) <= 1e-5*force_norm
        if uw.mpi.rank == 0:
            print("NORM_PC_NONZERO_COMPARISON", dict(wall=wall, repeat=repeat, errors=errors,
                  velocity_scale=velocity_norm, pressure_traction_scale=force_norm, **checks), flush=True)


@pytest.mark.tier_b
def test_norm_pc_rejects_implicit_solver_choices_before_assembly(shell):
    velocity = uw.discretisation.MeshVariable("NormGuardU", shell, 3, degree=2)
    pressure = uw.discretisation.MeshVariable("NormGuardP", shell, 1, degree=1)
    stokes = uw.systems.Stokes(shell, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.zeros(1, 3)
    with pytest.raises(ValueError, match="explicit ksp_type=fgmres"):
        _solve_spherical_gap(stokes, "Lower", "Upper", .55, 1., 1., sp.Integer(0), 0.,
                             sp.zeros(1, 3), 6, {"ksp_type": "preonly", "pc_type": "lu"},
                             128, None, 1/8)
