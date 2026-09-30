"""3-D spherical-shell constrained free-slip response regression.

This test records facts exposed by the Zhong et al. (2008)-style benchmark:

* the validated Nitsche/default field-split path reproduces the Zhong velocity
  scale for this low-resolution response case;
* a direct-LU diagnostic path gives matching Nitsche and constrained responses,
  but it does not reproduce the validated Nitsche/default response and should
  not be treated as the benchmark reference. A monolithic direct factorisation
  of the *constrained* saddle point is a SERIAL diagnostic only: it gives the
  wrong velocity response and segfaults in parallel (the indefinite KKT
  factorisation is not robust here), so ``Stokes_Constrained`` now emits a
  warning when ``pc_type`` is ``lu``/``cholesky``. Use the grouped
  ``u | [p,h]`` field-split for production;
* the DEFAULT constrained field-split path now reproduces the Zhong velocity
  response: ``Stokes_Constrained`` defaults to a flexible outer Krylov (fgmres)
  with an unpreconditioned (true-residual) convergence test and a tightened
  Eisenstat-Walker tolerance — required because the grouped ``u | [p,h]`` Schur
  preconditioner uses inner *iterative* sub-solves (a variable preconditioner),
  which a non-flexible method false-converged on, giving a wrong, partition-
  dependent answer;
* two *alternative* sub-solver choices remain expected failures for this load:
  the fast grouped-Schur variant with a ``gasm`` Schur block, and the
  direct-LU-sub-solve field-split (LU on the indefinite KKT blocks is not robust
  and segfaults in parallel).

Run:
    pixi run -e amr-dev pytest -q tests/test_1064_constrained_spherical_shell_response.py
"""

from functools import cache

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_3, pytest.mark.slow, pytest.mark.tier_c]

RADIUS_INNER = 0.55
RADIUS_INTERNAL = 0.775
RADIUS_OUTER = 1.0
CELL_SIZE = 1.0 / 8.0
HARMONIC_DEGREE = 2
NITSCHE_GAMMA = 10.0

ZHONG_SURFACE_VELOCITY = 1.006e-2
ZHONG_CMB_VELOCITY = 1.186e-2


@cache
def solve_response(method, solver_mode):
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=RADIUS_OUTER,
        radiusInternal=RADIUS_INTERNAL,
        radiusInner=RADIUS_INNER,
        cellSize=CELL_SIZE,
        qdegree=2,
        degree=1,
    )

    velocity = uw.discretisation.MeshVariable(
        f"U_{method}_{solver_mode}",
        mesh,
        mesh.dim,
        degree=2,
        vtype=uw.VarType.VECTOR,
    )
    pressure = uw.discretisation.MeshVariable(
        f"P_{method}_{solver_mode}",
        mesh,
        1,
        degree=1,
        continuous=True,
    )

    theta = mesh.CoordinateSystem.xR[1]
    unit_r = mesh.CoordinateSystem.unit_e_0
    y_l0 = sympy.assoc_legendre(HARMONIC_DEGREE, 0, sympy.cos(theta))
    harmonic_norm = 4.0 * np.pi / (2 * HARMONIC_DEGREE + 1)

    if method == "constrained":
        stokes = uw.systems.Stokes_Constrained(
            mesh,
            velocityField=velocity,
            pressureField=pressure,
        )
    elif method == "nitsche":
        stokes = uw.systems.Stokes(
            mesh,
            velocityField=velocity,
            pressureField=pressure,
        )
    else:
        raise ValueError(f"Unknown response method {method!r}")

    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, 0.0, 0.0])
    stokes.add_natural_bc(y_l0 * unit_r, mesh.boundaries.Internal.name)

    if method == "nitsche":
        stokes.add_nitsche_bc(0.0, "Upper", normal=unit_r, gamma=NITSCHE_GAMMA)
        stokes.add_nitsche_bc(0.0, "Lower", normal=-unit_r, gamma=NITSCHE_GAMMA)
    else:
        stokes.add_constraint_bc(
            0.0,
            "Upper",
            normal=unit_r,
            augmentation_base=1.0e4,
            degree=2,
        )
        stokes.add_constraint_bc(
            0.0,
            "Lower",
            normal=-unit_r,
            augmentation_base=1.0e4,
            degree=2,
        )

    stokes.petsc_use_nullspace = True
    stokes.tolerance = 1.0e-7
    stokes.petsc_options["snes_type"] = "ksponly"

    if solver_mode == "monolithic":
        stokes.petsc_options["ksp_type"] = "preonly"
        stokes.petsc_options["pc_type"] = "lu"
        stokes.petsc_options["pc_factor_mat_solver_type"] = "mumps"
        stokes.petsc_options["pc_use_amat"] = None
    elif solver_mode == "default":
        pass
    elif solver_mode == "fieldsplit_exact":
        if method != "constrained":
            raise ValueError(
                "fieldsplit_exact mode is only defined for constrained runs"
            )
        stokes.petsc_options["pc_fieldsplit_schur_precondition"] = "selfp"
        stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "preonly"
        stokes.petsc_options["fieldsplit_velocity_pc_type"] = "lu"
        stokes.petsc_options["fieldsplit_1_ksp_type"] = "preonly"
        stokes.petsc_options["fieldsplit_1_pc_type"] = "lu"
    elif solver_mode == "fast_schur":
        if method != "constrained":
            raise ValueError("fast_schur mode is only defined for constrained runs")
        stokes.petsc_options["pc_fieldsplit_schur_precondition"] = "selfp"
        stokes.petsc_options["fieldsplit_1_ksp_type"] = "preonly"
        stokes.petsc_options["fieldsplit_1_pc_type"] = "gasm"
    else:
        raise ValueError(f"Unknown solver mode {solver_mode!r}")

    stokes.solve()

    horizontal_v2 = velocity.sym.dot(velocity.sym) - velocity.sym.dot(unit_r) ** 2

    surface_velocity = np.sqrt(
        uw.maths.BdIntegral(mesh, horizontal_v2, boundary="Upper").evaluate()
        / (
            RADIUS_OUTER**2
            * HARMONIC_DEGREE
            * (HARMONIC_DEGREE + 1)
            * harmonic_norm
        )
    )
    cmb_velocity = np.sqrt(
        uw.maths.BdIntegral(mesh, horizontal_v2, boundary="Lower").evaluate()
        / (
            RADIUS_INNER**2
            * HARMONIC_DEGREE
            * (HARMONIC_DEGREE + 1)
            * harmonic_norm
        )
    )

    return (
        float(surface_velocity),
        float(cmb_velocity),
        int(stokes.snes.getConvergedReason()),
    )


def test_default_nitsche_matches_zhong_velocity_response():
    surface_velocity, cmb_velocity, snes_reason = solve_response(
        "nitsche",
        "default",
    )

    assert snes_reason > 0
    assert abs(surface_velocity - ZHONG_SURFACE_VELOCITY) / ZHONG_SURFACE_VELOCITY < 0.05
    assert abs(cmb_velocity - ZHONG_CMB_VELOCITY) / ZHONG_CMB_VELOCITY < 0.05


def test_default_constrained_matches_zhong_velocity_response():
    """The DEFAULT constrained field-split path reproduces the Zhong response.

    This is the headline of the fgmres + unpreconditioned-norm + tightened-EW
    solver defaults: the grouped u | [p,h] Schur solve now genuinely converges
    (it previously false-converged on the variable preconditioner) and matches
    the validated Nitsche/default velocity scale to within the benchmark tol.
    """
    surface_velocity, cmb_velocity, snes_reason = solve_response(
        "constrained",
        "default",
    )

    assert snes_reason > 0
    assert abs(surface_velocity - ZHONG_SURFACE_VELOCITY) / ZHONG_SURFACE_VELOCITY < 0.05
    assert abs(cmb_velocity - ZHONG_CMB_VELOCITY) / ZHONG_CMB_VELOCITY < 0.05


def test_direct_lu_diagnostic_constrained_matches_direct_lu_diagnostic_nitsche():
    nitsche_surface, nitsche_cmb, nitsche_reason = solve_response(
        "nitsche",
        "monolithic",
    )
    constrained_surface, constrained_cmb, constrained_reason = solve_response(
        "constrained",
        "monolithic",
    )

    assert nitsche_reason > 0
    assert constrained_reason > 0
    assert abs(constrained_surface - nitsche_surface) / nitsche_surface < 0.01
    assert abs(constrained_cmb - nitsche_cmb) / nitsche_cmb < 0.01


@pytest.mark.xfail(
    reason=(
        "Known constrained field-split failure: LU sub-solves in the "
        "velocity | [p,h] preconditioner still do not reproduce the validated "
        "Nitsche/default velocity response."
    ),
    strict=True,
)
def test_lu_subsolve_fieldsplit_constrained_matches_default_nitsche_response():
    nitsche_surface, nitsche_cmb, nitsche_reason = solve_response(
        "nitsche",
        "default",
    )
    constrained_surface, constrained_cmb, constrained_reason = solve_response(
        "constrained",
        "fieldsplit_exact",
    )

    assert nitsche_reason > 0
    assert constrained_reason > 0
    assert abs(constrained_surface - nitsche_surface) / nitsche_surface < 0.01
    assert abs(constrained_cmb - nitsche_cmb) / nitsche_cmb < 0.01


@pytest.mark.xfail(
    reason=(
        "Known fast grouped-Schur constrained response failure for the "
        "3-D SphericalShellInternalBoundary Zhong-style load."
    ),
    strict=True,
)
def test_fast_schur_constrained_matches_zhong_velocity_response():
    surface_velocity, cmb_velocity, snes_reason = solve_response(
        "constrained",
        "fast_schur",
    )

    assert snes_reason > 0
    assert abs(surface_velocity - ZHONG_SURFACE_VELOCITY) / ZHONG_SURFACE_VELOCITY < 0.05
    assert abs(cmb_velocity - ZHONG_CMB_VELOCITY) / ZHONG_CMB_VELOCITY < 0.05
