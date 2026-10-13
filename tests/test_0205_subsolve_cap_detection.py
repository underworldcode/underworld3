"""A sub-solve that stops at its iteration cap must say so.

PETSc's ``KSPCheckSolve`` deliberately does not treat ``DIVERGED_MAX_IT`` on a
sub-KSP as a failure — truncating an inner solve is normal — so a block can run
out of iterations on every application and the outer solve still reports
CONVERGED. That silence is expensive: ``S = -B A^-1 B^T`` is applied *through*
the velocity solve, so once those are truncated the pressure Krylov is chasing
an operator that moves between applications and cannot converge either.
Measured on SolCx at h=1/30 (#625): raising only
``fieldsplit_velocity_ksp_max_it`` from 200 to 5000 took the solve from 976 s to
25.6 s, with an identical answer.

``solve_report.sub[...].capped`` counts the applications that ended that way.
"""

import pytest

import underworld3 as uw
from underworld3 import analytic as A

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _solcx_stokes(velocity_cap):
    """SolCx with a viscosity contrast hard enough to need real iterations."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 10, qdegree=3
    )
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("Ucap", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pcap", mesh, 1, degree=0, continuous=False)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    stokes.saddle_preconditioner = 1.0 / sol.fn_viscosity
    stokes.bodyforce = sol.fn_bodyforce
    for wall, condition in (("Left", (0.0, None)), ("Right", (0.0, None)),
                            ("Bottom", (None, 0.0)), ("Top", (None, 0.0))):
        stokes.add_dirichlet_bc(condition, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["fieldsplit_velocity_ksp_max_it"] = velocity_cap
    stokes.solve()
    return stokes


def _blocks(stokes):
    return {entry.name: entry for entry in stokes.solve_report.sub.values()}


def test_a_capped_velocity_block_is_reported():
    """Cap the velocity block hard; every application must be counted."""
    blocks = _blocks(_solcx_stokes(velocity_cap=2))
    velocity = blocks["velocity"]

    assert velocity.applications > 0
    assert velocity.capped == velocity.applications, (
        f"{velocity.capped} of {velocity.applications} velocity applications "
        "were reported as capped; a 2-iteration cap on a 1e6 viscosity "
        "contrast cannot converge any of them"
    )
    assert "AT THE ITERATION CAP" in str(velocity)


def test_an_untruncated_solve_reports_no_caps():
    """Negative control. Without this, a counter stuck at `applications` passes.

    The same problem with a cap it never reaches must come back clean, or the
    check is reporting the cap rather than detecting it.
    """
    for block in _blocks(_solcx_stokes(velocity_cap=5000)).values():
        assert block.capped == 0, (
            f"{block.name} reported {block.capped} capped application(s) with a "
            f"5000-iteration cap it never approached: {block}"
        )
        assert "AT THE ITERATION CAP" not in str(block)


def test_the_count_distinguishes_some_from_all():
    """`capped` counts applications, not a boolean.

    A block that truncates on some applications and converges on others is the
    interesting middle case — reporting only "did it ever cap" would lose the
    difference between an occasional truncation and a block that never solves.
    """
    velocity = _blocks(_solcx_stokes(velocity_cap=2))["velocity"]
    assert isinstance(velocity.capped, int)
    assert 0 < velocity.capped <= velocity.applications
