"""The penalty is paired with the velocity preconditioner, automatically.

Grad-div augmentation only pays under the custom-P multigrid. Measured on SolCx
(eta 1e6, P2-P0disc, 2592 cells, #625): with FMG, `penalty = 10` is 21% faster,
cuts the Schur count per application 59 -> 18 and total velocity iterations
546 -> 270. With GAMG the identical value makes the solve SLOWER, 15.5 -> 20.7 s,
because augmentation is exactly what drives GAMG into its iteration cap.

So the penalty defaults on where FMG will be used and off where it will not, and
the harmful pairing warns rather than sitting there costing time. The trap this
closes: FMG needs a mesh hierarchy, and on an unrefined base the velocity block
silently falls back to GAMG.
"""

import warnings

import pytest

import underworld3 as uw
from underworld3 import analytic as A

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _stokes(refinement, penalty=None, preconditioner=None):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8,
        qdegree=3, refinement=refinement,
    )
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1)
    v = uw.discretisation.MeshVariable("Upen", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Ppen", mesh, 1, degree=0, continuous=False)

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
    if preconditioner is not None:
        stokes.preconditioner = preconditioner
    if penalty is not None:
        stokes.penalty = penalty
    return stokes


def _solve(stokes):
    """Solve, returning the installed velocity PC and any penalty warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stokes.solve()
    installed = stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC().getType()
    mine = [c for c in caught if "penalty" in str(c.message)]
    return installed, float(stokes.penalty.sym), mine


def test_penalty_switches_on_where_fmg_will_run():
    """A refined base has a hierarchy, so FMG runs and the penalty pays."""
    installed, penalty, warned = _solve(_stokes(refinement=2))
    assert installed == "mg"
    assert penalty == uw.systems.Stokes.AUTO_PENALTY_WITH_FMG
    assert not warned


def test_penalty_stays_off_where_it_would_cost():
    """An unrefined base falls back to GAMG, where augmentation is harmful.

    This is the trap: nothing about the call says GAMG, and before this the
    penalty would have been applied into the pairing measured slowest.
    """
    installed, penalty, warned = _solve(_stokes(refinement=0))
    assert installed == "gamg"
    assert penalty == 0.0
    assert not warned, "the automatic value stood down; there is nothing to warn about"


def test_an_explicit_gamg_choice_also_turns_the_penalty_off():
    """Asking for GAMG on a mesh that could do FMG must not keep the penalty."""
    installed, penalty, _warned = _solve(_stokes(refinement=2, preconditioner="gamg"))
    assert installed == "gamg"
    assert penalty == 0.0


def test_an_explicit_penalty_is_honoured_over_the_automatic_one():
    """The latch: once set, the value is the user's."""
    installed, penalty, _warned = _solve(_stokes(refinement=2, penalty=0.0))
    assert installed == "mg"
    assert penalty == 0.0, "an explicit 0 must not be overwritten by the default"


def test_the_harmful_pairing_warns_and_is_recorded():
    """Explicit penalty on GAMG: measured slower, so it must not be silent.

    The warning is on the PAIRING, not on who chose it -- a user who asks for
    augmentation on an unrefined mesh gets the same slow solve as one who had it
    chosen for them.
    """
    stokes = _stokes(refinement=0, penalty=3.0)
    installed, penalty, warned = _solve(stokes)

    assert installed == "gamg"
    assert penalty == 3.0, "the requested value is still honoured"
    assert warned, "the measured-harmful pairing was applied silently"
    assert "SLOWER" in str(warned[0].message)
    assert "penalty.expected_fmg" in stokes.pc_fallbacks, (
        "pc_fallbacks is the place to read PC degradations; this one must "
        "appear there and not only as a warning"
    )
