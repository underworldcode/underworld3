"""The penalty default, and the silent fall off the multigrid.

Grad-div augmentation pays under the custom-P multigrid and costs under GAMG.
Measured on SolCx (eta 1e6, P2-P0disc, 2592 cells, #625): with FMG,
`penalty = 10` is 21% faster and cuts total velocity iterations 546 -> 270;
with GAMG the identical value makes the solve slower, 15.5 -> 20.7 s, because
augmentation is what drives GAMG into its iteration cap.

The penalty is applied **unconditionally** anyway, and that is the point of this
file. Selecting it from the preconditioner was tried and rejected: a
preconditioner must change the path, not the answer, and `test_1017`,
`test_0835` and `test_0836` assert exactly that by comparing FMG and GAMG
solutions to 1e-4. So the penalty is a discretisation choice, and the real
defect -- a velocity block that drops off the multigrid with nothing said -- is
made loud instead.
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
    """Solve, returning the installed velocity PC, the penalty, and warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stokes.solve()
    installed = stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC().getType()
    mine = [c for c in caught
            if "velocity block fell back" in str(c.message)]
    return installed, float(stokes.penalty.sym), mine


def test_the_penalty_default_does_not_depend_on_the_preconditioner(monkeypatch):
    """The invariant. Same default whichever velocity PC ends up installed.

    This is the property the rejected design broke: an operator term chosen by
    the solver means the preconditioner changes the answer, and `test_1017`,
    `test_0835` and `test_0836` all assert it does not.

    The default is patched to a value that is not the shipped one, so the test
    cannot pass by accident if `DEFAULT_PENALTY` is ever changed to whatever is
    asserted here — and could not pass by comparing 0 to 0 back when it shipped
    at 0.
    """
    monkeypatch.setattr(uw.systems.Stokes, "DEFAULT_PENALTY", 7.0)

    on_fmg, penalty_fmg, _ = _solve(_stokes(refinement=2))
    on_gamg, penalty_gamg, _ = _solve(_stokes(refinement=0))

    assert on_fmg == "mg" and on_gamg == "gamg"
    assert penalty_fmg == penalty_gamg == 7.0


def test_an_explicit_penalty_is_honoured():
    """The latch: once set, the value is the user's."""
    _installed, penalty, _warned = _solve(_stokes(refinement=2, penalty=0.0))
    assert penalty == 0.0, "an explicit 0 must not be overwritten by the default"


def test_falling_off_the_multigrid_is_loud():
    """The recurring defect: no hierarchy, so GAMG, and nothing said.

    FMG needs a mesh hierarchy. Without one the velocity block drops to GAMG,
    which degrades under refinement until it hits its cap -- and a capped
    velocity solve corrupts the Schur operator and stalls the pressure block
    (976 s vs 25.6 s at h=1/30). Silence there is what makes it recur.
    """
    stokes = _stokes(refinement=0)
    installed, _penalty, warned = _solve(stokes)

    assert installed == "gamg"
    assert warned, "the velocity block fell off the multigrid silently"
    assert "refinement>=1" in str(warned[0].message), "the message must say how to fix it"
    assert "velocity.fell_back_from_fmg" in stokes.pc_fallbacks, (
        "pc_fallbacks is where PC degradations are read; this one must appear "
        "there and not only as a warning"
    )


def test_a_hierarchy_means_no_warning():
    """Negative control. Without it, a warning that always fires would pass."""
    _installed, _penalty, warned = _solve(_stokes(refinement=2))
    assert not warned


def test_asking_for_gamg_is_a_choice_not_a_fallback():
    """An explicit `preconditioner="gamg"` must not be nagged about.

    A warning that fires on a deliberate choice trains people to ignore it,
    which costs the case it exists for.
    """
    stokes = _stokes(refinement=2, preconditioner="gamg")
    installed, _penalty, warned = _solve(stokes)

    assert installed == "gamg"
    assert not warned
    assert "velocity.fell_back_from_fmg" not in stokes.pc_fallbacks
