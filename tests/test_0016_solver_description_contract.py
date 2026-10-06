"""The description contract — every solver says what it solves.

``solver.describe()`` returns the residual templates, the named expressions
inside them, the boundary conditions and the terms the solver was given. It
has two consumers: ``view()`` renders it for a reader, and the run transcript
serialises it, so a note and a run cannot quote different equations.

That only holds if every solver satisfies the contract. This file is the
enforcement: a solver added without it fails here rather than being discovered
later by a reader wondering why its equations are missing from the transcript.

The contract is three things:

  * ``_constraint_mechanisms()`` — every way a constraint can be on the solver,
    ONE enumeration shared with the mixed-mechanism guard;
  * ``_solver_terms`` — the terms the solver is given, as
    ``(attribute, description)`` pairs, from which ``_declared_terms()``
    is built;
  * ``describe()`` runs, on a solver that has been configured and on one that
    has not.
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _solver_classes():
    """Every solver class reachable from the base, by name."""
    import underworld3 as uw  # noqa: F401  (registers the subclasses)
    from underworld3.cython.generic_solvers import SolverBaseClass

    found = {}

    def walk(cls):
        for sub in cls.__subclasses__():
            found[sub.__name__] = sub
            walk(sub)

    walk(SolverBaseClass)
    return found


# Solvers that do not name their terms. Each one here is a solver whose
# description says "terms not declared" rather than showing what it was given.
#
# To adopt: add a ``_solver_terms`` roster to the class — a tuple of
# ``(attribute, description)`` — and delete the name from this list. Do NOT
# add a new solver to this list to make the test pass; the roster is two lines
# and a new solver has no legacy to carry.
NOT_DECLARING_TERMS = {
    "SNES_MultiComponent",       # generic base: the component fields are the terms
    "SNES_Scalar",               # generic base: F0/F1 are set directly
    "SNES_Stokes_SaddlePt",      # generic base: F0/F1/PF0 are set directly
    "SNES_Vector",               # generic base: F0/F1 are set directly
}


def test_every_solver_names_the_terms_it_is_given():
    """A solver that does not declare its terms must be on the list, by name.

    The list is the point: it fails in both directions. A new solver arrives
    undeclared and the test fails; a solver adopts the contract and the test
    fails until the name comes off, so the list cannot rot into a blanket
    exemption.
    """
    from underworld3.cython.generic_solvers import SolverBaseClass

    classes = _solver_classes()
    silent = {
        name for name, cls in classes.items()
        if getattr(cls, "_solver_terms", None) is None
        and cls._declared_terms is SolverBaseClass._declared_terms
    }

    newly_silent = sorted(silent - NOT_DECLARING_TERMS)
    assert not newly_silent, (
        f"these solvers do not name the terms they are given: {newly_silent}. "
        "Add a `_solver_terms` roster — a tuple of (attribute, description) — "
        "to the class. Their equations will otherwise be recorded in the run "
        "transcript without the names the user wrote them under."
    )

    adopted = sorted(NOT_DECLARING_TERMS - silent)
    assert not adopted, (
        f"these solvers now declare their terms: {adopted}. "
        "Remove them from NOT_DECLARING_TERMS in this file."
    )


def test_the_guard_reads_the_same_enumeration_the_description_does(mesh):
    """``_constraint_mechanisms`` is ONE list, read by the mixed-mechanism
    guard and by ``describe()``.

    A second list kept for reporting would let a mechanism added later go
    unmentioned in every description and say nothing. Sharing it means a new
    mechanism breaks the guard first, which is loud — so this injects a
    mechanism into the enumeration and checks the guard sees it.
    """
    import underworld3 as uw

    V = uw.discretisation.MeshVariable("V_guard", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable("P_guard", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

    real = stokes._constraint_mechanisms()
    assert set(real) >= {
        "essential", "natural", "rotated_freeslip", "fault_contact", "multipliers"
    }, real

    mixed = dict(real, rotated_freeslip=[("Top", None)], multipliers=["block"])
    stokes._constraint_mechanisms = lambda: mixed
    with pytest.raises(RuntimeError, match="issue #464"):
        stokes._reject_mixed_constraint_mechanisms("solve")


def test_a_rotated_constraint_appears_in_the_description(mesh):
    """Rotated free-slip is applied by machinery outside the solver. A
    description that skipped it would report "no boundary conditions" for a
    model whose whole boundary treatment is rotated."""
    import underworld3 as uw

    V = uw.discretisation.MeshVariable("V_rot", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable("P_rot", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.add_rotated_freeslip_bc(0.0, "Top")

    conditions = stokes.describe()["boundary_conditions"]
    rotated = [c for c in conditions if c["mechanism"] == "rotated_freeslip"]
    assert rotated, conditions
    assert rotated[0]["boundary"] == "Top"


# ---------------------------------------------------------------------------
# describe() on real solvers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh():
    import underworld3 as uw

    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


def _built(mesh):
    """One of each family, configured as a user would configure it."""
    import underworld3 as uw

    T = uw.discretisation.MeshVariable("T_desc", mesh, 1, degree=2)
    S = uw.discretisation.MeshVariable("S_desc", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable("V_desc", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable("P_desc", mesh, 1, degree=1)

    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0

    darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=S)
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

    advdiff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=V.sym)
    advdiff.constitutive_model = uw.constitutive_models.DiffusionModel
    advdiff.constitutive_model.Parameters.diffusivity = 1.0

    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

    projection = uw.systems.Projection(mesh, u_Field=S)
    projection.uw_function = T.sym[0]

    return {
        "Poisson": poisson,
        "Darcy": darcy,
        "AdvDiffusion": advdiff,
        "Stokes": stokes,
        "Projection": projection,
    }


def test_describe_runs_for_every_family_and_reports_its_terms(mesh):
    """The regression this file exists for: ``SNES_Darcy`` raised on
    ``describe()`` because the contract methods had been put on the
    saddle-point class instead of the base, and eight unrelated tests failed
    at collection."""
    for label, solver in _built(mesh).items():
        described = solver.describe()
        assert described["solver"], label
        assert described["forms"], f"{label} described no residual form"
        assert described["terms_declared"] is True, f"{label} declares no terms"
        names = [t["name"] for t in described["terms"]]
        assert names, f"{label} declared an empty term list"


def test_a_named_expression_is_followed_into_its_definition(mesh):
    """The recursion is the point of the contract: a body force written as a
    product of named quantities must appear under the names the user wrote,
    not only as the number it collapsed to."""
    import underworld3 as uw

    V = uw.discretisation.MeshVariable("V_named", mesh, 2, degree=2)
    P = uw.discretisation.MeshVariable("P_named", mesh, 1, degree=1)
    T = uw.discretisation.MeshVariable("T_named", mesh, 1, degree=2)

    buoyancy = uw.expression(
        r"\rho_0 \alpha g", 1.0e4, "reference density x expansivity x gravity"
    )
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = -buoyancy * T.sym[0] * mesh.CoordinateSystem.unit_e_1

    described = stokes.describe()
    term = next(t for t in described["terms"] if t["name"] == "bodyforce")
    symbols = [w["symbol"] for w in term["where"]]
    assert r"\rho_0 \alpha g" in symbols, symbols
    where = next(w for w in term["where"] if w["symbol"] == r"\rho_0 \alpha g")
    assert "expansivity" in where["description"]


def test_describe_survives_an_unconfigured_solver(mesh):
    """Collection-time safety: ``describe()`` is called on solvers that have
    been constructed and not set up, and must not raise there."""
    import underworld3 as uw

    S = uw.discretisation.MeshVariable("S_bare", mesh, 1, degree=2)
    bare = uw.systems.Poisson(mesh, u_Field=S)
    described = bare.describe()
    assert described["solver"] == "SNES_Poisson"
