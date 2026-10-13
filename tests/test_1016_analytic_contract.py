"""The uw.analytic contract.

`uw.analytic` is the namespace for exact solutions used to validate a solve.
These tests fix the contract itself — that the base class exposes the fields and
error norms every solution promises, that the boundary-condition helpers
configure a solver rather than returning something the caller has to apply, and
that a solution reached through the new namespace is the *same object* as the one
reached through the old one.

That last check matters more than it looks: `uw.function.analytic` is a compiled
extension module, and a namespace that re-declared its classes rather than
re-exporting them would silently break `isinstance` for anyone holding an object
built through the other path.

Run: pixi run python -m pytest tests/test_1016_analytic_contract.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy
import underworld3 as uw


@pytest.fixture
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )


class Quadratic(uw.analytic.AnalyticSolution):
    r"""A manufactured Stokes solution used to exercise the contract.

    Divergence-free by construction: :math:`\mathbf u = (y, -x)` is a rigid
    rotation, so any solver-independent check of the contract has an exact answer
    that is trivial to verify by hand.
    """

    dim = 2
    eqn_velocity = r"(z, -x)"
    eqn_viscosity = r"1"
    reference = "Contract test fixture; not a published benchmark."

    def __init__(self, mesh):
        super().__init__(mesh)

        x, y = mesh.X

        self.fn_velocity = sympy.Matrix([[y, -x]])
        self.fn_pressure = sympy.sympify(0)
        self.fn_viscosity = sympy.sympify(1)
        self.fn_bodyforce = sympy.Matrix([[0, 0]])
        self.fn_strainrate = sympy.Matrix([[0, 0], [0, 0]])
        self.fn_stress = sympy.Matrix([[0, 0], [0, 0]])

    def apply_boundary_conditions(self, solver):
        uw.analytic.prescribed_velocity(solver, self.boundaries, self.fn_velocity)
        solver.petsc_use_pressure_nullspace = True


def test_contract_is_exported():
    """The contract and the BC helpers are reachable from the package namespace.

    A solution written outside this package has to be able to reach the helpers
    it composes, so they are public API rather than a private convenience.
    """
    for name in (
        "AnalyticSolution",
        "free_slip",
        "prescribed_scalar",
        "prescribed_velocity",
    ):
        assert name in uw.analytic.__all__
        assert hasattr(uw.analytic, name)


def test_solcx_is_the_same_object_in_both_namespaces():
    """uw.analytic re-exports SolCx; it does not redeclare it.

    If the old namespace redeclared its classes rather than resolving to the
    new ones, `isinstance` would fail for any object built through the other
    path.
    """
    from underworld3.function import analytic as legacy

    assert uw.analytic.SolCx is legacy.SolCx


def test_legacy_namespace_is_a_package_not_an_extension():
    """The shim must stay a package directory, not a module file.

    `underworld3.function.analytic` used to be a compiled extension. An
    orphaned `.so` from an earlier install cannot be removed by `pip uninstall`
    if it has fallen out of the wheel RECORD, and it would be imported in place
    of the shim — silently restoring the old module and bypassing every
    redirect. Python's path finder checks for a package directory *before* an
    extension of the same name, which is the only thing that makes the stale
    `.so` harmless. Keep this a directory.
    """
    from underworld3.function import analytic as legacy

    assert legacy.__file__.endswith("__init__.py")


def test_legacy_namespace_warns_on_use():
    """Importing the shim is quiet; using a name from it is not."""
    import importlib

    from underworld3.function import analytic as legacy

    # The warning fires once per process, so reset the latch to observe it.
    legacy._warned = False

    with pytest.warns(DeprecationWarning, match="moved to underworld3.analytic"):
        legacy.SolCx


def test_legacy_namespace_rejects_unknown_names():
    from underworld3.function import analytic as legacy

    with pytest.raises(AttributeError):
        legacy.SolNotAThing


def test_available_lists_solcx():
    """The registry reports what can actually be constructed."""
    assert "SolCx" in uw.analytic.available()
    assert uw.analytic.available() == sorted(uw.analytic.available())


def test_describe_rejects_unknown_names():
    with pytest.raises(ValueError, match="no analytic solution named"):
        uw.analytic.describe("SolNotAThing")


def test_dimension_is_checked(mesh):
    """A 2D solution refuses a 3D mesh rather than producing nonsense."""

    class ThreeDimensional(Quadratic):
        dim = 3

    with pytest.raises(ValueError, match="3D solution"):
        ThreeDimensional(mesh)


def test_named_fields_resolve(mesh):
    sol = Quadratic(mesh)

    assert sol._exact("velocity") is sol.fn_velocity
    assert sol._exact("pressure") is sol.fn_pressure

    # An expression passes straight through, so error() and evaluate() work on
    # anything derived from the solution, not only its named fields.
    x, y = mesh.X
    assert sol._exact(x + y) == x + y


def test_unknown_field_names_are_reported(mesh):
    sol = Quadratic(mesh)

    with pytest.raises(ValueError, match="no field 'temperature'"):
        sol._exact("temperature")


def test_evaluate_returns_exact_values(mesh):
    """evaluate() is the exact field, at arbitrary points, vectorised."""
    sol = Quadratic(mesh)

    coords = np.array([[0.25, 0.75], [0.5, 0.5], [0.9, 0.1]])
    values = np.asarray(sol.evaluate("velocity", coords)).reshape(3, 2)

    expected = np.column_stack([coords[:, 1], -coords[:, 0]])
    assert np.allclose(values, expected)


def test_error_is_zero_for_the_exact_field(mesh):
    """A variable carrying the exact solution has zero error."""
    sol = Quadratic(mesh)

    velocity = uw.discretisation.MeshVariable("Uc", mesh, mesh.dim, degree=2)
    velocity.array[:, 0, :] = np.asarray(
        sol.evaluate("velocity", velocity.coords)
    ).reshape(-1, mesh.dim)

    assert sol.error("velocity", velocity) < 1.0e-12


def test_error_scales_with_the_perturbation(mesh):
    """A known relative perturbation is reported as that relative error.

    Scaling the computed field by (1 + eps) must give a relative L2 error of
    exactly eps — this pins the normalisation, which a zero-error test cannot.
    """
    sol = Quadratic(mesh)

    velocity = uw.discretisation.MeshVariable("Up", mesh, mesh.dim, degree=2)
    exact = np.asarray(sol.evaluate("velocity", velocity.coords)).reshape(
        -1, mesh.dim
    )

    for eps in (1.0e-3, 1.0e-1):
        velocity.array[:, 0, :] = exact * (1.0 + eps)
        assert np.isclose(sol.error("velocity", velocity), eps, rtol=1.0e-8)


def test_error_is_the_same_on_every_rank(mesh):
    """The nodal norm is a global reduction, not each rank's own partition.

    The perturbation is deliberately confined to x < 0.5, so under np > 1 the
    ranks own very different shares of it. A rank-local norm would therefore
    return a different number on each rank — which is exactly the regression this
    reduction was written to fix (issue #370: the rank owning the SolCx viscosity
    jump reported an error 10-20x its neighbours', so parallel tolerance
    assertions depended on the partition).

    A uniform perturbation cannot detect this: it would agree across ranks even
    if the norm were rank-local.
    """
    sol = Quadratic(mesh)

    velocity = uw.discretisation.MeshVariable("Ug", mesh, mesh.dim, degree=2)
    exact = np.asarray(sol.evaluate("velocity", velocity.coords)).reshape(
        -1, mesh.dim
    )

    perturbed = exact.copy()
    left = velocity.coords[:, 0] < 0.5
    perturbed[left] *= 1.5
    velocity.array[:, 0, :] = perturbed

    error = sol.error("velocity", velocity)

    assert error > 1.0e-3, "perturbation too small to discriminate"
    assert len(set(uw.mpi.comm.allgather(error))) == 1


def test_integral_norm_agrees_with_the_nodal_norm(mesh):
    """The continuous norm reports the same relative error for a uniform scaling.

    Scaling the field by (1 + eps) scales the error integrand uniformly, so both
    norms must return eps. They disagree in general — the integral norm is the
    one to use across discretisations — but this case pins both normalisations
    against the same known answer.
    """
    sol = Quadratic(mesh)

    velocity = uw.discretisation.MeshVariable("Ui", mesh, mesh.dim, degree=2)
    exact = np.asarray(sol.evaluate("velocity", velocity.coords)).reshape(
        -1, mesh.dim
    )

    eps = 1.0e-2
    velocity.array[:, 0, :] = exact * (1.0 + eps)

    assert np.isclose(sol.error("velocity", velocity, norm="integral"), eps, rtol=1e-4)


def test_unknown_norm_is_rejected(mesh):
    sol = Quadratic(mesh)
    velocity = uw.discretisation.MeshVariable("Un", mesh, mesh.dim, degree=2)

    with pytest.raises(ValueError, match="norm must be"):
        sol.error("velocity", velocity, norm="linf")


def test_missing_boundary_conditions_are_an_error(mesh):
    """A solution that declares no BCs says so, rather than solving something else."""

    class NoBoundaryConditions(uw.analytic.AnalyticSolution):
        dim = 2

    sol = NoBoundaryConditions(mesh)
    stokes = uw.systems.Stokes(mesh)

    with pytest.raises(NotImplementedError, match="does not declare its boundary"):
        sol.apply_boundary_conditions(stokes)


def test_prescribed_velocity_configures_the_solver(mesh):
    """The velocity helper imposes the exact velocity on the boundaries it is given."""
    sol = Quadratic(mesh)
    stokes = uw.systems.Stokes(mesh)

    sol.apply_boundary_conditions(stokes)

    assert stokes.petsc_use_pressure_nullspace
    assert {bc.boundary for bc in stokes.essential_bcs} == set(sol.boundaries)


def test_free_slip_uses_the_rotated_constraint(mesh):
    """free_slip imposes u.n = 0 by rotation, not by masking a component.

    Component masking is only equivalent on an axis-aligned box; the rotated form
    is what still holds when a solution is used to validate a curved or adapted
    mesh. The registration list is private because it has no public reader — this
    is the only signal that distinguishes the two paths.
    """

    sol = Quadratic(mesh)
    stokes = uw.systems.Stokes(mesh)

    uw.analytic.free_slip(stokes, sol.boundaries)

    registered = {boundary for boundary, _ in stokes._rotated_freeslip_bcs}
    assert registered == set(sol.boundaries)
    assert stokes.essential_bcs == []


def test_boundaries_can_carry_different_conditions(mesh):
    """The reason the helpers take a boundary list rather than reading self.

    A shell, an annulus or a channel is not "walls of one kind": the condition
    differs from one boundary to the next. A solution says so by calling more
    than one helper, each with the boundaries it applies to.
    """

    sol = Quadratic(mesh)
    stokes = uw.systems.Stokes(mesh)

    uw.analytic.free_slip(stokes, ["Left", "Right"])
    uw.analytic.prescribed_velocity(stokes, ["Bottom", "Top"], sol.fn_velocity)

    slipping = {boundary for boundary, _ in stokes._rotated_freeslip_bcs}
    driven = {bc.boundary for bc in stokes.essential_bcs}

    assert slipping == {"Left", "Right"}
    assert driven == {"Bottom", "Top"}


def test_free_slip_passes_an_analytic_normal_through(mesh):
    """A curved-boundary solution can choose the normal; otherwise it says nothing.

    The distinction is between passing `normal=None` and not passing `normal` at
    all: the first overrides the solver's default with nothing, the second leaves
    the solver to choose. So this records the *call*, rather than reading the
    value back off the solver — reading it back would only confirm what the
    solver's own default happens to be.
    """

    x, y = mesh.X
    radial = sympy.Matrix([[x, y]]) / sympy.sqrt(x**2 + y**2)

    class Recorder:
        def __init__(self):
            self.calls = []

        def add_rotated_freeslip_bc(self, value, boundary, **kwargs):
            self.calls.append((value, boundary, kwargs))

    solver = Recorder()
    uw.analytic.free_slip(solver, ["Left"])
    uw.analytic.free_slip(solver, ["Right"], normal=radial)

    assert solver.calls == [
        (0.0, "Left", {}),
        (0.0, "Right", {"normal": radial}),
    ]


def test_free_slip_reaches_a_real_solver(mesh):
    """The recorded call is the one a Stokes solver actually accepts."""

    stokes = uw.systems.Stokes(mesh)
    uw.analytic.free_slip(stokes, ["Left"])

    assert {boundary for boundary, _ in stokes._rotated_freeslip_bcs} == {"Left"}


def test_boundaries_follow_the_mesh_dimension(mesh):
    sol = Quadratic(mesh)
    assert sol.boundaries == ["Left", "Right", "Bottom", "Top"]
