"""The installation auditor, and a second Stokes solver over fields the first already owns.

Both subjects used to sit at module level, so importing this file built a mesh,
two solvers and solved both — during pytest COLLECTION, before any test ran.
A `--collect-only` run was measured 20+ minutes inside `SNESSolve`, which to the
caller is indistinguishable from pytest dying silently (#505). The auditor
assertion was disabled at the same time by the name `dont_test_auditor`, so the
file did all of that work and checked nothing.

The auditor check is now written as a delta rather than the absolute
`uw_object_count == 7` it used to assert. The counter is process-wide and
monotonic, so an absolute count is only true in a fresh process running this
file alone — which is why the assertion could not survive being enabled.
"""

import pytest
import sympy
import numpy as np

import underworld3 as uw

pytestmark = pytest.mark.level_1


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(elementRes=(5,) * 2)


def test_auditor_reads_the_whole_installation(mesh):
    """Every installation field the auditor advertises is populated.

    The auditor sets a field to None and warns when it cannot import the
    package behind it, so a None here is a real gap in what we can report.
    """

    unreadable = [k for k, v in uw.auditor.get_installation_data.items() if v is None]

    assert not unreadable, f"auditor could not read: {unreadable}"


def test_auditor_counts_objects_as_they_are_created(mesh):
    """The runtime count advances when objects are built, and only then."""

    before = uw.auditor.get_runtime_data["uw_object_count"]

    scratch = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    uw.discretisation.MeshVariable("audited", scratch, 1, degree=1)

    after = uw.auditor.get_runtime_data["uw_object_count"]

    # At least the two objects named above; the constructors may build more.
    assert after - before >= 2

    # The control: reading the auditor is not itself what moves the count.
    assert uw.auditor.get_runtime_data["uw_object_count"] == after


def test_second_solver_over_the_same_fields(mesh):
    """Two Stokes solvers share one velocity/pressure pair and give different flows.

    `sympy.oo` in a component slot leaves that component unconstrained, so the
    two boundary condition sets below constrain different components on
    different walls. The assertion is that the second solve reaches its own
    answer rather than returning the first solver's field: the two are set up
    over the same `MeshVariable`s, which is the situation where a stale
    setup would go unnoticed.
    """

    x, y = mesh.X
    v = uw.discretisation.MeshVariable(
        r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
    )
    p = uw.discretisation.MeshVariable(
        r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )

    first = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    first.constitutive_model = uw.constitutive_models.ViscousFlowModel
    first.constitutive_model.Parameters.shear_viscosity_0 = 1
    first.add_dirichlet_bc((0.0, 0.0), "Bottom")
    first.add_dirichlet_bc((y, 0.0), "Top")
    first.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    first.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    first.solve()

    driven_from_the_top = v.data.copy()

    second = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    second.constitutive_model = uw.constitutive_models.ViscousFlowModel
    second.constitutive_model.Parameters.shear_viscosity_0 = 1
    second.add_dirichlet_bc((0.0, sympy.oo), "Bottom")
    second.add_dirichlet_bc((0.0, sympy.oo), "Top")
    second.add_dirichlet_bc((0.0, 0.0), "Left")
    second.add_dirichlet_bc((0.0, x), "Right")
    second.solve()

    driven_from_the_side = v.data.copy()

    assert np.isfinite(driven_from_the_top).all()
    assert np.isfinite(driven_from_the_side).all()
    assert abs(driven_from_the_top).max() > 0.0
    assert not np.allclose(driven_from_the_top, driven_from_the_side)
