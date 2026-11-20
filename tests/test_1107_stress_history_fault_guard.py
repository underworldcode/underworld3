"""A stress history must not trace across a split-node fault (#871).

A trajectory traced from a point within about h/2 of the slit can be located on
the other side of it, and the history read there belongs to the other block
(measured on a Maxwell thin sheet: a period-2-step slip-rate oscillation that the
grid history removes). Until the trace respects the fault, a Stokes solver on a
mesh with a split-node fault carries its viscoelastic stress history on the grid
("eulerian") by default and refuses the tracing flavours; "none" and the particle
history do not trace and stay available. The same checks run at np 2 in
tests/parallel/test_1107_stress_history_fault_guard_parallel.py.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

TIP_A = np.array([0.30, 0.45])
TIP_B = np.array([0.70, 0.55])


def faulted_box(cell_size=1 / 16):
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=cell_size, regular=False, qdegree=2)
    return base.add_fault(("Flt", np.vstack([TIP_A, TIP_B])))


def plain_box(cell_size=1 / 16):
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=cell_size, regular=False, qdegree=2)


def ve_stokes(mesh, transport=None, fault_bc=True):
    """A Maxwell Stokes solver in simple shear, with a frictionless fault if the mesh has one."""
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=0, continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    if transport is not None:
        stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1)
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 1.0
    cm.Parameters.dt_elastic = 0.1
    stokes.tolerance = 1e-6
    stokes.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
    if fault_bc and getattr(mesh, "_fault_point_pairs", None):
        stokes.add_fault_bc(0, "Flt")
    return stokes


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def test_a_faulted_mesh_carries_its_stress_on_the_grid_by_default():
    stokes = ve_stokes(faulted_box())
    assert stokes.stress_transport == "eulerian"
    history = stokes.Unknowns.DFDt
    assert type(history).__name__ == "EulerianSUPG"
    assert history.transport_on_update
    stokes.solve(timestep=0.1)
    stress = np.asarray(history.psi_star[0].data)
    assert np.isfinite(stress).all() and np.abs(stress).max() > 1.0e-3


@pytest.mark.parametrize("transport", ["semi_lagrangian", "integration_point", "forward"])
def test_a_tracing_history_is_refused_on_a_faulted_mesh(transport):
    with pytest.raises(ValueError, match="#871"):
        ve_stokes(faulted_box(), transport)


@pytest.mark.parametrize("transport", ["none", "eulerian", "lagrangian"])
def test_the_grid_histories_are_allowed_on_a_faulted_mesh(transport):
    stokes = ve_stokes(faulted_box(), transport)
    assert stokes.stress_transport == transport


def test_an_unfaulted_mesh_keeps_the_usual_default():
    stokes = ve_stokes(plain_box())
    assert stokes.stress_transport == "semi_lagrangian"
    assert type(stokes.Unknowns.DFDt).__name__ == "SemiLagrangian"


def test_the_getter_reports_the_grid_default_on_a_faulted_mesh_before_the_model():
    mesh = faulted_box()
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=0, continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    assert stokes.stress_transport == "eulerian"


def test_a_faulted_thin_sheet_refuses_a_tracing_history():
    """The guard is the stress-history mixin's, so a flat ThinSheet on a faulted
    mesh refuses a trace-back as Stokes does (its own default is the grid)."""
    mesh = faulted_box()
    sheet = uw.systems.ThinSheet(mesh)
    sheet.stress_transport = "semi_lagrangian"
    with pytest.raises(ValueError, match="#871"):
        sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
