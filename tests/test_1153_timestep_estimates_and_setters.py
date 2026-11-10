"""Timestep estimates read model units, and the timestep setters keep or refuse what
they are given (bundle B of the issue sweep).

- **#891.** In a model with reference scales ``mesh.X.coords`` is dimensional.
  The semi-Lagrangian solver's direction-aware crossing time and the Eulerian
  solver's first accuracy estimate read it as model units: the crossing time came
  out the length scale times too long (2.2e6 on this annulus), and the accuracy
  estimate read the velocity far outside the mesh (0.61 of the right step). A
  units model and its dimensionless twin now give the same estimates.
- **#978.** On a mesh of degree 2 the accuracy estimate evaluated the velocity at
  the vertices while the gradient recovery returns the coordinate DM's nodes,
  midpoints included, and raised ``ValueError``.
- **#1031.** The timestep setters (``TransientDarcy`` and ``Richards``,
  ``NavierStokesSLCN``, the semi-Lagrangian and Eulerian solvers, ``Diffusion``)
  share ``_nondimensionalise_timestep``. ``TransientDarcy`` unwrapped a
  ``UWexpression``, freezing it; a length was taken for a timestep; a time with no
  reference scales was taken at its magnitude; a 0-d array rewired the kernels on
  every step.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _annulus_estimates(units):
    uw.reset_default_model()
    if units:
        uw.get_default_model().set_reference_quantities(
            shell_thickness=uw.quantity(2200, "km"),
            thermal_diffusivity=uw.quantity(1e-6, "m**2/s"),
            mantle_viscosity=uw.quantity(1e22, "Pa*s"),
            temperature_contrast=uw.quantity(2500, "K"))
    mesh = uw.meshing.Annulus(radiusInner=0.55, radiusOuter=1.0, cellSize=0.15, qdegree=3)
    v = uw.discretisation.MeshVariable("V1153", mesh, mesh.dim, degree=2)
    T = uw.discretisation.MeshVariable("T1153", mesh, 1, degree=2)
    cv, cT = np.asarray(v.coords_nd), np.asarray(T.coords_nd)
    v.data[:, 0] = -cv[:, 1]                                     # rigid rotation
    v.data[:, 1] = cv[:, 0]
    T.data[:, 0] = np.cos(3 * np.arctan2(cT[:, 1], cT[:, 0])) * (1 - cT[:, 0] ** 2)

    slcn = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v.sym)
    slcn.constitutive_model = uw.constitutive_models.DiffusionModel
    slcn.constitutive_model.Parameters.diffusivity = 1.0
    slcn.estimate_dt(direction_aware=True)

    T2 = uw.discretisation.MeshVariable("T1153e", mesh, 1, degree=2)
    T2.data[...] = T.data[...]
    eulerian = uw.systems.AdvDiffusion(mesh, u_Field=T2, V_fn=v.sym)
    eulerian.constitutive_model = uw.constitutive_models.DiffusionModel
    eulerian.constitutive_model.Parameters.diffusivity = 1.0
    eulerian.estimate_dt()
    estimates = (float(slcn.dt_adv), float(eulerian.dt_accuracy))
    uw.reset_default_model()
    return estimates


def test_a_units_model_and_its_twin_estimate_the_same_step():
    dimensionless = _annulus_estimates(units=False)
    with_units = _annulus_estimates(units=True)
    # development: the crossing time 2.2e6 times too long, the accuracy step 0.61
    assert with_units == pytest.approx(dimensionless, rel=1.0e-9)


def test_the_accuracy_estimate_works_on_a_degree_2_mesh():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.125, degree=2, qdegree=3)
    v = uw.discretisation.MeshVariable("V1153d", mesh, 2, degree=2)
    T = uw.discretisation.MeshVariable("T1153d", mesh, 1, degree=2)
    v.data[:, 0] = 1.0
    T.data[:, 0] = np.sin(np.pi * np.asarray(T.coords)[:, 0])
    adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v.sym)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    # development: ValueError, 714 gradient values into the 98 vertices
    dt = adv.estimate_dt()
    # u.grad(T) peaks at pi on [0, 1]; the range is 1: dt = 0.02 / pi within the
    # recovery's O(h) error
    assert 0.5 * 0.02 / np.pi < dt < 2.0 * 0.02 / np.pi


def _darcy():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    h = uw.discretisation.MeshVariable("H1153", mesh, 1, degree=2)
    v = uw.discretisation.MeshVariable("U1153", mesh, 2, degree=1)
    return uw.systems.TransientDarcy(mesh, h_Field=h, v_Field=v)


def _navier_stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    v = uw.discretisation.MeshVariable("U1153n", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1153n", mesh, 1, degree=1)
    return uw.systems.NavierStokesSLCN(mesh, velocityField=v, pressureField=p, rho=1.0)


SOLVERS = pytest.mark.parametrize("build", [_darcy, _navier_stokes],
                                  ids=["transient_darcy", "navier_stokes_slcn"])


@SOLVERS
def test_an_expression_timestep_stays_live(build):
    uw.reset_default_model()
    solver = build()
    E = uw.expression(r"\Delta t_{1153}", 0.03, "a timestep held in an expression")
    solver.delta_t = E
    # development (TransientDarcy): unwrapped to Float 0.03, so E.sym = 0.05 was lost
    assert solver.delta_t.sym is E
    E.sym = 0.05
    assert float(sympy.sympify(solver.delta_t.sym.sym)) == 0.05


@SOLVERS
def test_a_zero_dimensional_array_is_a_number(build):
    uw.reset_default_model()
    solver = build()
    solver.delta_t = 0.02
    solver.is_setup = True
    solver._needs_function_rewire = False
    solver.delta_t = np.array(0.03)
    # development: an ImmutableDenseNDimArray, and a rewire on every step
    assert not solver._needs_function_rewire
    assert float(solver.delta_t.sym) == 0.03


@SOLVERS
@pytest.mark.parametrize("value", [uw.quantity(5, "m"), uw.quantity(600, "kyr")],
                         ids=["a_length", "a_time_with_no_reference_scales"])
def test_a_timestep_the_model_cannot_read_is_refused(build, value):
    uw.reset_default_model()
    solver = build()
    # development: 5 (a length) and 600 (kyr, as if model units) were accepted
    with pytest.raises(ValueError):
        solver.delta_t = value
