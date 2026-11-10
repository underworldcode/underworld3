"""Timestep estimates read model units, and the timestep setters keep or refuse what
they are given (bundle B of the issue sweep).

- **#891.** In a model with reference scales ``mesh.X.coords`` is dimensional.
  The semi-Lagrangian solver's direction-aware crossing time and the Eulerian
  solver's first accuracy estimate read it as model units: the crossing time came
  out the length scale times too long (2.2e6 on this annulus), and the accuracy
  estimate read the velocity far outside the mesh (0.61 of the right step).
  With units on the variables it also took the field range and the velocity
  dimensional. A units model and its dimensionless twin now give the same
  estimates, with and without units on the variables.
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


def _annulus_estimates(units, variable_units=False):
    uw.reset_default_model()
    if units:
        uw.get_default_model().set_reference_quantities(
            shell_thickness=uw.quantity(2200, "km"),
            thermal_diffusivity=uw.quantity(1e-6, "m**2/s"),
            mantle_viscosity=uw.quantity(1e22, "Pa*s"),
            temperature_contrast=uw.quantity(2500, "K"))
    mesh = uw.meshing.Annulus(radiusInner=0.55, radiusOuter=1.0, cellSize=0.15, qdegree=3)
    v_units, T_units = ("cm/yr", "K") if variable_units else (None, None)
    v = uw.discretisation.MeshVariable("V1153", mesh, mesh.dim, degree=2, units=v_units)
    T = uw.discretisation.MeshVariable("T1153", mesh, 1, degree=2, units=T_units)
    cv, cT = np.asarray(v.coords_nd), np.asarray(T.coords_nd)
    v.data[:, 0] = -cv[:, 1]                                     # rigid rotation
    v.data[:, 1] = cv[:, 0]
    T.data[:, 0] = np.cos(3 * np.arctan2(cT[:, 1], cT[:, 0])) * (1 - cT[:, 0] ** 2)

    slcn = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v.sym)
    slcn.constitutive_model = uw.constitutive_models.DiffusionModel
    slcn.constitutive_model.Parameters.diffusivity = 1.0
    slcn.estimate_dt(direction_aware=True)

    T2 = uw.discretisation.MeshVariable("T1153e", mesh, 1, degree=2, units=T_units)
    T2.data[...] = T.data[...]
    eulerian = uw.systems.AdvDiffusion(mesh, u_Field=T2, V_fn=v.sym)
    eulerian.constitutive_model = uw.constitutive_models.DiffusionModel
    eulerian.constitutive_model.Parameters.diffusivity = 1.0
    eulerian.estimate_dt()
    first = float(eulerian.dt_accuracy)
    eulerian.solve(timestep=1.0e-3)
    eulerian.estimate_dt()
    estimates = (float(slcn.dt_adv), first, float(eulerian.dt_accuracy))
    uw.reset_default_model()
    return estimates


@pytest.mark.parametrize("variable_units", [False, True],
                         ids=["variables_without_units", "variables_with_units"])
def test_a_units_model_and_its_twin_estimate_the_same_step(variable_units):
    dimensionless = _annulus_estimates(units=False)
    with_units = _annulus_estimates(units=True, variable_units=variable_units)
    # development: the crossing time 2.2e6 times too long, the accuracy step 0.61;
    # with V in cm/yr and T in K the accuracy step read the velocity and the
    # field range dimensional, 5.5e15 times too long before a solve and 2500
    # (the temperature scale) times after one
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
@pytest.mark.parametrize("value", [np.array(0.03), np.array([0.03])],
                         ids=["zero_dimensional", "one_element"])
def test_a_single_element_array_is_a_number(build, value):
    uw.reset_default_model()
    solver = build()
    solver.delta_t = 0.02
    solver.is_setup = True
    solver._needs_function_rewire = False
    solver.delta_t = value
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


def _set_scales():
    uw.get_default_model().set_reference_quantities(
        shell_thickness=uw.quantity(2200, "km"),
        thermal_diffusivity=uw.quantity(1e-6, "m**2/s"),
        mantle_viscosity=uw.quantity(1e22, "Pa*s"),
        temperature_contrast=uw.quantity(2500, "K"))


@SOLVERS
def test_a_dimensionless_ratio_of_times_is_reduced(build):
    uw.reset_default_model()
    _set_scales()
    solver = build()
    time_scale = uw.get_default_model().fundamental_scales["time"]
    reference = float(uw.quantity(600, "kyr").to("s").magnitude / time_scale.to("s").magnitude)
    # an unreduced ratio: 600 kyr over the time scale in its own units, whose
    # unreduced magnitude is 9.5e-13
    solver.delta_t = uw.quantity(600, "kyr") / time_scale
    assert float(solver.delta_t.sym) == pytest.approx(reference, rel=1.0e-12)
    # (600 kyr) / (20 Myr) is 0.03, not 30
    solver.delta_t = uw.quantity(600, "kyr") / uw.quantity(20, "Myr")
    assert float(solver.delta_t.sym) == pytest.approx(0.03, rel=1.0e-12)


@SOLVERS
def test_a_time_valued_expression_needs_reference_scales(build):
    uw.reset_default_model()
    solver = build()
    E = uw.expression(r"\Delta t_{1153q}", uw.quantity(1, "kyr"), "a timestep in kyr")
    # a bare 1 kyr is refused (above); in an expression it was taken as dt = 1.0
    with pytest.raises(ValueError):
        solver.delta_t = E
