"""Transporting a stress history on the grid instead of along characteristics.

A viscoelastic solve carries a stress that is not its own unknown. The history
manager owns that transport: the semi-Lagrangian flavour traces it back along
characteristics, and ``EulerianSUPG`` with ``transport_on_update`` assembles it
implicitly with the same SUPG stabilisation the solvers use. These are the
transport tests the existing viscoelastic benchmarks cannot give us, because
those are all spatially uniform and so transport nothing.

Run: pixi run python -m pytest tests/test_1059_stress_transport.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

COMPONENTS = ((0, 0), (0, 1), (1, 1))
AMPLITUDES = (1.0, 0.5, -1.0)


def _blob(x, y, x0, y0=0.0, width=0.04):
    return sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2) / width)


def _plant(var, expression):
    """Set every independent component of a symmetric tensor from one shape."""
    values = uw.function.evaluate(expression, var.coords).reshape(-1)
    for (i, j), amplitude in zip(COMPONENTS, AMPLITUDES):
        var.array[:, i, j] = amplitude * values
        if i != j:
            var.array[:, j, i] = amplitude * values
    return values


def _error(var, expression):
    """Relative L2 error of the tensor against one shape times the amplitudes."""
    exact = uw.function.evaluate(expression, var.coords).reshape(-1)
    numerator = denominator = 0.0
    for (i, j), amplitude in zip(COMPONENTS, AMPLITUDES):
        target = amplitude * exact
        numerator += float(np.sum((np.asarray(var.array[:, i, j]) - target) ** 2))
        denominator += float(np.sum(target ** 2))
    return np.sqrt(numerator / denominator)


def _grid_manager(mesh, tag, velocity, order=1):
    """A stress variable and the Eulerian-SUPG history that transports it."""
    stress = uw.discretisation.MeshVariable(
        f"S_{tag}", mesh, vtype=uw.VarType.SYM_TENSOR, degree=2)
    return stress, uw.systems.ddt.EulerianSUPG(
        mesh, stress, velocity, vtype=uw.VarType.SYM_TENSOR, degree=2,
        continuous=True, order=order, transport_on_update=True)


def _traced_manager(mesh, tag, velocity, order=1):
    """The same, with the semi-Lagrangian trace-back carrying the stress.

    Each manager needs its OWN stress variable: they read the field back on
    every step, so sharing one couples the two transports.
    """
    stress = uw.discretisation.MeshVariable(
        f"S_{tag}", mesh, vtype=uw.VarType.SYM_TENSOR, degree=2)
    return stress, uw.systems.ddt.SemiLagrangian(
        mesh, stress.sym, velocity, vtype=uw.VarType.SYM_TENSOR, degree=2,
        continuous=True, order=order, varsymbol=rf"S^{{{tag}}}")


def _march(stress, manager, dt, steps):
    """The step a solver takes: carry the history, then commit the new value.

    With no constitutive update the commit is the identity, which is what
    makes this a pure transport comparison.
    """
    for _ in range(steps):
        manager.update_pre_solve(dt)
        stress.array[...] = manager.psi_star[0].array[...]


def test_a_stress_blob_is_carried_by_a_uniform_flow():
    """Uniform translation has an exact answer: the blob arrives where the flow
    puts it, with its components unchanged (no rotation in a uniform flow)."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 24, qdegree=3)
    x, y = mesh.X
    speed, dt, steps, start = 0.5, 0.05, 8, -0.5
    velocity = sympy.Matrix([[speed, 0.0]])
    grid_stress, eulerian = _grid_manager(mesh, "uniform_g", velocity)
    traced_stress, lagrangian = _traced_manager(mesh, "uniform_s", velocity)
    assert eulerian.transport_on_update and eulerian._advection_mode == "assembled"

    for stress, manager in ((grid_stress, eulerian), (traced_stress, lagrangian)):
        _plant(stress, _blob(x, y, start))
        manager.initialise_history()
        _march(stress, manager, dt, steps)

    exact = _blob(x, y, start + speed * dt * steps)
    grid = _error(eulerian.psi_star[0], exact)
    traced = _error(lagrangian.psi_star[0], exact)
    # Uniform translation is the trace-back's best case (the departure point is
    # exact), so it sets the bar; the grid transport must be of the same order.
    assert traced < 0.02, traced
    assert grid < 0.05, grid
    # negative control: the field really moved
    assert _error(eulerian.psi_star[0], _blob(x, y, start)) > 0.5
    # symmetry is preserved component by component
    carried = np.asarray(eulerian.psi_star[0].array)
    assert np.allclose(carried[:, 0, 1], carried[:, 1, 0])


def test_the_components_ride_round_a_rigid_rotation_unchanged():
    """With no rotation terms in the transport the tensor components are
    advected as scalars: after a quarter turn each component sits where the
    flow carried it, with the same amplitude."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 24, qdegree=3)
    x, y = mesh.X
    stress, eulerian = _grid_manager(mesh, "rot", sympy.Matrix([[-y, x]]))

    radius, dt, steps = 0.5, np.pi / 2 / 40, 40      # a quarter turn
    _plant(stress, _blob(x, y, radius))
    eulerian.initialise_history()
    _march(stress, eulerian, dt, steps)

    turned = _blob(x, y, 0.0, radius)                 # a quarter turn from (r, 0)
    assert _error(eulerian.psi_star[0], turned) < 0.2
    peak = float(np.abs(np.asarray(eulerian.psi_star[0].array[:, 0, 0])).max())
    assert 0.6 < peak < 1.05, peak                    # amplitude carried, not created


def test_transport_is_off_unless_asked_for():
    """A manager built for a solver's own unknown assembles its advection in
    that solver's residual and must not also move its history."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T_off", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(_blob(x, y, 0.3), T.coords).reshape(-1)
    manager = uw.systems.ddt.EulerianSUPG(
        mesh, T, sympy.Matrix([[1.0, 0.0]]), vtype=uw.VarType.SCALAR,
        degree=2, continuous=True)
    manager.initialise_history()
    before = np.array(manager.psi_star[0].array)
    manager.update_pre_solve(0.05)
    assert manager.transport_on_update is False
    assert np.array_equal(np.asarray(manager.psi_star[0].array), before)
    assert manager._transport_solver is None


def _maxwell_shear(transport, order, steps=20, dt=0.1):
    """The analytic Maxwell shear box, with the stress history of one's choosing.

    Simple shear of a Maxwell material: sigma_xy = eta gammadot (1 - exp(-t/t_r)).
    The stress is spatially uniform, so its transport is a no-op and the two
    flavours must agree exactly. That is what makes this the correctness check
    on the plumbing rather than on the transport.
    """
    eta = shear_modulus = 1.0
    speed, height, width = 0.5, 1.0, 2.0
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-width / 2, -height / 2),
        maxCoords=(width / 2, height / 2))
    v = uw.discretisation.MeshVariable(f"U_{transport}{order}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{transport}{order}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=False)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=order)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.constitutive_model.Parameters.shear_modulus = shear_modulus
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((speed, 0.0), "Top")
    stokes.add_dirichlet_bc((-speed, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False, evalf=False)

    stress = float(np.asarray(uw.function.evaluate(
        stokes.DFDt.psi_star[0].sym[0, 1], np.array([[0.0, 0.0]]))).reshape(-1)[0])
    rate = 2.0 * speed / height
    exact = eta * rate * (1.0 - np.exp(-steps * dt * shear_modulus / eta))
    return type(stokes.DFDt).__name__, stress, exact


@pytest.mark.parametrize("order, tolerance", [(1, 0.02), (2, 0.002)])
def test_either_stress_history_solves_the_maxwell_shear_box(order, tolerance):
    """A Stokes solve carries its viscoelastic stress with the history its
    `stress_transport` names, and on a uniform stress the two agree exactly."""
    traced_kind, traced, exact = _maxwell_shear("semi_lagrangian", order)
    grid_kind, grid, _ = _maxwell_shear("eulerian", order)
    assert traced_kind == "SemiLagrangian" and grid_kind == "EulerianSUPG"
    assert abs(traced - exact) / exact < tolerance, (traced, exact)
    assert abs(grid - exact) / exact < tolerance, (grid, exact)
    # transport is a no-op on a uniform field: the plumbing must not add anything
    assert abs(grid - traced) < 1e-6 * abs(exact), (grid, traced)


def test_stress_transport_is_validated_and_fixed_once_the_history_exists():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("U_val", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P_val", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    with pytest.raises(ValueError, match="stress_transport must be"):
        stokes.stress_transport = "lagrangian"
    stokes.stress_transport = "eulerian"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.dt_elastic = 0.1
    assert type(stokes.DFDt).__name__ == "EulerianSUPG"
    with pytest.raises(RuntimeError, match="already exists"):
        stokes.stress_transport = "semi_lagrangian"


def _sheared_varying_modulus(transport, order, steps=10, dt=0.1, res=6):
    """Simple shear of a Maxwell material whose shear modulus varies in x.

    The stress is then non-uniform and the shear carries it, so the transport
    term is genuinely active: the case the uniform benchmarks cannot provide.
    There is no closed form, so the schemes are judged against each other and
    against their own behaviour under refinement.
    """
    eta, modulus, speed, height, width = 1.0, 1.0, 0.5, 1.0, 2.0
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(2 * res, res), minCoords=(-width / 2, -height / 2),
        maxCoords=(width / 2, height / 2))
    x, _y = mesh.X
    tag = f"{transport[0]}{order}{steps}"
    v = uw.discretisation.MeshVariable(f"Uv_{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"Pv_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=False)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=order)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.constitutive_model.Parameters.shear_modulus = (
        modulus * (1 + 0.5 * sympy.sin(2 * sympy.pi * x / width)))
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((speed, 0.0), "Top")
    stokes.add_dirichlet_bc((-speed, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False, evalf=False)

    shear = stokes.DFDt.psi_star[0].sym[0, 1]
    norm = float(np.sqrt(uw.maths.Integral(mesh, shear ** 2).evaluate()))
    slope = float(np.sqrt(uw.maths.Integral(mesh, shear.diff(x) ** 2).evaluate()))
    return norm, slope


def test_the_two_stress_histories_agree_when_the_stress_moves_and_evolves():
    """With a stress that is carried as well as relaxed the schemes must agree
    to within their own time-discretisation error, and more tightly at order 2."""
    traced, traced_slope = _sheared_varying_modulus("semi_lagrangian", 2)
    grid, grid_slope = _sheared_varying_modulus("eulerian", 2)
    assert traced_slope > 0.1 and grid_slope > 0.1, "the stress must not be uniform"
    assert abs(grid - traced) / traced < 5.0e-3, (grid, traced)

    # halving the step moves each scheme by no more than they differ from
    # each other: the gap between them is discretisation, not a defect
    refined, _ = _sheared_varying_modulus("eulerian", 2, steps=20, dt=0.05)
    assert abs(refined - grid) / grid < 5.0e-3, (refined, grid)


def test_navier_stokes_carries_a_viscoelastic_stress_either_way():
    """With inertia the momentum solve takes several passes over a step, so the
    stress history must advance once per step, not once per pass. Both flavours
    must give the same answer; the shear box with inertia has no closed form
    (the velocity is far from steady simple shear at this effective viscosity),
    so they are judged against each other."""
    def run(transport):
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(16, 8), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5))
        v = uw.discretisation.MeshVariable(f"Un_{transport[0]}", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable(f"Pn_{transport[0]}", mesh, 1, degree=1)
        ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=2)
        ns.stress_transport = transport
        ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            ns.Unknowns, order=2)
        ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        ns.constitutive_model.Parameters.shear_modulus = 1.0
        ns.constitutive_model.Parameters.dt_elastic = 0.1
        ns.add_dirichlet_bc((0.5, 0.0), "Top")
        ns.add_dirichlet_bc((-0.5, 0.0), "Bottom")
        ns.add_dirichlet_bc((sympy.oo, 0.0), "Left")
        ns.add_dirichlet_bc((sympy.oo, 0.0), "Right")
        ns.bodyforce = sympy.Matrix([[0.0, 0.0]])
        for _ in range(10):
            ns.solve(timestep=0.1)
        return type(ns.DFDt).__name__, float(np.asarray(uw.function.evaluate(
            ns.DFDt.psi_star[0].sym[0, 1], np.array([[0.0, 0.0]]))).reshape(-1)[0])

    traced_kind, traced = run("semi_lagrangian")
    grid_kind, grid = run("eulerian")
    assert traced_kind == "SemiLagrangian" and grid_kind == "EulerianSUPG"
    assert traced > 0.1 and abs(grid - traced) / traced < 1.0e-3, (grid, traced)


def test_a_viscoelastic_navier_stokes_step_refuses_the_theta_rule():
    """At order 1 the theta rule weights the viscous flux at the stored velocity
    levels, which rebuilds them blind to elasticity."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Uo", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Po", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
    ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        ns.Unknowns, order=1)
    ns.constitutive_model.Parameters.shear_modulus = 1.0
    ns.constitutive_model.Parameters.dt_elastic = 0.1
    ns.bodyforce = sympy.Matrix([[0.0, 0.0]])
    for boundary in ("Top", "Bottom", "Left", "Right"):
        ns.add_dirichlet_bc((0.0, 0.0), boundary)
    with pytest.raises(ValueError, match="needs order=2"):
        ns.solve(timestep=0.1)
