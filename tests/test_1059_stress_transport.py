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


def _maxwell_shear(transport, order, steps=20, dt=0.1, integrator="bdf", solver="stokes", initial_velocity=False,
                   objective_rate="none", solvent=0.0):
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
    if solver == "stokes":
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=False)
    else:
        # The trace-back Navier-Stokes solver at negligible inertia: the same box.
        stokes = uw.systems.NavierStokesSLCN(mesh, v, p, rho=1.0e-6, order=1)
        stokes.bodyforce = sympy.Matrix([[0.0, 0.0]])
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=order, integrator=integrator, objective_rate=objective_rate)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
    stokes.constitutive_model.Parameters.shear_modulus = shear_modulus
    stokes.constitutive_model.Parameters.solvent_viscosity = solvent
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((speed, 0.0), "Top")
    stokes.add_dirichlet_bc((-speed, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"

    if initial_velocity:
        # The steady shear profile, already in place before the first solve.
        v.array[:, 0, :] = uw.function.evaluate(
            sympy.Matrix([[2.0 * speed * mesh.X[1] / height, 0.0]]), v.coords).reshape(-1, 2)
    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False)

    # After a solve the Stokes family has committed the new stress into the
    # history's first level; the trace-back Navier-Stokes solver records it at
    # the NEXT carry, so there its first level still holds the previous step
    # and the stress just solved for is the constitutive flux (#742).
    latest = stokes.DFDt.psi_star[0].sym if solver == "stokes" else stokes.constitutive_model.flux
    origin = np.array([[0.0, 0.0]])
    stress = float(np.asarray(uw.function.evaluate(latest[0, 1], origin)).reshape(-1)[0])
    rate = 2.0 * speed / height
    exact = eta * rate * (1.0 - np.exp(-steps * dt * shear_modulus / eta))
    if objective_rate != "none" or solvent:
        n1 = float(np.asarray(uw.function.evaluate(latest[0, 0] - latest[1, 1], origin)).reshape(-1)[0])
        # the momentum flux and the polymer part of it, both formed on the
        # just-committed level (one step ahead of `latest`, see #742): their
        # difference is the solvent's stress alone
        total_xy = float(np.asarray(uw.function.evaluate(stokes.constitutive_model.flux[0, 1], origin)).reshape(-1)[0])
        polymer_flux_xy = float(np.asarray(uw.function.evaluate(stokes.constitutive_model.history_flux[0, 1], origin)).reshape(-1)[0])
        return type(stokes.DFDt).__name__, stress, exact, n1, total_xy - polymer_flux_xy
    return type(stokes.DFDt).__name__, stress, exact


KINDS = {
    "semi_lagrangian": "SemiLagrangian",
    "integration_point": "IntegrationPointSemiLagrangian",
    "eulerian": "EulerianSUPG",
}


@pytest.mark.parametrize("order, integrator, tolerance",
                         [(1, "bdf", 0.02), (2, "bdf", 0.002), (1, "etd", 1e-4), (2, "etd", 0.01)])
def test_every_stress_history_solves_the_maxwell_shear_box(order, integrator, tolerance):
    """A Stokes solve carries its viscoelastic stress with the history its
    `stress_transport` names, and on a uniform stress all three agree exactly.

    The analytic tolerance is what catches a history that is rebuilt from its
    own flux instead of carried: that applies the constitutive update twice a
    step and lands at 12.8% on this box at order 1 (#732). The exponential
    integrator is exact for a constant strain rate, and it must run on every
    flavour, not only the nodal one (#739).
    """
    results = {}
    for transport, expected_kind in KINDS.items():
        if integrator == "etd" and order == 2 and transport == "eulerian":
            continue        # the grid flavour has no forcing-history slot yet
        kind, stress, exact = _maxwell_shear(transport, order, integrator=integrator)
        assert kind == expected_kind
        assert abs(stress - exact) / exact < tolerance, (transport, stress, exact)
        results[transport] = stress
    # Transport is a no-op on a uniform field, so the flavours differ only in
    # how they carry it: the plumbing must not add anything of its own.
    spread = max(results.values()) - min(results.values())
    assert spread < 1e-6 * abs(exact), results


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


def test_the_theta_rule_takes_its_stored_flux_from_the_stress_history():
    """Crank-Nicolson weights the momentum flux across time levels. For a
    viscous fluid the stored level is rebuilt from the stored velocity; for a
    viscoelastic one that rebuild is wrong, and the stress history already holds
    exactly what the theta rule is asking for. Check the flux reads it."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Uf", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pf", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    viscous_flux = ns._viscous_flux()

    ve = uw.systems.NavierStokes(
        mesh, uw.discretisation.MeshVariable("Ug", mesh, mesh.dim, degree=2),
        uw.discretisation.MeshVariable("Pg", mesh, 1, degree=1), rho=1.0, order=1)
    ve.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        ve.Unknowns, order=1)
    ve.constitutive_model.Parameters.shear_modulus = 1.0
    ve.constitutive_model.Parameters.dt_elastic = 0.1
    assert ve.integrator == "am", "order 1 is the theta rule"
    stored = set(sympy.Matrix(ve.DFDt.psi_star[0].sym).atoms(sympy.Function))
    assert stored & set(ve._viscous_flux().atoms(sympy.Function)), \
        "the theta rule must read the stress history at the stored level"
    assert not (stored & set(viscous_flux.atoms(sympy.Function))), \
        "a viscous fluid has no stress history to read"


def test_crank_nicolson_carries_a_viscoelastic_stress_either_way():
    """The scheme the Navier-Stokes benchmarks use, order 1, with a stress
    history: both transports must agree."""
    def run(transport):
        mesh = uw.meshing.StructuredQuadBox(
            elementRes=(16, 8), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5))
        v = uw.discretisation.MeshVariable(f"Uc_{transport[0]}", mesh, mesh.dim, degree=2)
        p = uw.discretisation.MeshVariable(f"Pc_{transport[0]}", mesh, 1, degree=1)
        ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
        ns.stress_transport = transport
        ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            ns.Unknowns, order=1)
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
        return float(np.asarray(uw.function.evaluate(
            ns.DFDt.psi_star[0].sym[0, 1], np.array([[0.0, 0.0]]))).reshape(-1)[0])

    traced, grid = run("semi_lagrangian"), run("eulerian")
    assert traced > 0.1 and abs(grid - traced) / traced < 1.0e-3, (grid, traced)


def test_devss_is_off_by_default_and_vanishes_on_a_uniform_strain_rate():
    """DEVSS adds 2 eta_a (edot - D) to the momentum flux with D the projected
    strain rate. On the uniform Maxwell shear box D equals edot exactly, so the
    term must vanish and the answer must not move for any eta_a; and it must be
    off unless asked for. The varying-modulus box is the negative control: there
    D differs from edot by projection error, the term is live, and the answer
    must move -- by projection error, which is small, but not by nothing."""
    _kind, off, exact = _maxwell_shear("integration_point", 1)
    assert abs(off - exact) / exact < 0.02

    def with_devss(builder, *args, **kw):
        # the helpers build their own solver; re-run them with the term on by
        # patching the class default for the duration of the call
        original = uw.systems.Stokes.__init__
        def patched(self, *a, **k):
            original(self, *a, **k)
            self.devss_viscosity = 1.0
        uw.systems.Stokes.__init__ = patched
        try:
            return builder(*args, **kw)
        finally:
            uw.systems.Stokes.__init__ = original

    _kind, on, _ = with_devss(_maxwell_shear, "integration_point", 1)
    assert abs(on - off) < 1e-8 * abs(exact), (on, off)      # the pair cancelled

    # the varying-modulus helper differentiates the history in a weak form,
    # which an integration-point variable refuses; the term is on the solver
    # and flavour-independent, so the nodal history serves for this half
    norm_off, _ = _sheared_varying_modulus("semi_lagrangian", 1)
    norm_on, _ = with_devss(_sheared_varying_modulus, "semi_lagrangian", 1)
    moved = abs(norm_on - norm_off) / norm_off
    assert 1e-6 < moved < 5e-2, moved                        # live, and only projection-sized


def test_the_exponential_integrator_runs_on_the_trace_back_navier_stokes():
    """The trace-back Navier-Stokes solver has its own history path. It must
    tell a viscoelastic model the step and refresh the integrator coefficients
    as the Stokes family does; it did neither, so the memory term was absent
    and the exponential integrator ran in its viscous limit (#741)."""
    for integrator, tolerance in (("etd", 1e-3), ("bdf", 0.02)):
        _, stress, exact = _maxwell_shear("semi_lagrangian", 1, integrator=integrator, solver="ns_slcn")
        assert abs(stress - exact) / exact < tolerance, (integrator, stress, exact)


def test_a_preset_velocity_gives_both_integrators_the_same_first_stress():
    """A trace-back history initialises its first level from the constitutive
    flux of the velocity it finds. With a velocity already in place that flux
    must be formed with the integrator's coefficients for this step -- the
    exponential one read alpha = phi = 0 and recorded the full viscous stress,
    seven times the BDF value on the cylinder (#740). One step: the two
    first-order integrators agree to O(dt / t_r)."""
    _, bdf, _ = _maxwell_shear("semi_lagrangian", 1, steps=1, integrator="bdf", initial_velocity=True)
    _, etd, _ = _maxwell_shear("semi_lagrangian", 1, steps=1, integrator="etd", initial_velocity=True)
    assert abs(etd - bdf) < 0.05 * abs(bdf), (etd, bdf)


def test_the_integration_point_history_takes_the_inflow_value_at_an_inlet():
    """A departure point that leaves through the inlet is restored to the
    boundary by the trace. Left to sample the boundary edge, the
    integration-point history on a box channel grew a mode in the inlet cell
    column at Wi 1 and 5 (#745); given an inflow value it takes that instead.
    Uniform flow into a box carrying zero stress: after k steps the inflow
    value has entered a distance k * speed * dt, and no further."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 24, qdegree=3)
    speed, dt, steps = 0.5, 0.05, 8
    velocity = sympy.Matrix([[speed, 0.0]])
    incoming = sympy.Matrix([[1.0, 0.25], [0.25, -1.0]])
    results = {}
    for tag, inflow in (("with", incoming), ("without", None)):
        manager = uw.systems.ddt.IntegrationPointSemiLagrangian(
            mesh, sympy.Matrix.zeros(2, 2), velocity, vtype=uw.VarType.SYM_TENSOR,
            degree=1, continuous=True, order=1, varsymbol=rf"S^{{{tag}}}")
        assert manager.applies_inflow_value
        if inflow is not None:
            manager.inflow_value = inflow
        manager.initialise_history()
        for _ in range(steps):
            manager.update_pre_solve(dt, store_result=False)
            manager.commit_flux_to_history(manager.psi_star[0].sym)   # identity commit
        points = np.asarray(manager.psi_star[0].integration_points).reshape(-1, 2)
        xy = np.asarray(manager.psi_star[0].data)[:, manager._components.index((0, 1))]
        entered = points[:, 0] < -1.0 + 0.6 * speed * dt * steps
        untouched = points[:, 0] > -1.0 + 1.5 * speed * dt * steps
        results[tag] = (xy[entered], xy[untouched])
    with_in, with_out = results["with"]
    assert np.allclose(with_in, 0.25, atol=0.02), (with_in.min(), with_in.max())
    assert np.abs(with_out).max() < 0.02
    # negative control: sampling the edge of a zero field brings nothing in
    without_in, _ = results["without"]
    assert np.abs(without_in).max() < 1e-6




@pytest.mark.parametrize("transport", ["semi_lagrangian", "integration_point"])
def test_the_upper_convected_element_builds_the_first_normal_stress_in_shear(transport):
    """Start-up of simple shear for the UCM fluid has the closed form
    sigma_xy = eta gdot (1 - e^{-t/lambda}) and
    N1 = sigma_xx - sigma_yy = 2 eta lambda gdot^2 (1 - e^{-t/lambda}(1 + t/lambda)).
    The passive Maxwell element makes no normal stress at all; the objective
    rate is what makes it. First order in time on the stretching term, so the
    tolerance is loose; the shear stress is unchanged by the term."""
    eta = lam = 1.0; gdot = 1.0; steps, dt = 20, 0.1
    t = steps * dt
    _, xy, exact_xy, n1, _ = _maxwell_shear(transport, 1, steps=steps, dt=dt, objective_rate="upper_convected")
    n1_exact = 2 * eta * lam * gdot ** 2 * (1 - np.exp(-t / lam) * (1 + t / lam))
    assert abs(xy - exact_xy) / exact_xy < 0.02, (xy, exact_xy)
    assert abs(n1 - n1_exact) / n1_exact < 0.10, (n1, n1_exact)
    _, _, _, n1_passive, _ = _maxwell_shear(transport, 1, steps=steps, dt=dt, objective_rate="none", solvent=1e-12)
    assert abs(n1_passive) < 1e-6 * n1_exact


def test_a_solvent_viscosity_adds_its_newtonian_stress():
    """Oldroyd-B in shear: the total shear stress is the solvent's eta_s gdot at
    once plus the polymer's eta_p gdot (1 - e^{-t/lambda}) building up."""
    steps, dt, eta_s = 20, 0.1, 0.5
    _, polymer_xy, polymer_exact, _, solvent_xy = _maxwell_shear("semi_lagrangian", 1, steps=steps, dt=dt, solvent=eta_s)
    gdot = 1.0
    assert abs(polymer_xy - polymer_exact) / polymer_exact < 0.02, (polymer_xy, polymer_exact)
    assert abs(solvent_xy - eta_s * gdot) < 1e-6, solvent_xy
