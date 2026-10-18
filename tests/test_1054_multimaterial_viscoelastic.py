"""Viscoelastic constituents inside the multi-material constitutive model.

A box in pure shear, :math:`v = (\\dot\\varepsilon x, -\\dot\\varepsilon y)`, prescribed
on every wall, so the strain rate is uniform and known and each material relaxes
on its own Maxwell time :math:`\\lambda = \\eta/\\mu` towards
:math:`2\\eta\\dot\\varepsilon`. Material A fills :math:`x < 1/2` and material B
the rest; the deviatoric :math:`\\sigma_{xx}` of the committed stress history is
sampled at :math:`x = 0.15` and :math:`x = 0.85`, away from the interface. Closed
forms per step :math:`n`:

- ETD-1, exact for constant forcing: :math:`2\\eta\\dot\\varepsilon(1 - e^{-n\\Delta t/\\lambda})`.
- BDF-1: :math:`\\sigma_{n} = \\eta_e(2\\dot\\varepsilon + \\sigma_{n-1}/(\\mu\\Delta t))`
  with :math:`\\eta_e = \\eta\\mu\\Delta t/(\\eta + \\mu\\Delta t)`.

The history is a continuous projection of the composite stress, which smears
the stress jump at the interface into the sample points. Measured 2026-10-18
(8x8 quads, P2/P1, six steps of 0.25), error over the run relative to the peak
stress, BDF / ETD: identical constituents 5.6e-12 / 5.7e-12 (the composite is
the single material); 4:1 Maxwell times 5.6e-4 / 5.2e-4; BDF-1 beside ETD-1
6.0e-4, BDF-2 beside BDF-1 2.3e-4; yield in one
constituent 5.7e-4 / 6.0e-4; isotropic beside transversely isotropic 1.42e-3 /
1.40e-3 (the larger jump, 1.55 against 0.47, read relative to the smaller
stress); viscous eta = 4 beside VEP, 7.9e-3 / 7.3e-3 in the VEP material.
The smearing grows with the stress jump measured against the local stress.
Tolerances are about three times these.

Also here, single-material regressions found on the way: a transversely
isotropic model given its parameters before it meets a solver (its tensor was
built, and cached, with an infinite timestep) and the viscoelastic effective
viscosity at an infinite timestep (0 * inf).
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function.expressions import unwrap_expression

pytestmark = pytest.mark.level_2

EDOT = 1.0
MU = 1.0
DT = 0.25
NSTEPS = 6
POINTS = np.array([[0.15, 0.5], [0.85, 0.5]])
# Interface smearing of the projected history at the sample points (see the
# module docstring): about three times the largest measured error.
SMEARED = 2.0e-3
SMEARED_TI = 4.0e-3
SMEARED_VISCOUS = 2.5e-2
EXACT = 1.0e-8


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _pure_shear_box(tag):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8), qdegree=3)
    v = uw.discretisation.MeshVariable(f"v{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=1, continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, p_continuous=False)
    x, y = mesh.X
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((EDOT * x, -EDOT * y), wall)
    stokes.tolerance = 1.0e-10
    return mesh, stokes


def _left_right_materials(mesh, tag):
    swarm = uw.swarm.Swarm(mesh)
    material = uw.swarm.IndexSwarmVariable(f"M{tag}", swarm, indices=2, proxy_degree=1)
    swarm.populate(fill_param=3)
    material.array[:, 0, 0] = (np.asarray(swarm.coords)[:, 0] > 0.5).astype(int)
    return swarm, material


def _vep(stokes, eta, integrator, name=None, yield_stress=None, order=1):
    constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=order, integrator=integrator, material_name=name)
    constitutive_model.Parameters.shear_viscosity_0 = eta
    constitutive_model.Parameters.shear_modulus = MU
    if yield_stress is not None:
        constitutive_model.Parameters.yield_stress = yield_stress
        constitutive_model.yield_mode = "min"
    return constitutive_model


def _ti_vep(stokes, integrator, name=None):
    """Fault-plane viscosity 0.25 on a plane at 45 degrees: pure shear in x, y is
    pure shear on that plane, so sigma_xx relaxes on eta_1 / mu."""
    constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
        stokes.Unknowns, order=1, integrator=integrator, material_name=name)
    _ti_parameters(constitutive_model)
    return constitutive_model


def _ti_parameters(constitutive_model):
    constitutive_model.Parameters.shear_viscosity_0 = 1.0
    constitutive_model.Parameters.shear_viscosity_1 = 0.25
    constitutive_model.Parameters.shear_modulus = MU
    constitutive_model.Parameters.director = sympy.Matrix([[1, 1]]) / sympy.sqrt(2)


def _history_sigma_xx(stokes, nsteps=NSTEPS):
    """sigma_xx of the committed history at the two sample points, per step."""
    out = []
    for _ in range(nsteps):
        stokes.solve(timestep=DT)
        sigma = stokes.Unknowns.DFDt.psi_star[0].sym
        out.append(np.asarray(uw.function.global_evaluate(sigma[0, 0], POINTS)).reshape(-1))
    return np.array(out)


def _closed_form(eta, integrator, nsteps=NSTEPS, yield_stress=np.inf, order=1):
    """sigma_xx after each step; a yield stress caps the pure-shear invariant |sigma_xx|.
    BDF-2 (weights 3/2, -2, 1/2) starts once the history holds two committed
    levels, so its first two steps are BDF-1."""
    sigma, previous, out = 0.0, 0.0, []
    for n in range(nsteps):
        if integrator == "etd":
            alpha = np.exp(-DT * MU / eta)
            new = alpha * sigma + (1.0 - alpha) * 2.0 * eta * EDOT
        elif order == 2 and n >= 2:
            eta_e = eta * MU * DT / (1.5 * eta + MU * DT)
            new = eta_e * (2.0 * EDOT + (2.0 * sigma - 0.5 * previous) / (MU * DT))
        else:
            eta_e = eta * MU * DT / (eta + MU * DT)
            new = eta_e * (2.0 * EDOT + sigma / (MU * DT))
        previous, sigma = sigma, min(new, yield_stress)
        out.append(sigma)
    return np.array(out)


def _relative_error(sim, ref):
    return np.abs(sim - ref).max() / np.abs(ref).max()


@pytest.mark.tier_a
@pytest.mark.parametrize("integrator", ["bdf", "etd"])
def test_identical_constituents_are_the_single_material(integrator):
    mesh, stokes = _pure_shear_box(f"i{integrator}")
    swarm, material = _left_right_materials(mesh, f"i{integrator}")
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material,
        [_vep(stokes, 1.0, integrator, "A"), _vep(stokes, 1.0, integrator, "B")])
    sim = _history_sigma_xx(stokes)
    ref = _closed_form(1.0, integrator)
    for side in (0, 1):
        assert _relative_error(sim[:, side], ref) < EXACT, (side, sim[:, side], ref)


@pytest.mark.tier_a
@pytest.mark.parametrize("integrator", ["bdf", "etd"])
def test_each_material_relaxes_on_its_own_maxwell_time(integrator):
    mesh, stokes = _pure_shear_box(f"m{integrator}")
    swarm, material = _left_right_materials(mesh, f"m{integrator}")
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material,
        [_vep(stokes, 1.0, integrator, "A"), _vep(stokes, 4.0, integrator, "B")])
    sim = _history_sigma_xx(stokes)
    for side, eta in ((0, 1.0), (1, 4.0)):
        ref = _closed_form(eta, integrator)
        assert _relative_error(sim[:, side], ref) < SMEARED, (side, sim[:, side], ref)


@pytest.mark.tier_b
@pytest.mark.parametrize("a, b", [(("bdf", 1), ("etd", 1)), (("bdf", 2), ("bdf", 1))],
                         ids=["bdf1-etd1", "bdf2-bdf1"])
def test_mixed_integrators_and_orders(a, b):
    """Constituents read the same stored stress through their own schemes."""
    tag = f"x{a[0]}{a[1]}{b[0]}{b[1]}"
    mesh, stokes = _pure_shear_box(tag)
    swarm, material = _left_right_materials(mesh, tag)
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material,
        [_vep(stokes, 1.0, a[0], "A", order=a[1]), _vep(stokes, 4.0, b[0], "B", order=b[1])])
    sim = _history_sigma_xx(stokes)
    for side, eta, (integrator, order) in ((0, 1.0, a), (1, 4.0, b)):
        ref = _closed_form(eta, integrator, order=order)
        assert _relative_error(sim[:, side], ref) < SMEARED, (side, sim[:, side], ref)


@pytest.mark.tier_b
@pytest.mark.parametrize("integrator", ["bdf", "etd"])
def test_isotropic_and_transversely_isotropic_constituents(integrator):
    mesh, stokes = _pure_shear_box(f"t{integrator}")
    swarm, material = _left_right_materials(mesh, f"t{integrator}")
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material,
        [_vep(stokes, 1.0, integrator, "A"), _ti_vep(stokes, integrator, "B")])
    sim = _history_sigma_xx(stokes)
    for side, eta in ((0, 1.0), (1, 0.25)):
        ref = _closed_form(eta, integrator)
        assert _relative_error(sim[:, side], ref) < SMEARED_TI, (side, sim[:, side], ref)


@pytest.mark.tier_b
@pytest.mark.parametrize("integrator", ["bdf", "etd"])
def test_yield_in_one_constituent(integrator):
    tau_y = 0.6
    mesh, stokes = _pure_shear_box(f"y{integrator}")
    swarm, material = _left_right_materials(mesh, f"y{integrator}")
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material,
        [_vep(stokes, 1.0, integrator, "A", yield_stress=tau_y),
         _vep(stokes, 1.0, integrator, "B")])
    sim = _history_sigma_xx(stokes)
    yielding = _closed_form(1.0, integrator, yield_stress=tau_y)
    elastic = _closed_form(1.0, integrator)
    assert _relative_error(sim[:, 0], yielding) < SMEARED, (sim[:, 0], yielding)
    assert _relative_error(sim[:, 1], elastic) < SMEARED, (sim[:, 1], elastic)


@pytest.mark.tier_b
@pytest.mark.parametrize("integrator", ["bdf", "etd"])
def test_a_constituent_change_reaches_the_solver(integrator):
    """A yield stress given to B between steps caps B's stress at the next step.

    The change is structural (a new term in B's flux, not a new value of a
    constant), so it reaches the solve only if the composite tells the solver
    to rebuild what it compiled. Measured 2026-10-18: B after the change
    [0.4996, 0.5067] (BDF), within 1.4e-2 / 1.5e-2 (BDF / ETD) of tau_y = 0.5
    (the stress of the unyielded A beside it smears in), A within 7.5e-4 /
    7.8e-4 of its closed form; without the rebuild B carries on to
    [0.976, 1.18]."""
    tau_y = 0.5
    mesh, stokes = _pure_shear_box(f"c{integrator}")
    swarm, material = _left_right_materials(mesh, f"c{integrator}")
    a, b = _vep(stokes, 1.0, integrator, "A"), _vep(stokes, 1.0, integrator, "B")
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(stokes.Unknowns, material, [a, b])
    _history_sigma_xx(stokes, nsteps=2)
    b.Parameters.yield_stress = tau_y
    b.yield_mode = "min"
    after = _history_sigma_xx(stokes, nsteps=2)
    assert np.abs(after[:, 1] - tau_y).max() < 0.05 * tau_y, after[:, 1]
    assert _relative_error(after[:, 0], _closed_form(1.0, integrator, nsteps=4)[2:]) < SMEARED, after[:, 0]


@pytest.mark.tier_a
def test_reassigning_the_composite_keeps_each_constituent_order():
    """The solver passes its history's order to a model it is given again; a
    BDF-1 constituent beside a BDF-2 one stays BDF-1."""
    mesh, stokes = _pure_shear_box("o")
    swarm, material = _left_right_materials(mesh, "o")
    a = _vep(stokes, 1.0, "bdf", "A", order=2)
    b = _vep(stokes, 4.0, "bdf", "B", order=1)
    composite = uw.MultiMaterialConstitutiveModel(stokes.Unknowns, material, [a, b])
    stokes.constitutive_model = composite
    stokes.constitutive_model = composite
    assert (a.order, b.order) == (2, 1)
    sim = _history_sigma_xx(stokes)
    for side, eta, order in ((0, 1.0, 2), (1, 4.0, 1)):
        ref = _closed_form(eta, "bdf", order=order)
        assert _relative_error(sim[:, side], ref) < SMEARED, (side, sim[:, side], ref)


@pytest.mark.tier_b
@pytest.mark.parametrize("integrator", ["bdf", "etd"])
def test_viscous_beside_viscoelastic(integrator):
    """A purely viscous constituent (eta = 4: stress 8 from the first step)
    beside a VEP one. The stress jump is large against A's stress, so the
    smearing is larger than for two VEP constituents. Measured 2026-10-18, A /
    B, BDF 7.9e-3 / 8.1e-4, ETD 7.3e-3 / 8.0e-4."""
    mesh, stokes = _pure_shear_box(f"v{integrator}")
    swarm, material = _left_right_materials(mesh, f"v{integrator}")
    viscous = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns, material_name="B")
    viscous.Parameters.shear_viscosity_0 = 4.0
    stokes.constitutive_model = uw.MultiMaterialConstitutiveModel(
        stokes.Unknowns, material, [_vep(stokes, 1.0, integrator, "A"), viscous])
    sim = _history_sigma_xx(stokes)
    assert _relative_error(sim[:, 0], _closed_form(1.0, integrator)) < SMEARED_VISCOUS, sim[:, 0]
    assert _relative_error(sim[:, 1], np.full(NSTEPS, 2.0 * 4.0 * EDOT)) < SMEARED, sim[:, 1]


# --- single-material regressions -------------------------------------------------


@pytest.mark.tier_a
@pytest.mark.parametrize("integrator", ["bdf", "etd"])
@pytest.mark.parametrize("parameters_first", [True, False])
def test_transversely_isotropic_parameters_before_the_solver(integrator, parameters_first):
    """Setting the parameters before the model meets the solver (its tensor built
    at an infinite timestep) gives the same relaxation as setting them after.
    The regression is [True-bdf]; the other three are controls (setting after,
    and the exponential tensor, which never read the step, passed before)."""
    mesh, stokes = _pure_shear_box(f"p{integrator}{int(parameters_first)}")
    constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
        stokes.Unknowns, order=1, integrator=integrator)
    if parameters_first:
        _ti_parameters(constitutive_model)
        stokes.constitutive_model = constitutive_model
    else:
        stokes.constitutive_model = constitutive_model
        _ti_parameters(constitutive_model)
    sim = _history_sigma_xx(stokes, nsteps=3)
    ref = _closed_form(0.25, integrator, nsteps=3)
    for side in (0, 1):
        assert _relative_error(sim[:, side], ref) < EXACT, (side, sim[:, side], ref)


@pytest.mark.tier_a
@pytest.mark.parametrize("model_class", ["isotropic", "transverse"])
def test_infinite_timestep_is_the_viscous_limit(model_class):
    """With no timestep (dt = inf) the viscoelastic effective viscosity is the
    viscosity itself, whatever the modulus: eta mu dt / (eta + mu dt) -> eta."""
    mesh, stokes = _pure_shear_box(f"d{model_class}")
    if model_class == "isotropic":
        constitutive_model = _vep(stokes, 2.0, "bdf")
    else:
        constitutive_model = _ti_vep(stokes, "bdf")
    assert constitutive_model.Parameters.dt_elastic.sym is sympy.oo
    eta = constitutive_model.Parameters.ve_effective_viscosity
    value = sympy.sympify(unwrap_expression(eta, mode="nondimensional"))
    expected = 2 if model_class == "isotropic" else sympy.Rational(1, 4)
    assert value.is_finite and sympy.nsimplify(value) == expected, value
