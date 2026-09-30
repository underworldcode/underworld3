"""The DDt history manager as the transport plugin of the Eulerian solvers.

A solver composes its residual from three contributions of its history
manager (``time_derivative``, ``advection``, ``stabilisation_flux``) and
never asks which flavour it holds: the ``EulerianSUPG`` manager assembles
implicit advection with streamline-upwind stabilisation, the history-carrying
flavours answer zero for both. These checks cover the contract on each
flavour, the shapes for scalar, vector and tensor unknowns, the
semi-Lagrangian manager dropped into the SUPG solver, and a tensor unknown
transported through the multi-component solver.

Run: pixi run python -m pytest tests/test_1057_ddt_transport_plugin.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities._api_tools import Template

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 16, qdegree=3)


def _gaussian(x, y, x0=0.5, y0=0.0, width=0.03):
    return sympy.exp(-((x - x0) ** 2 + (y - y0) ** 2) / width)


def _is_zero(M):
    return all(e == 0 for e in sympy.Matrix(M))


def test_history_flavours_answer_zero_for_advection_and_stabilisation(mesh):
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T_c", mesh, 1, degree=2)
    V = sympy.Matrix([[-y, x]])
    sl = uw.systems.ddt.SemiLagrangian(
        mesh, T.sym, V, vtype=uw.VarType.SCALAR, degree=2, continuous=True, order=1)
    assert sl.integrator == "am"
    assert _is_zero(sl.advection()) and sl.advection().shape == (1, 1)
    flux = sl.stabilisation_flux(sympy.Matrix([[7]]))
    assert flux.shape == (1, 2) and _is_zero(flux)
    td = sl.time_derivative()
    assert td.shape == (1, 1)
    assert sl.states()[1] == sl.psi_star[0].sym and len(sl.spatial_weights()) == 2
    # the timestep is a runtime constant the manager writes on every pre-solve
    T.array[:, 0, 0] = uw.function.evaluate(_gaussian(x, y), T.coords).reshape(-1)
    sl.update_pre_solve(0.02)
    assert float(sl.delta_t.sym) == 0.02
    assert T.sym[0] in td.atoms(sympy.Function) and sl.psi_star[0].sym[0] in td.atoms(sympy.Function)

    eulerian = uw.systems.ddt.Eulerian(
        mesh, T, vtype=uw.VarType.SCALAR, degree=2, continuous=True, V_fn=V)
    assert eulerian._advection_mode == "split"       # the velocity corrects the history
    assert _is_zero(eulerian.advection()) and _is_zero(eulerian.stabilisation_flux(sympy.ones(1, 1)))


def test_supg_manager_contributions_have_the_unknowns_shape(mesh):
    x, y = mesh.X
    V = sympy.Matrix([[-y, x]])
    SUPG = uw.systems.ddt.EulerianSUPG

    T = uw.discretisation.MeshVariable("T_s", mesh, 1, degree=2)
    scalar = SUPG(mesh, T, V, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    assert scalar._advection_mode == "assembled" and scalar.V_fn == V
    assert scalar.time_derivative().shape == (1, 1) and scalar.advection().shape == (1, 1)
    R = scalar.time_derivative() + scalar.advection()
    assert scalar.stabilisation_flux(R).shape == (1, 2)
    # the advection is the velocity dotted with the gradient of each level, weighted
    w0, w1 = scalar.spatial_weights()
    expected = sum(w * V.dot(mesh.vector.gradient(level[0]))
                   for w, level in zip((w0, w1), scalar.states()))
    assert sympy.simplify(scalar.advection()[0] - expected) == 0

    U = uw.discretisation.MeshVariable("U_s", mesh, 2, degree=2)
    vector = SUPG(mesh, U, V, vtype=uw.VarType.VECTOR, degree=2, continuous=True, order=2)
    assert vector.integrator == "bdf" and vector.advection().shape == (1, 2)
    R = sympy.Matrix([[sympy.Symbol("R_0"), sympy.Symbol("R_1")]])
    F = vector.stabilisation_flux(R)
    assert F.shape == (2, 2)
    assert sympy.simplify(F - vector.tau() * (R.T * V)) == sympy.zeros(2, 2)   # F_ij = tau R_i a_j
    # a self-advected unknown names the stored velocity at the stored levels
    vector.V_fn_history = [ps.sym for ps in vector.psi_star]
    assert vector.advecting_velocity(1) == vector.psi_star[0].sym

    S = uw.discretisation.MeshVariable("S_s", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)
    tensor = SUPG(mesh, S, V, vtype=uw.VarType.SYM_TENSOR, degree=1, continuous=True)
    assert tensor.advection().shape == (2, 2)
    assert tensor.advection()[0, 1] == tensor.advection()[1, 0]
    assert tensor.stabilisation_flux(tensor.advection()).shape == (4, 2)

    with pytest.raises(ValueError, match="tau_shape"):
        SUPG(mesh, T, V, vtype=uw.VarType.SCALAR, degree=2, continuous=True, tau_shape="optimal")
    with pytest.raises(ValueError, match="theta applies"):
        SUPG(mesh, T, V, vtype=uw.VarType.SCALAR, degree=2, continuous=True, order=2, theta=0.5)


def test_semi_lagrangian_manager_drops_into_the_supg_solver():
    """With a semi-Lagrangian history the SUPG solver assembles no advection
    and no stabilisation: on pure advection its equation is the one the
    semi-Lagrangian solver solves, and the two fields agree to the solver
    tolerances after a quarter revolution of a Gaussian."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 16, qdegree=3)
    x, y = mesh.X
    V = sympy.Matrix([[-y, x]])
    dt, steps = 0.1, 16

    def field(tag):
        T = uw.discretisation.MeshVariable(f"T_{tag}", mesh, 1, degree=2)
        T.array[:, 0, 0] = uw.function.evaluate(_gaussian(x, y), T.coords).reshape(-1)
        return T

    T_plug = field("plug")
    history = uw.systems.ddt.SemiLagrangian(
        mesh, T_plug.sym, V, vtype=uw.VarType.SCALAR, degree=2, continuous=True, order=1)
    plug = uw.systems.AdvDiffusion(mesh, T_plug, V, DuDt=history)
    assert plug.DuDt is history and plug.integrator == "am" and plug.order == 1
    assert _is_zero(plug.DuDt.advection()) and _is_zero(plug._stabilisation_flux())
    with pytest.raises(AttributeError):
        plug.supg_weight                      # no stabilisation knobs on this manager

    T_slcn = field("slcn")
    slcn = uw.systems.AdvDiffusionSLCN(mesh, T_slcn, V)
    slcn.constitutive_model = uw.constitutive_models.DiffusionModel
    slcn.constitutive_model.Parameters.diffusivity = 0.0

    T_supg = field("supg")
    supg = uw.systems.AdvDiffusion(mesh, T_supg, V)

    for solver in (plug, slcn, supg):
        for b in ("Left", "Right", "Top", "Bottom"):
            solver.add_dirichlet_bc(0.0, b)
    for _ in range(steps):
        plug.solve(timestep=dt)
        slcn.solve(timestep=dt)
        supg.solve(timestep=dt)

    a, b, c = (np.asarray(T.array[:, 0, 0]) for T in (T_plug, T_slcn, T_supg))
    assert np.abs(a - b).max() < 1e-5 * np.abs(b).max()      # measured 5e-8 per step
    # negative control: the assembled scheme is a different discretisation
    assert np.abs(a - c).max() > 1e-3
    # and every scheme moved the Gaussian
    T0 = uw.function.evaluate(_gaussian(x, y), T_plug.coords).reshape(-1)
    assert np.abs(a - T0).max() > 0.3


def test_tensor_unknown_is_transported_through_the_multicomponent_solver():
    """A flattened symmetric tensor (xx, xy, yy) carried by a uniform velocity
    with the SUPG manager as the transport of a multi-component solver whose
    residual is just the manager's terms."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1 / 16, qdegree=3)
    x, y = mesh.X
    V = sympy.Matrix([[0.5, 0.0]])
    S = uw.discretisation.MeshVariable("S_t", mesh, (1, 3), vtype=uw.VarType.MATRIX, degree=2)
    amplitudes = (1.0, 0.5, -1.0)
    g0 = uw.function.evaluate(_gaussian(x, y, x0=-0.25), S.coords).reshape(-1)
    for k, amp in enumerate(amplitudes):
        S.array[:, 0, k] = amp * g0

    transport = uw.systems.ddt.EulerianSUPG(
        mesh, S, V, vtype=uw.VarType.MATRIX, degree=2, continuous=True,
        num_components=(1, 3))

    class TensorTransport(uw.systems.SNES_MultiComponent):
        F0 = Template(r"f_0", lambda self: self.DuDt.time_derivative() + self.DuDt.advection(),
                      "time derivative and advection of every component")
        F1 = Template(r"F_1", lambda self: self.DuDt.stabilisation_flux(
            self.DuDt.time_derivative() + self.DuDt.advection()), "the SUPG flux per component")

    solver = TensorTransport(mesh, u_Field=S, DuDt=transport)
    solver.constitutive_model = uw.constitutive_models.Constitutive_Model
    solver.petsc_options["snes_rtol"] = 1e-8
    solver.petsc_options["ksp_rtol"] = 1e-9
    dt, steps = 0.05, 10
    for _ in range(steps):
        transport.update_pre_solve(dt)
        solver.solve()
        transport.update_post_solve(dt)
    assert float(transport.delta_t.sym) == dt

    exact = uw.function.evaluate(_gaussian(x, y, x0=-0.25 + 0.5 * dt * steps), S.coords).reshape(-1)
    data = np.asarray(S.array)
    for k, amp in enumerate(amplitudes):
        err = np.linalg.norm(data[:, 0, k] - amp * exact) / np.linalg.norm(amp * exact)
        assert err < 0.05, (k, err)
        # negative control: the field moved away from where it started
        assert np.linalg.norm(data[:, 0, k] - amp * g0) / np.linalg.norm(amp * g0) > 0.3


def test_dimensional_timestep_reaches_the_manager_non_dimensional():
    """A quantity timestep handed to a manager's pre-solve is scaled by the model's
    reference time before it becomes the kernels' runtime constant (#701): the
    semi-Lagrangian solver passes its Pint step straight through."""
    from underworld3.systems.ddt import _as_float
    q = uw.quantity(100.0, "kyr")
    assert _as_float(q) == 100.0                       # no reference scales: the magnitude
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        length=uw.quantity(1000.0, "km"), time=uw.quantity(1.0, "Myr"))
    try:
        assert abs(_as_float(q) - 0.1) < 1e-12
        assert abs(_as_float(q._pint_qty) - 0.1) < 1e-12   # a raw Pint quantity too
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)
        x, y = mesh.X
        T = uw.discretisation.MeshVariable("T_dim", mesh, 1, degree=2)
        T.array[:, 0, 0] = uw.function.evaluate(_gaussian(x, y), T.coords).reshape(-1)
        slcn = uw.systems.AdvDiffusionSLCN(mesh, T, sympy.Matrix([[-y, x]]))
        slcn.constitutive_model = uw.constitutive_models.DiffusionModel
        slcn.constitutive_model.Parameters.diffusivity = 0.0
        slcn.solve(timestep=q)
        assert abs(float(slcn.DuDt.delta_t.sym) - 0.1) < 1e-12, slcn.DuDt.delta_t.sym
    finally:
        uw.reset_default_model()
