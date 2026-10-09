"""Thin sheet on the surface of a sphere: the velocity lives in the embedding
space with a strong tangency constraint at every node.

Closed forms: a rigid rotation is a Killing field of the sphere (zero surface
strain rate), so a force c (omega x X) against basal drag c returns exactly
u = omega x X; a toroidal manufactured solution of the sheet operator
converges at second order (flat-facet geometry) with machine-exact tangency.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]


def _sphere(cell_size, degree=2):
    return uw.meshing.SphericalManifold(radius=1.0, cellSize=cell_size, degree=1,
                                        qdegree=2 * degree + 1)


def _sheet(sphere, eta=1):
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = eta
    sheet.tolerance = 1.0e-9
    return sheet


def _leak(sheet):
    c = sheet.u.coords
    u = sheet.u.array[:, 0, :]
    rhat = c / np.linalg.norm(c, axis=1)[:, None]
    return np.abs((u * rhat).sum(axis=1)).max()


@pytest.mark.tier_b
def test_rigid_rotation_is_a_killing_field():
    """u = omega x X has no surface strain rate: with basal drag c and the
    force c (omega x X) the sheet returns the rotation exactly (P2 carries the
    linear field) and the tangency constraint holds to machine precision."""
    sphere = _sphere(0.25)
    X = sphere.X
    x, y, z = X[0], X[1], X[2]
    omega = sympy.Matrix([[sympy.Rational(1, 3), sympy.Rational(-1, 2), sympy.Rational(2, 7)]])
    rot = sympy.Matrix([[omega[1] * z - omega[2] * y,
                         omega[2] * x - omega[0] * z,
                         omega[0] * y - omega[1] * x]])
    sheet = _sheet(sphere)
    sheet.basal_drag = 2
    sheet.bodyforce = list(2 * rot)
    sheet.solve()
    assert sheet._rotated_freeslip_info["converged"]
    c = sheet.u.coords
    exact = np.cross(np.array([1 / 3, -1 / 2, 2 / 7]), c)
    assert np.abs(sheet.u.array[:, 0, :] - exact).max() < 1.0e-7
    assert _leak(sheet) < 1.0e-13


def _toroidal_mms(sphere, eta, alpha):
    """Manufactured toroidal l=2 field u_e = rhat x grad(z^2) = (2yz, -2xz, 0)
    and the force that makes it the sheet solution with basal drag alpha."""
    X = sphere.X
    x, y, z = X[0], X[1], X[2]
    r = sympy.sqrt(x**2 + y**2 + z**2)
    rhat = sympy.Matrix([[x, y, z]]) / r
    P = sympy.eye(3) - rhat.T * rhat
    u_e = sympy.Matrix([[2 * y * z, -2 * x * z, 0]])
    G = u_e.jacobian(sympy.Matrix([x, y, z])) * P
    E = P * ((G + G.T) / 2) * P
    sigma = 2 * eta * (E + E.trace() * P)
    Xs = [x, y, z]
    divs = sympy.Matrix([[sum(sigma[i, j].diff(Xs[k]) * P[k, j]
                              for j in range(3) for k in range(3)) for i in range(3)]])
    return u_e, (-divs + alpha * u_e).subs(r, 1)


@pytest.mark.tier_b
def test_manufactured_toroidal_solution_converges_at_second_order():
    """Relative nodal L2 error of the tangential velocity at h = 0.2 and 0.1:
    second order (the flat-facet geometry limit for P2 on a curved surface),
    tangency exact. Measured 2026-10-08: 6.9e-3 -> 1.8e-3."""
    eta, alpha = sympy.Integer(1), sympy.Integer(1)
    errors = {}
    for h in (0.2, 0.1):
        sphere = _sphere(h)
        u_e, force = _toroidal_mms(sphere, eta, alpha)
        sheet = _sheet(sphere, eta=eta)
        sheet.basal_drag = alpha
        sheet.bodyforce = list(force)
        sheet.solve()
        assert sheet._rotated_freeslip_info["converged"]
        assert _leak(sheet) < 1.0e-13
        c = sheet.u.coords
        exact = np.column_stack([2 * c[:, 1] * c[:, 2], -2 * c[:, 0] * c[:, 2], np.zeros(len(c))])
        err = sheet.u.array[:, 0, :] - exact
        errors[h] = np.sqrt((err**2).sum()) / np.sqrt((exact**2).sum())
    order = np.log(errors[0.2] / errors[0.1]) / np.log(2.0)
    assert errors[0.1] < 3.0e-3, errors
    assert order > 1.7, f"order {order:.2f}: {errors}"


@pytest.mark.tier_b
def test_a_maxwell_sheet_keeps_a_rigid_rotation_stress_free():
    """With the grid-carried stress history, the Killing field stays the exact
    solution step after step and the carried stress stays at round-off."""
    sphere = _sphere(0.25)
    X = sphere.X
    x, y, z = X[0], X[1], X[2]
    omega = sympy.Matrix([[sympy.Rational(1, 3), sympy.Rational(-1, 2), sympy.Rational(2, 7)]])
    rot = sympy.Matrix([[omega[1] * z - omega[2] * y,
                         omega[2] * x - omega[0] * z,
                         omega[0] * y - omega[1] * x]])
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd")
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 10
    sheet.constitutive_model.Parameters.shear_modulus = 1
    sheet.basal_drag = 2
    sheet.bodyforce = list(2 * rot)
    sheet.tolerance = 1.0e-9
    c = sheet.u.coords
    exact = np.cross(np.array([1 / 3, -1 / 2, 2 / 7]), c)
    for _ in range(4):
        sheet.solve(timestep=0.1, zero_init_guess=False)
        assert np.abs(sheet.u.array[:, 0, :] - exact).max() < 1.0e-7
        assert np.abs(np.asarray(sheet.tau.array)).max() < 1.0e-7
        assert _leak(sheet) < 1.0e-13


@pytest.mark.tier_b
def test_a_manifold_sheet_refuses_a_semi_lagrangian_history():
    """The history is carried on the grid on a manifold; asking for the
    trace-back is refused before any history exists."""
    sheet = uw.systems.ThinSheet(_sphere(0.4))
    assert sheet.stress_transport == "eulerian"
    with pytest.raises(ValueError, match="manifold"):
        sheet.stress_transport = "semi_lagrangian"
    sheet.stress_transport = "eulerian"


@pytest.mark.tier_b
def test_reloadable_checkpoint_round_trips_on_a_manifold(tmp_path):
    """write_timestep(petsc_reload=True) writes the coordinates with one
    component per embedding dimension; it used to reshape them by the
    topological dimension and fail on a surface mesh. The written vector
    reloads exactly."""
    sphere = _sphere(0.3)
    v = uw.discretisation.MeshVariable("v", sphere, 3, degree=2)
    v.array[:, 0, :] = v.coords
    sphere.write_timestep("manifold", 0, outputPath=str(tmp_path), meshVars=[v],
                          petsc_reload=True)
    w = uw.discretisation.MeshVariable("w", sphere, 3, degree=2)
    # the in-place reload (same mesh object); the DMPlex-migration path
    # segfaults on a manifold mesh (TODO(BUG) in read_checkpoint)
    w.read_checkpoint(str(tmp_path / "manifold.mesh.v.00000.h5"), data_name="v",
                      same_layout=True)
    assert np.abs(w.array[:, 0, :] - w.coords).max() < 1.0e-12


@pytest.mark.tier_b
def test_the_grid_history_carries_a_tensor_round_the_sphere():
    """The grid (Eulerian SUPG) history advects each component as a scalar.
    A rigid rotation about the x axis, u = (0, -z, y), carries a component
    planted as z to -y after a quarter turn; the rotation has a z velocity, so a
    transport that sums only the first mesh.dim velocity components (as it
    once did on a surface mesh) leaves the field where it started (measured
    1.126 then). Measured 2026-10-09, h = 0.15, 40 steps: 2.27e-3."""
    sphere = _sphere(0.15)
    X = sphere.X
    x, y, z = X[0], X[1], X[2]
    stress = uw.discretisation.MeshVariable(
        "S_turn", sphere, vtype=uw.VarType.SYM_TENSOR, degree=2)
    manager = uw.systems.ddt.EulerianSUPG(
        sphere, stress, sympy.Matrix([[0, -z, y]]), vtype=uw.VarType.SYM_TENSOR,
        degree=2, continuous=True, order=1, transport_on_update=True)
    c = stress.coords
    stress.array[:, 0, 0] = c[:, 2]
    manager.initialise_history()
    dt, steps = np.pi / 2 / 40, 40                 # a quarter turn
    for _ in range(steps):
        manager.update_pre_solve(dt)
        stress.array[...] = manager.psi_star[0].array[...]
    carried = np.asarray(manager.psi_star[0].array[:, 0, 0])
    turned = -c[:, 1]
    error = np.sqrt(((carried - turned) ** 2).sum() / (turned ** 2).sum())
    assert error < 5.0e-3, f"relative error {error:.2e} after a quarter turn"


@pytest.mark.tier_b
def test_supg_advection_turns_a_scalar_round_the_sphere():
    """The Eulerian SUPG advection-diffusion solver on a surface mesh: a rigid
    rotation about the x axis, u = (0, -z, y), carries T = z to T = -y after a
    quarter turn (no diffusion). The velocity has three components and the
    surface gradient three entries; the solver must use all of them.
    Measured 2026-10-09, h = 0.15, 40 steps: 2.27e-3."""
    sphere = _sphere(0.15)
    X = sphere.X
    x, y, z = X[0], X[1], X[2]
    T = uw.discretisation.MeshVariable("T_turn", sphere, 1, degree=2)
    T.array[:, 0, 0] = T.coords[:, 2]
    adv = uw.systems.AdvDiffusion(sphere, T, sympy.Matrix([[0, -z, y]]))
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 0
    dt, steps = np.pi / 2 / 40, 40                 # a quarter turn
    for _ in range(steps):
        adv.solve(timestep=dt)
    turned = -T.coords[:, 1]
    error = np.sqrt(((T.array[:, 0, 0] - turned) ** 2).sum() / (turned ** 2).sum())
    assert error < 5.0e-3, f"relative error {error:.2e} after a quarter turn"
