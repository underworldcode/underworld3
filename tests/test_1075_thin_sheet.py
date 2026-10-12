"""Thin viscous sheet (plane stress) on a flat mesh: closed-form checks.

The sheet closure is :math:`\\sigma = \\tau + \\mathrm{tr}(\\tau)\\,\\mathbf{I}`
with :math:`\\tau = 2\\eta\\dot\\varepsilon`. Every test here has an exact
answer the discrete solution must reproduce to solver tolerance, except the
manufactured-solution test, whose contract is the convergence order.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


def _box(cell_size=0.1):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size, qdegree=3)


def _sheet(mesh, eta=1):
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = eta
    sheet.tolerance = 1.0e-10
    return sheet


def _probe_points(n=7):
    s = np.linspace(0.15, 0.85, n)
    return np.array([[a, b] for a in s for b in s])


@pytest.mark.tier_b
def test_trouton_uniaxial_stretch():
    """Uniaxial stretch with free lateral edges: the lateral contraction is
    half the stretch and the axial stress is the Trouton value 3 eta edot."""
    mesh = _box()
    x, y = mesh.X
    edot = sympy.Rational(1, 2)
    eta = 2
    sheet = _sheet(mesh, eta=eta)
    # exact solution u = (edot x, -edot (y - 1/2) / 2): prescribe it on the
    # loaded ends, leave the lateral edges traction-free
    u_y = -edot * (y - sympy.Rational(1, 2)) / 2
    sheet.add_essential_bc([0, u_y], "Left")
    sheet.add_essential_bc([edot, u_y], "Right")
    sheet.solve()

    pts = _probe_points()
    u = uw.function.evaluate(sheet.u.sym, pts)
    exact = np.column_stack([edot * pts[:, 0], -edot * (pts[:, 1] - 0.5) / 2])
    assert np.abs(u.reshape(-1, 2) - exact).max() < 1.0e-8

    # the stress is a derivative of the P2 field sampled at arbitrary points:
    # point-evaluation noise sits ~1/h above the velocity's 1e-8
    sigma = uw.function.evaluate(sheet.stress, pts).reshape(-1, 2, 2)
    trouton = 3 * eta * float(edot)
    assert np.abs(sigma[:, 0, 0] - trouton).max() < 1.0e-6 * trouton
    assert np.abs(sigma[:, 1, 1]).max() < 1.0e-6 * trouton
    assert np.abs(sigma[:, 0, 1]).max() < 1.0e-6 * trouton


@pytest.mark.tier_b
def test_rigid_motion_carries_no_stress():
    """A rigid translation plus rotation prescribed on the whole boundary is
    reproduced inside, and the sheet stress vanishes."""
    mesh = _box()
    x, y = mesh.X
    omega, cx, cy = sympy.Rational(3, 10), sympy.Rational(1, 5), sympy.Rational(-1, 7)
    rigid = [cx - omega * y, cy + omega * x]
    sheet = _sheet(mesh)
    for boundary in ("Left", "Right", "Top", "Bottom"):
        sheet.add_essential_bc(rigid, boundary)
    sheet.solve()

    pts = _probe_points()
    u = uw.function.evaluate(sheet.u.sym, pts).reshape(-1, 2)
    exact = np.column_stack([float(cx) - float(omega) * pts[:, 1],
                             float(cy) + float(omega) * pts[:, 0]])
    assert np.abs(u - exact).max() < 1.0e-8
    sigma = uw.function.evaluate(sheet.stress, pts)
    assert np.abs(sigma).max() < 1.0e-6


@pytest.mark.tier_b
def test_basal_drag_balances_a_uniform_load():
    """With no kinematic boundary condition, a uniform body force against a
    basal drag c gives the uniform velocity u = L f / c exactly (L = 1)."""
    mesh = _box()
    sheet = _sheet(mesh)
    c = sympy.Rational(5, 2)
    f = (sympy.Rational(3, 4), sympy.Rational(-1, 3))
    sheet.basal_drag = c
    sheet.bodyforce = f
    sheet.solve()

    pts = _probe_points()
    u = uw.function.evaluate(sheet.u.sym, pts).reshape(-1, 2)
    exact = np.array([float(f[0] / c), float(f[1] / c)])
    assert np.abs(u - exact).max() < 1.0e-8


@pytest.mark.tier_b
def test_manufactured_solution_converges_at_third_order():
    """P2 velocity on a smooth manufactured solution: the L2 error of the
    sheet momentum balance with basal drag converges at order 3 in h."""
    errors = {}
    for h in (0.2, 0.1):
        mesh = _box(cell_size=h)
        x, y = mesh.X
        eta = sympy.Rational(3, 2)
        c = sympy.Integer(2)
        u_e = sympy.Matrix([[sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y),
                             sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y) / 3]])
        G = u_e.jacobian(sympy.Matrix([x, y]))
        E = (G + G.T) / 2
        sigma = 2 * eta * (E + E.trace() * sympy.eye(2))
        div_sigma = sympy.Matrix([[sum(sigma[i, j].diff(v) for j, v in enumerate((x, y)))
                                   for i in range(2)]])
        sheet = _sheet(mesh, eta=eta)
        sheet.basal_drag = c
        sheet.bodyforce = list(-div_sigma + c * u_e)
        for boundary in ("Left", "Right", "Top", "Bottom"):
            sheet.add_essential_bc(list(u_e), boundary)
        sheet.solve()
        err2 = uw.maths.Integral(mesh, ((sheet.u.sym - u_e) * (sheet.u.sym - u_e).T)[0, 0]).evaluate()
        errors[h] = np.sqrt(err2)

    order = np.log(errors[0.2] / errors[0.1]) / np.log(2.0)
    assert order > 2.6, f"convergence order {order:.2f}, errors {errors}"


@pytest.mark.tier_b
def test_basal_coupling_and_body_force_per_unit_volume():
    """The aligned forcing: a body force per unit VOLUME (integrated through
    the thickness L, as Stokes integrates it through a thick layer), a drag
    against a moving base and a basal traction. Uniform, so exactly
    u = u_ref + (L f + t) / c."""
    mesh = _box()
    sheet = _sheet(mesh)
    L, c = sympy.Integer(2), sympy.Rational(5, 2)
    f = (sympy.Rational(3, 4), sympy.Rational(-1, 3))
    t = (sympy.Rational(1, 5), sympy.Rational(1, 2))
    u_ref = (sympy.Rational(-1, 2), sympy.Rational(1, 7))
    sheet.thickness = L
    sheet.bodyforce = f
    sheet.add_basal_drag(c, "Bottom", reference_velocity=u_ref)
    sheet.add_basal_traction(t, "Bottom")
    sheet.solve()

    u = np.asarray(sheet.u.array).reshape(-1, 2)
    exact = np.array([float(u_ref[i] + (L * f[i] + t[i]) / c) for i in range(2)])
    assert np.abs(u - exact).max() < 1.0e-8


@pytest.mark.tier_b
def test_pressure_diagnostic_is_minus_the_trace():
    """Uniaxial stretch: the plane-stress pressure p = -tr(tau) = -eta edot,
    recovered into the diagnostic field ``sheet.p`` after the solve."""
    mesh = _box()
    x, y = mesh.X
    edot, eta = sympy.Rational(1, 2), 2
    sheet = _sheet(mesh, eta=eta)
    u_y = -edot * (y - sympy.Rational(1, 2)) / 2
    sheet.add_essential_bc([0, u_y], "Left")
    sheet.add_essential_bc([edot, u_y], "Right")
    p = sheet.p                               # created before the solve
    sheet.solve()
    assert np.abs(np.asarray(p.array) + eta * float(edot)).max() < 1.0e-6


@pytest.mark.tier_b
def test_plastic_sheet_yields_at_the_von_mises_stress():
    """A perfectly plastic sheet (viscosity far above tau_y / edot) in
    uniaxial stretch carries sigma_xx = sqrt(3) tau_y, the von Mises uniaxial
    yield stress: the invariant includes the out-of-plane strain rate
    eps_nn = -tr(eps_s) (with the in-plane invariant alone the sheet yields at
    sqrt(18/5) tau_y = 1.897 tau_y)."""
    mesh = _box()
    x, y = mesh.X
    edot, tau_y = sympy.Rational(1, 2), 1
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1.0e4
    sheet.constitutive_model.Parameters.yield_stress = tau_y
    sheet.tolerance = 1.0e-10
    u_y = -edot * (y - sympy.Rational(1, 2)) / 2
    sheet.add_essential_bc([0, u_y], "Left")
    sheet.add_essential_bc([edot, u_y], "Right")
    # start from the uniform stretch: the plastic viscosity is singular at rest
    sheet.u.array[:, 0, 0] = float(edot) * sheet.u.coords[:, 0]
    sheet.u.array[:, 0, 1] = -float(edot) * (sheet.u.coords[:, 1] - 0.5) / 2
    sheet.solve(zero_init_guess=False)

    sigma = uw.function.evaluate(sheet.stress, _probe_points()).reshape(-1, 2, 2)
    assert np.abs(sigma[:, 0, 0] - np.sqrt(3.0) * tau_y).max() < 1.0e-3
    assert np.abs(sigma[:, 1, 1]).max() < 1.0e-3


@pytest.mark.tier_b
def test_sheet_solves_on_a_refined_mesh_with_a_hierarchy():
    """On a mesh with refinement levels the field decomposition (which the
    rotated solves address the field by) recurses into the coarse levels; it
    must run after the fields are copied there. Before, it raised PETSc error
    62 ("number of requested fields 1 greater than number of DM fields 0")."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, refinement=1, qdegree=3)
    sheet = _sheet(mesh)
    c = sympy.Rational(5, 2)
    sheet.basal_drag = c
    sheet.bodyforce = (1, 0)
    sheet.solve()
    u = np.asarray(sheet.u.array).reshape(-1, 2)
    assert np.abs(u - np.array([1 / 2.5, 0.0])).max() < 1.0e-8
