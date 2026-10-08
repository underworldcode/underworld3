"""Split-node faults in a thin sheet on the surface of a sphere.

The fault is a great-circle arc cut into the SphericalManifold; its coincident
pairs carry the no-opening constraint along the in-surface conormal and the
tangency of both sides, and slip along the trace. A rigid rotation is a Killing
field (zero surface strain), so it loads a frictionless fault with no traction
at all: the faulted solution must be the rotation exactly, with zero slip. A
short arc under the shear of the toroidal flow slips as the Griffith mode-II
crack of the plane-stress sheet, E' = 3 eta, up to the curvature of the sphere.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities.fault_contact import fault_pair_jumps, fault_slip

pytestmark = [pytest.mark.level_2]


def _equator_arc(half_angle, n=9):
    phi = np.linspace(-half_angle, half_angle, n)
    return np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])


def _faulted_sphere(cell_size, half_angle):
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=cell_size, degree=1, qdegree=5)
    return sphere.add_fault(("Flt", _equator_arc(half_angle)))


def _sheet(split, eta=1):
    sheet = uw.systems.ThinSheet(split)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = eta
    sheet.tolerance = 1.0e-9
    return sheet


@pytest.mark.tier_b
def test_a_killing_field_leaves_the_fault_untouched():
    """u = omega x X has no surface strain, so the frictionless fault sees no
    traction: the solution is the rotation exactly, the slip and the opening
    are zero, and every node (both sides of the fault included) stays tangent."""
    split = _faulted_sphere(0.2, np.pi / 6)
    X = split.X
    x, y, z = X[0], X[1], X[2]
    omega = (sympy.Rational(1, 3), sympy.Rational(-1, 2), sympy.Rational(2, 7))
    rot = sympy.Matrix([[omega[1] * z - omega[2] * y,
                         omega[2] * x - omega[0] * z,
                         omega[0] * y - omega[1] * x]])
    sheet = _sheet(split)
    sheet.basal_drag = 2
    sheet.bodyforce = list(2 * rot)
    sheet.add_fault_bc(0, "Flt")
    sheet.solve()
    info = sheet._rotated_freeslip_info
    assert info["converged"]
    c = sheet.u.coords
    exact = np.cross(np.array([1 / 3, -1 / 2, 2 / 7]), c)
    assert np.abs(sheet.u.array[:, 0, :] - exact).max() < 1.0e-7
    rhat = c / np.linalg.norm(c, axis=1)[:, None]
    assert np.abs((sheet.u.array[:, 0, :] * rhat).sum(axis=1)).max() < 1.0e-13
    _s, V, leak = fault_slip(sheet, "Flt", info)
    assert np.abs(V).max() < 1.0e-7
    assert np.abs(leak).max() < 1.0e-13


@pytest.mark.tier_b
def test_short_arc_slips_as_the_griffith_crack():
    """The toroidal field u = rhat x grad(z^2) shears the equator with
    E_yz = -x: at the arc's centre (1, 0, 0) the fault-parallel shear traction
    is tau = 2 eta. A frictionless arc of half-length a slips as the plane-stress
    mode-II crack, (4 tau / 3 eta) sqrt(a^2 - s^2), with the curvature of the
    sphere (a / R = 0.25) and the basal screening inside the 12% margin
    (measured 2026-10-09, recorded in the commit that added this test)."""
    a = 0.25
    split = _faulted_sphere(0.05, a)
    X = split.X
    x, y, z = X[0], X[1], X[2]
    r = sympy.sqrt(x**2 + y**2 + z**2)
    rhat = sympy.Matrix([[x, y, z]]) / r
    P = sympy.eye(3) - rhat.T * rhat
    u_e = sympy.Matrix([[2 * y * z, -2 * x * z, 0]])
    G = u_e.jacobian(sympy.Matrix([x, y, z])) * P
    E = P * ((G + G.T) / 2) * P
    sigma = 2 * (E + E.trace() * P)
    Xs = [x, y, z]
    divs = sympy.Matrix([[sum(sigma[i, j].diff(Xs[k]) * P[k, j]
                              for j in range(3) for k in range(3)) for i in range(3)]])
    sheet = _sheet(split, eta=1)
    sheet.basal_drag = 1
    sheet.bodyforce = list((-divs + u_e).subs(r, 1))
    sheet.add_fault_bc(0, "Flt")
    sheet.solve()
    info = sheet._rotated_freeslip_info
    assert info["converged"]
    coords, jumps, normals = fault_pair_jumps(sheet, "Flt", info)
    leak = np.einsum("ij,ij->i", jumps, normals)
    assert np.abs(leak).max() < 1.0e-13
    s_arc = np.arctan2(coords[:, 1], coords[:, 0])          # arc length on the unit sphere
    V = np.linalg.norm(jumps, axis=1)
    tau = 2.0
    ellipse = (4.0 * tau / 3.0) * np.sqrt(np.maximum(a**2 - s_arc**2, 0.0))
    peak = 4.0 * tau * a / 3.0
    assert abs(V.max() - peak) < 0.12 * peak, f"peak slip {V.max():.4f} vs {peak:.4f}"
    rms = np.sqrt(np.mean((V - ellipse) ** 2))
    assert rms < 0.12 * peak, f"rms profile error {rms:.4f} vs peak {peak:.4f}"


@pytest.mark.tier_b
def test_a_viscous_interface_law_penalises_the_slip_on_the_sphere():
    """tau = eta_f V on the arc under the toroidal shear: the interface law is
    assembled on the edge facets of the surface mesh (one slip component along
    the trace), and the slip falls at first order in 1 / eta_f."""
    a = 0.25
    split = _faulted_sphere(0.1, a)
    X = split.X
    x, y, z = X[0], X[1], X[2]
    r = sympy.sqrt(x**2 + y**2 + z**2)
    rhat = sympy.Matrix([[x, y, z]]) / r
    P = sympy.eye(3) - rhat.T * rhat
    u_e = sympy.Matrix([[2 * y * z, -2 * x * z, 0]])
    G = u_e.jacobian(sympy.Matrix([x, y, z])) * P
    E = P * ((G + G.T) / 2) * P
    sigma = 2 * (E + E.trace() * P)
    Xs = [x, y, z]
    divs = sympy.Matrix([[sum(sigma[i, j].diff(Xs[k]) * P[k, j]
                              for j in range(3) for k in range(3)) for i in range(3)]])
    slip = {}
    for eta_f in (1.0e1, 1.0e2):
        sheet = _sheet(split, eta=1)
        sheet.basal_drag = 1
        sheet.bodyforce = list((-divs + u_e).subs(r, 1))
        sheet.add_fault_bc(eta_f, "Flt")
        sheet.solve()
        info = sheet._rotated_freeslip_info
        assert info["converged"]
        _s, V, leak = fault_slip(sheet, "Flt", info)
        assert np.abs(leak).max() < 1.0e-13
        slip[eta_f] = np.abs(V).max()
    order = np.log(slip[1.0e1] / slip[1.0e2]) / np.log(10.0)
    assert order > 0.8, f"slip order in 1/eta_f {order:.2f}: {slip}"
