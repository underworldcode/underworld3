"""A sheet on the sphere moving along its own normal: the prescribed normal
velocity is the datum of the tangency constraint, and the in-plane strain rate
carries its curvature term. Closed forms on the unit sphere for a uniform
membrane with no basal drag (the tidally flexed shell over an ocean):

- breathing, w = W: no tangential motion, strain rate W P, sheet stress
  sigma = 2 eta (eps + tr eps P) = 6 eta W P;
- a degree-l spheroidal pattern, w = W Y_l: tangential velocity
  v = 3 W / (2 l (l + 1) - 1) grad_s Y_l (l = 2: 3W/11), from div_s sigma = 0
  with div_s Hess_s Y = grad_s(Delta_s Y) + grad_s Y on the unit sphere.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _sphere(cell_size, degree=2):
    return uw.meshing.SphericalManifold(radius=1.0, cellSize=cell_size, degree=1,
                                        qdegree=2 * degree + 1)


def _sheet(sphere, eta=1.0):
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = eta
    sheet.tolerance = 1.0e-9
    return sheet


def _split(sheet):
    """Velocity at the nodes split into its normal and tangential parts."""
    c = np.asarray(sheet.u.coords)
    u = np.asarray(sheet.u.array[:, 0, :])
    rhat = c / np.linalg.norm(c, axis=1)[:, None]
    un = (u * rhat).sum(axis=1)
    return c, rhat, un, u - un[:, None] * rhat


def test_breathing_sphere():
    """w = W everywhere: the velocity is W r^ at every node to solver
    precision, no tangential flow, and the sheet stress is 6 eta W P."""
    W, eta = 0.3, 2.0
    sphere = _sphere(0.25)
    sheet = _sheet(sphere, eta)
    sheet.normal_velocity = W
    sheet.solve()
    assert sheet._rotated_freeslip_info["converged"]
    c, rhat, un, ut = _split(sheet)
    assert np.abs(un - W).max() < 1.0e-9
    # the curvature term is discretised on flat facets through the nodal normal
    # field, so a small tangential flow of order h^2 remains (0.25% at h = 0.25)
    assert np.abs(ut).max() < 1.0e-2 * W
    # the stress: evaluate the sheet's in-plane stress at a few cell interiors
    pts = np.asarray(sphere.X.coords)[::97]
    pts = pts / np.linalg.norm(pts, axis=1)[:, None]
    sig = np.asarray(uw.function.evaluate(sheet.stress, pts)).reshape(len(pts), 3, 3)
    P = np.eye(3)[None] - pts[:, :, None] * pts[:, None, :]
    assert np.abs(sig - 6 * eta * W * P).max() < 0.1 * 6 * eta * W


def test_degree_two_membrane():
    """w = W Y20, Y20 = (3 z^2 - 1)/2: the tangential velocity is
    (3W/11) grad_s Y20 = (9W/11) z (z^ - z r^), second order in h."""
    W = 0.5
    errs = []
    for h in (0.3, 0.15):
        sphere = _sphere(h)
        x, y, z = sphere.X
        sheet = _sheet(sphere)
        sheet.normal_velocity = W * (3 * z ** 2 - 1) / 2
        sheet.solve()
        assert sheet._rotated_freeslip_info["converged"]
        c, rhat, un, ut = _split(sheet)
        zz = c[:, 2]
        assert np.abs(un - W * (3 * zz ** 2 - 1) / 2).max() < 1.0e-8
        exact = (9 * W / 11) * zz[:, None] * (np.array([0.0, 0.0, 1.0])[None] - zz[:, None] * rhat)
        errs.append(np.abs(ut - exact).max() / np.abs(exact).max())
    # measured: 7.2% (h 0.3), 3.4% (0.2), 2.5% (0.15) in the max norm; the
    # curvature term is first order on flat facets, so the order is between 1 and 2
    assert errs[0] < 0.10
    assert errs[1] < 0.04
    assert errs[1] < 0.6 * errs[0]


def test_fault_reaction_is_the_membrane_resultant():
    """A split fault in the breathing sheet carries the membrane stress across
    it: the no-opening reaction, de-smeared by the trace mass, is the resultant
    L * 6 eta W on every pair (the trace mass once took the triangle's weights
    on a surface mesh and read 5 to 8 times that)."""
    from underworld3.utilities.fault_contact import fault_normal_traction
    W, eta, L = 0.1, 1.0, 0.08
    sphere = _sphere(0.15)
    phi = np.linspace(-0.3, 0.3, 9)
    split = sphere.add_fault(("Flt", np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])))
    sheet = _sheet(split, eta)
    sheet.thickness = L
    sheet.add_fault_bc(0, "Flt")
    sheet.normal_velocity = W
    sheet.solve()
    _c, sig = fault_normal_traction(sheet, "Flt", sheet._rotated_freeslip_info)
    expected = L * 6 * eta * W
    assert len(sig) > 5
    assert np.abs(sig - expected).max() < 0.02 * expected


@pytest.mark.level_3
def test_maxwell_transfer_function_under_a_tide():
    """A Maxwell membrane under the degree-2 eccentricity tide: the stress trace
    on the cos pattern has amplitude (24/11) mu A * wt / sqrt(1 + wt^2) and lags
    the displacement by atan(1/wt), wt = 2 pi De (the velocity is instantaneous:
    3/11 of the pattern's gradient whatever the rheology). Measured at 128 steps
    per period: amplitude within 1.5%, lag 32 deg against 28 at De 0.3; at the
    64 steps per period used here the lag is about 4 deg larger still (first
    order in the step)."""
    De, A, mu = 0.3, 0.01, 1.0
    steps, periods = 64, 3
    sphere = _sphere(0.3)
    x, y, z = sphere.X
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        sheet.Unknowns, order=1, integrator="etd", objective_rate="none")
    sheet.constitutive_model.Parameters.shear_modulus = mu
    sheet.constitutive_model.Parameters.shear_viscosity_0 = De
    sheet.thickness = 0.08
    sheet.tolerance = 1.0e-8
    pat_c = -sympy.Rational(3, 2) * (3 * z ** 2 - 1) / 2 + sympy.Rational(9, 4) * (x ** 2 - y ** 2)
    pat_s = 6 * x * y
    # projection of the stress trace on the cos pattern, area-weighted on a lat-lon grid
    lon = np.linspace(-np.pi, np.pi, 73)[:-1]; lat = np.linspace(-np.pi / 2, np.pi / 2, 37)[1:-1]
    LON, LAT = np.meshgrid(lon, lat)
    g = np.column_stack([(np.cos(LAT) * np.cos(LON)).ravel(), (np.cos(LAT) * np.sin(LON)).ravel(), np.sin(LAT).ravel()])
    wts = np.cos(LAT).ravel()
    pc = np.asarray(sympy.lambdify((x, y, z), pat_c, "numpy")(*g.T)).ravel()
    rhat = g / np.linalg.norm(g, axis=1)[:, None]
    P = np.eye(3)[None] - rhat[:, :, None] * rhat[:, None, :]
    dt = 1.0 / steps
    t, tc, ts = 0.0, [], []
    for k in range(steps * periods):
        t += dt
        c, s = float(np.cos(2 * np.pi * t)), float(np.sin(2 * np.pi * t))
        sheet.normal_velocity = A * 2 * np.pi * (-pat_c * s + pat_s * c)
        sheet.solve(timestep=dt, zero_init_guess=(k == 0))
        assert sheet._rotated_freeslip_info["converged"]
        if t > periods - 1 - 1e-9:
            S = np.asarray(uw.function.evaluate(sheet.stress, g * (1 - 0.3 * 0.3 ** 2))).reshape(-1, 3, 3)
            tr = np.einsum("nij,nij->n", P, S)
            tc.append((wts * tr * pc).sum() / (wts * pc ** 2).sum()); ts.append(t)
    ts, tc = np.array(ts), np.array(tc)
    M = np.column_stack([np.cos(2 * np.pi * ts), np.sin(2 * np.pi * ts)])
    a, b = np.linalg.lstsq(M, tc, rcond=None)[0]
    amp, lag = np.hypot(a, b), -np.degrees(np.arctan2(b, a))
    wt = 2 * np.pi * De
    amp_th, lag_th = (24 / 11) * mu * A * wt / np.sqrt(1 + wt ** 2), np.degrees(np.arctan(1 / wt))
    assert abs(amp - amp_th) < 0.05 * amp_th, (amp, amp_th)
    assert lag_th - 2 < lag < lag_th + 12, (lag, lag_th)
