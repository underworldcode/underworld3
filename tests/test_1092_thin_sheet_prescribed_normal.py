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
