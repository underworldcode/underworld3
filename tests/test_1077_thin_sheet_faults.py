"""Split-node faults in a thin viscous sheet: closed-form checks.

The thin viscous sheet is plane-stress elasticity with nu = 1/2 (velocity for
displacement, eta for G), so its effective modulus is E' = 2 eta (1 + nu) =
3 eta — the Trouton value — and the crack solutions of plane-stress elasticity
carry over. A frictionless fault is the mode-II crack whose faces stay in
contact: zero shear traction, continuous normal velocity and normal traction.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities.fault_contact import fault_pair_jumps, fault_slip

pytestmark = [pytest.mark.level_2]


def _box(cell_size):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size, qdegree=3)


def _sheet(mesh, eta=1):
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = eta
    sheet.tolerance = 1.0e-10
    return sheet


def _probe_points(n=8):
    # an even count keeps the grid off the fault line y = 0.5 (and x = 0.5),
    # where a point evaluation on coincident split nodes is ambiguous
    s = np.linspace(0.1, 0.9, n)
    return np.array([[a, b] for a in s for b in s])


@pytest.mark.tier_b
@pytest.mark.parametrize("fault", [
    np.array([[0.3, 0.5], [0.7, 0.5]]),      # along the stretch
    np.array([[0.5, 0.3], [0.5, 0.7]]),      # across it
])
def test_a_fault_free_of_shear_traction_changes_nothing(fault):
    """Under uniaxial stretch the principal axes are x and y, so a frictionless
    fault along either carries no shear traction: the faulted solution is the
    unfaulted one (u = (edot x, -edot (y - 1/2) / 2)) and the slip is zero."""
    split = _box(0.1).add_fault(("Flt", fault))
    x, y = split.X
    edot = sympy.Rational(1, 2)
    u_y = -edot * (y - sympy.Rational(1, 2)) / 2
    sheet = _sheet(split, eta=2)
    sheet.add_essential_bc([0, u_y], "Left")
    sheet.add_essential_bc([edot, u_y], "Right")
    sheet.add_fault_bc(0, "Flt")
    sheet.solve()
    info = sheet._rotated_freeslip_info
    assert info["converged"]

    # The exact field is linear, so the P2 nodal values carry it exactly.
    # TODO(BUG): uw.function.evaluate mislocates points within ~h/2 of the slit
    # of a split mesh (coincident nodes) and returns O(1e-3) errors there while
    # the nodal solution is exact to 1e-9 (measured 2026-10-08, cellSize 0.1).
    c = sheet.u.coords
    u = sheet.u.array[:, 0, :]
    exact = np.column_stack([0.5 * c[:, 0], -0.5 * (c[:, 1] - 0.5) / 2])
    assert np.abs(u - exact).max() < 1.0e-8
    _s, V, leak = fault_slip(sheet, "Flt", info)
    assert np.abs(V).max() < 1.0e-8
    assert np.abs(leak).max() < 1.0e-13


def _remote_shear(split, eta_f):
    """Remote simple shear u = (y, 0) (tau_inf = eta * 1) on every boundary with
    an interior fault carrying the viscous law tau = eta_f V (0 = frictionless)."""
    x, y = split.X
    sheet = _sheet(split, eta=1)
    for boundary in ("Left", "Right", "Top", "Bottom"):
        sheet.add_essential_bc([y, 0], boundary)
    sheet.add_fault_bc(eta_f, "Flt")
    sheet.solve()
    info = sheet._rotated_freeslip_info
    assert info["converged"]
    return sheet, info


@pytest.mark.tier_b
def test_a_viscous_fault_approaches_the_welded_sheet_at_first_order():
    """tau = eta_f V penalises the slip: the slip, and with it the deviation
    from the unfaulted solution (u = (y, 0) exactly), is first order in
    1 / eta_f."""
    split = _box(0.1).add_fault(("Flt", np.array([[0.3, 0.5], [0.7, 0.5]])))
    pts = _probe_points()
    exact = np.column_stack([pts[:, 1], np.zeros(len(pts))])
    slip, deviation = {}, {}
    for eta_f in (1.0e2, 1.0e3):
        sheet, info = _remote_shear(split, eta_f)
        _s, V, _leak = fault_slip(sheet, "Flt", info)
        slip[eta_f] = np.abs(V).max()
        u = uw.function.evaluate(sheet.u.sym, pts).reshape(-1, 2)
        deviation[eta_f] = np.abs(u - exact).max()
    order = np.log(slip[1.0e2] / slip[1.0e3]) / np.log(10.0)
    assert order > 0.9, f"slip order in 1/eta_f {order:.2f}: {slip}"
    assert deviation[1.0e3] < deviation[1.0e2]


@pytest.mark.tier_b
def test_frictionless_fault_slips_as_the_griffith_crack():
    """A frictionless fault of half-length a under remote shear tau slips as
    the mode-II crack of plane-stress elasticity, delta u = (4 tau / E')
    sqrt(a^2 - x^2) with E' = 3 eta. Infinite-plane solution: the box is
    2 x 2 against a = 0.2 (walls at 5 a) and the P2 tip resolution is h = 1/25,
    which the 8% margins cover (measured 2026-10, recorded in the commit that
    added this test)."""
    a = 0.2
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 25, qdegree=3)
    split = mesh.add_fault(("Flt", np.array([[-a, 0.0], [a, 0.0]])))
    sheet, info = _remote_shear(split, 0)
    coords, jumps, normals = fault_pair_jumps(sheet, "Flt", info)
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])
    V = np.einsum("ij,ij->i", jumps, tangents)
    leak = np.einsum("ij,ij->i", jumps, normals)
    assert np.abs(leak).max() < 1.0e-13
    ellipse = (4.0 / 3.0) * np.sqrt(np.maximum(a**2 - coords[:, 0] ** 2, 0.0))
    peak = 4.0 * a / 3.0
    assert abs(np.abs(V).max() - peak) < 0.08 * peak, f"peak slip {np.abs(V).max():.4f} vs {peak:.4f}"
    rms = np.sqrt(np.mean((np.abs(V) - ellipse) ** 2))
    assert rms < 0.08 * peak, f"rms profile error {rms:.4f} vs peak {peak:.4f}"
