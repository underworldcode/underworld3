"""The rotated Newton/Picard loop honours a user-set ``snes_max_it`` (#920).

A solve with rotated free slip or a split fault runs its own outer loop,
``rotated_bc.solve_rotated_freeslip``, whose cap defaults to 50. The solver
called it without passing a cap, so a user's ``snes_max_it`` never reached it,
while the standard path resolves the same key through
``_resolve_snes_max_it`` (#483). On the add-on matrix's yielding-lid box, Coulomb
and rate-and-state faults stopped at 50 iterations still in their Picard
phase, and only wrapping the function in the driver raised the cap.

The fixture is test_1018's power-law box (eta = eps_II^(1/3 - 1)) under rotated
free slip on every wall, with the frozen (Picard) tangent, which converges in
more than three iterations. With ``snes_max_it = 3`` the loop must stop after
exactly three increments and report that it did not converge; the same solve
without the cap converges.
"""
import pytest

import underworld3 as uw

from test_1018_rotated_freeslip import _powerlaw_stokes

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

CAP = 3


def _rotated_powerlaw_box(prefix):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    s, _v, _p = _powerlaw_stokes(mesh, prefix, cj=False)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    return s


def test_the_uncapped_rotated_loop_needs_more_than_the_cap():
    s = _rotated_powerlaw_box("free")
    s.solve()
    info = s._rotated_freeslip_info
    assert info["converged"]
    assert info["nonlinear_iterations"] > CAP, (
        "the fixture converges within the cap, so the capped test cannot fail")


def test_snes_max_it_caps_the_rotated_loop():
    s = _rotated_powerlaw_box("capped")
    s.petsc_options["snes_max_it"] = CAP
    s.solve()
    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] == CAP, (
        f"the rotated loop ran {info['nonlinear_iterations']} iterations under "
        f"snes_max_it = {CAP} (#920)")
    assert not info["converged"]
