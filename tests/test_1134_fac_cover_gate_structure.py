"""The FAC cover gate decides on structure, not on the patch's size (#918).

The finest level's FAC patch is the rows its transfer from the coarse level
cannot represent, plus a one-cell halo. The gate exists to skip uniform
refinements, where whole-level smoothing is the right configuration. It used to
decline any patch whose subdomain (patch plus halo) exceeded 75 % of the level's
rows, and it said nothing when it did. A locally refined fault level with a fine
band crosses 75 % by size alone: on UWTN 2026-022's linear fixture at
h_b/S = 48 the patch was declined and the velocity solve took 909 iterations
against 66 with the patch.

The size of the patch cannot tell the two cases apart; the background outside
the subdomain can. On a uniform P2 pair the halo reaches every row, so nothing
lies outside the subdomain. A locally refined level keeps an untouched
background however large its patch: 38 %, 24 % and 14 % of the rows at the three
ratios of #918. The gate now declines only when the background outside the
subdomain is under 2 % of the level, and it records its verdict on the
hierarchy (``hierarchy.fac_gate``) either way.

The fixture is test_1017's NVB band hierarchy with a band wide enough that its
subdomain exceeds 75 % of the finest level while a background remains outside it.
"""
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg

from test_1017_nested_native_and_fac import _nvb_band, _poisson, _wrap

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# the band half-width, and the finest level's rows and subdomain rows with it:
# 355 of 417 is 85 %, over the old 75 % gate, with 62 background rows outside
BAND = 0.3
FINEST_ROWS, FINEST_SUBDOMAIN = 417, 355
# the uniform level's rows, all inside the subdomain
UNIFORM_ROWS = 225


def _wide_band_solve():
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=True, qdegree=2)
    dm1 = m0.dm.refine()
    dm2 = _nvb_band(dm1, m0, BAND)
    s = _poisson(_wrap(dm2, m0))
    custom_mg.set_custom_fmg(s, [_wrap(m0.dm, m0), _wrap(dm1, m0)], builder="barycentric")
    s.solve()
    assert s.snes.getConvergedReason() > 0
    return s._custom_mg["hierarchy"]


def test_a_large_patch_with_a_background_is_taken():
    h = _wide_band_solve()
    n2 = h.transfers[1].getSize()[0]
    patch = h.level_patch_rows[2]
    assert patch is not None, (
        "the band level's patch was declined although a background lies outside "
        "its subdomain (#918)")
    owned, sub = patch
    assert (n2, len(sub)) == (FINEST_ROWS, FINEST_SUBDOMAIN)
    gate = h.fac_gate[2]
    assert gate["taken"]
    assert (gate["rows"], gate["subdomain_rows"]) == (FINEST_ROWS, FINEST_SUBDOMAIN)


def test_the_uniform_level_is_declined_and_recorded():
    h = _wide_band_solve()
    assert h.level_patch_rows[1] is None
    gate = h.fac_gate[1]
    assert not gate["taken"]
    # the halo of a uniform pair reaches every row
    assert h.transfers[0].getSize()[0] == UNIFORM_ROWS
    assert (gate["rows"], gate["subdomain_rows"]) == (UNIFORM_ROWS, UNIFORM_ROWS)
