"""A level without a split keeps the FAC cover gate, and both verdicts are recorded (#918).

The cover gate declines a FAC patch whose subdomain (patch plus halo) exceeds
75 % of the level's rows, so that a uniform refinement, whose halo reaches every
row, keeps whole-level smoothing. A split fault level now takes its patch
whatever its size (``tests/parallel/test_1134_fac_cover_gate_split_parallel.py``).
A level without a split keeps the gate: the measurements that moved it were on
split levels, and on a wide adapt child the patch and halo reach 95-98 % of the
level, an LU over nearly all of it. Each verdict is recorded on
``hierarchy.fac_gate`` either way.

The fixture is test_1017's NVB band hierarchy (Poisson) with the band widened to
half-width 0.3. The band level has 417 rows, and its patch and halo hold 355 of
them (85 %), over the gate, so it declines. The uniform level below it has 225
rows (the interior P2 nodes of the 8 x 8 grid), all in its subdomain.
"""
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg

from test_1017_nested_native_and_fac import _nvb_band, _poisson, _wrap

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

BAND = 0.3
BAND_ROWS, BAND_SUBDOMAIN = 417, 355
UNIFORM_ROWS = 225


def _wide_band_hierarchy():
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=True, qdegree=2)
    dm1 = m0.dm.refine()
    dm2 = _nvb_band(dm1, m0, BAND)
    s = _poisson(_wrap(dm2, m0))
    custom_mg.set_custom_fmg(s, [_wrap(m0.dm, m0), _wrap(dm1, m0)], builder="barycentric")
    s.solve()
    assert s.snes.getConvergedReason() > 0
    return s._custom_mg["hierarchy"]


def test_an_unsplit_band_over_the_gate_is_declined_and_recorded():
    h = _wide_band_hierarchy()
    assert h.level_patch_rows[2] is None
    assert h.fac_gate[2] == {"rows": BAND_ROWS, "subdomain_rows": BAND_SUBDOMAIN,
                             "taken": False}


def test_the_uniform_level_is_declined_and_recorded():
    h = _wide_band_hierarchy()
    assert h.level_patch_rows[1] is None
    assert h.fac_gate[1] == {"rows": UNIFORM_ROWS, "subdomain_rows": UNIFORM_ROWS,
                             "taken": False}
