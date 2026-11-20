"""Mesh generators take the refinement they are given (UNFILED 28's class).

A sweep of every generator in uw.meshing, each called with its defaults and
with refinement=1 (W1, 2026-11-21), found two that did not:

* DiscInternalBoundaries passed refinement=refinement, with a refinement
  callback, to Mesh, but had no refinement parameter: every call raised
  NameError ("name 'refinement' is not defined"). Its callback then read another
  generator's radiusOuter and radiusInner.
* BoxInternalBoundary documented and accepted refinement and passed 0 to Mesh:
  refinement=1 and 2 gave the unrefined mesh, with a one-level hierarchy.

Each now refines: refinement=1 gives a two-level hierarchy with 2**dim times
the cells (the disc is 2-D, the box's default 3-D).
"""
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

DISC = dict(radiusUpper=1.5, radiusInternal=1.0, radiusLower=0.5, cellSize=0.3)


def test_the_disc_builds_with_its_defaults():
    mesh = uw.meshing.DiscInternalBoundaries(**DISC)
    assert mesh.dm.getHeightStratum(0)[1] > 0


@pytest.mark.parametrize("make", [
    lambda refinement: uw.meshing.DiscInternalBoundaries(refinement=refinement, **DISC),
    lambda refinement: uw.meshing.BoxInternalBoundary(refinement=refinement),
], ids=["DiscInternalBoundaries", "BoxInternalBoundary"])
def test_refinement_one_gives_a_two_level_hierarchy(make):
    coarse, fine = make(None), make(1)
    assert len(fine.dm_hierarchy) == 2
    ratio = 2 ** fine.dm.getDimension()
    assert fine.dm.getHeightStratum(0)[1] == ratio * coarse.dm.getHeightStratum(0)[1]
