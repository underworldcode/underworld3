"""A semi-Lagrangian history leaves a static field where it is, serially (#1237).

The serial half of tests/parallel/test_1244_semi_lagrangian_static_field_parallel.py.
That file's docstring has the measurements. Serially the nudged record moved x + y by
7.498e-5 in two steps; without the nudge it stays to the solve tolerance.
"""
import os
import sys

import pytest

import underworld3 as uw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "parallel"))
from test_1244_semi_lagrangian_static_field_parallel import (  # noqa: E402
    TOLERANCE, drift_from_x_plus_y, two_static_steps)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.mark.parametrize("cells_along_a_side_of", [1, 3])
def test_a_static_field_stays_put_serially(cells_along_a_side_of):
    """The triangle sized as the parallel test sizes it at np 1 and at np 3."""
    uw.reset_default_model()
    _, values, X = two_static_steps(1, f"s{cells_along_a_side_of}", size=cells_along_a_side_of)
    assert drift_from_x_plus_y(values, X) < 100 * TOLERANCE
