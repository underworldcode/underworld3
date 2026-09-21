"""The forward semi-Lagrangian stress history on the case that rings.

Waters and King start-up below Courant one on a pure Maxwell element, to t 6.5,
where the unsmoothed integration-point history has rung (0.626 at 1/16) and the
nodal one sits at 0.5171. The forward history transports its memory
consistently and has the same cell-scale mode: unsmoothed it diverges at t 2.4
here. With its read-back smoothing at c = 0.023 (the dose the integration-point
store needs) it runs clean to t 8. Twenty minutes: level 3.
"""
import pytest

from test_1060_stress_store_smoothing import waters_king_start_up

pytestmark = [pytest.mark.level_3, pytest.mark.tier_b]


def test_the_forward_history_with_its_read_back_smoothing_holds_the_maxwell_start_up():
    u1, _, u65 = waters_king_start_up(0.023, t_end=6.5, transport="forward")
    # measured with this dose at 1/16, dt 0.0125: 0.95427 at t 1.00, 0.51851 at t 6.5
    # (nodal 0.96215 and 0.51710)
    assert abs(u1 - 0.9543) < 0.003
    assert abs(u65 - 0.5185) < 0.003
