"""An essential condition on a fault that also carries a law is refused on
every rank together (#940).

At np 2 the mixture ended in PETSc error 63 in ``Vec.setValues`` under the
interface assembler, after reading pooled local vectors at constrained pair
nodes. The refusal reads only the solver's registrations, which are the same on
every rank, so the same ValueError is raised everywhere before any read.
"""
import os
import sys

import pytest

import underworld3 as uw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_0940_essential_condition_on_a_fault_with_a_law import (  # noqa: E402
    _projection_with_a_law, _sheared)

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


@pytest.mark.parametrize("where", ["FltMinus", "Flt"])
def test_every_rank_refuses_together(where):
    s = _sheared()
    s.add_dirichlet_bc((0.0, 0.0), where)
    message = None
    try:
        s.solve()
    except ValueError as err:
        message = str(err)
    messages = uw.mpi.comm.allgather(message)
    assert messages[0] is not None and "#940" in messages[0], messages
    assert len(set(messages)) == 1, messages


def test_a_vector_solve_is_refused_on_every_rank_together():
    """At np 2 this ended in PETSc error 63 under residual_add, then a hang."""
    message = None
    try:
        _projection_with_a_law().solve()
    except ValueError as err:
        message = str(err)
    messages = uw.mpi.comm.allgather(message)
    assert messages[0] is not None and "#940" in messages[0], messages
    assert len(set(messages)) == 1, messages
