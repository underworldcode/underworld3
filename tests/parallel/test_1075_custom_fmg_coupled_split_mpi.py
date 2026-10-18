"""Parallel (MPI) run of test_1075: custom non-nested FMG on a velocity split that
also holds a coupled field. The independent coarse box is partitioned on its own,
so its transfer is the cross-partition one; the per-field transfers are scattered
into the split's numbering through owned index sets on every rank.

    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1075_custom_fmg_coupled_split_mpi.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from test_1075_custom_fmg_coupled_split import check_custom_fmg_on_coupled_split  # noqa: E402

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(300),
              pytest.mark.level_2, pytest.mark.tier_b]


@pytest.mark.mpi(min_size=2)
def test_parallel_custom_fmg_on_a_coupled_split_matches_the_native_hierarchy():
    check_custom_fmg_on_coupled_split()
