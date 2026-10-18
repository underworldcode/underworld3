"""Parallel (MPI) run of test_1078: the Micromorphic regularisation option against
the hand-built formulation, and the multi-material mask. The option's field is
created and registered on every rank in the same order, so the coupled split is the
same on every partition.

    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1078_micromorphic_regularisation_mpi.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import test_1078_micromorphic_regularisation as serial  # noqa: E402

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(600),
              pytest.mark.level_2, pytest.mark.tier_b]


@pytest.mark.mpi(min_size=2)
def test_parallel_option_reproduces_the_hand_built_formulation():
    serial.test_option_reproduces_the_hand_built_formulation()


@pytest.mark.mpi(min_size=2)
def test_parallel_multimaterial_mask():
    serial.test_multimaterial_masks_each_source_by_its_material()
