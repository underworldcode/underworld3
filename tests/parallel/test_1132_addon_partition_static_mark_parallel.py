"""A placed fault leaves the base's levels static at any rank count (#919).

In parallel, ``place_fault_ribbon_2d`` runs ``addon_partition.partition_addon`` on
the base mesh before it places the band. A base built with ``refinement=2`` has no
add-on levels, so the level that moves is the base's own finest. The moved copy was
marked ``_mg_static = False`` whatever the level had been. So ``Mesh.add_fault``
took the base's finest for an add-on level, and ``fault_mg.split_tail_levels``
tried to cut the S-sampled trace into it.

Measured on UWTN 2026-022's linear fixture at 320 rungs: the snap inverted a cell,
the level stayed welded, and every np > 1 run warned "the fault(s) could not be
carried onto add-on level 0 of 1". The serial run classifies the same level as
background and never tries.

Whether the replay is attempted therefore depended on the rank count. A replay that
succeeded would split a background level at np > 1 and leave it welded at np 1.

This test places the same fixture at 80 rungs. It asserts, at any rank count, that:

- the solver mesh's own tail has no add-on level;
- no tail level is split;
- the carry warning is not raised.

It places the fault and does not solve. The file carries no ``mpi(min_size=2)``
mark: at np 1 it checks the serial baseline, and in the parallel batches the
partition.

    python -m pytest tests/parallel/test_1132_addon_partition_static_mark_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1132_addon_partition_static_mark_parallel.py
"""
import warnings

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.fault_mg import addon_level_indices
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80


def test_a_placed_fault_leaves_the_base_levels_static():
    length = float(np.linalg.norm(B - A))
    spacing = length / N_RUNGS
    trace = A + (np.arange(N_RUNGS + 1) * spacing)[:, None] * (B - A) / length
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=True,
        qdegree=2, refinement=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = place_fault_ribbon_2d(base, [("Fault", trace)], 2 * spacing,
                                     margin_rings=1, clearance=0.7, split=True,
                                     mesher="ladder")[0]
    carried = [str(w.message) for w in caught
               if "could not be carried onto add-on level" in str(w.message)]
    tail = list(getattr(mesh, "_custom_mg_coarse_meshes", None) or [])
    # the base's three levels (1/8, 1/16, 1/32), all static background
    assert len(tail) == 3, len(tail)
    assert addon_level_indices(tail) == [], addon_level_indices(tail)
    assert not any(getattr(m, "_fault_point_pairs", None) for m in tail)
    assert carried == [], carried
