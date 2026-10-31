"""The rotated solve's fine transfer zeroes the opening rows of a split fault by
GLOBAL row of the velocity block.

``_build_rotated_custom_Pl`` forms Q_v P and zeroes the rows that carry the
constrained (opening) component. It took the row positions from
``vel_is.getIndices()`` — this rank's entries only — and passed them to
``zeroRows`` as global rows. Serial, where the two coincide, was right; at np=2
the wrong rows were zeroed, 232 coarse unknowns lost every fine image, and the
Galerkin product failed on empty rows (found 2026-10-13). Contract, at any rank
count: every opening row is zero. Since the add-on levels are split by default
the split tail builds its transfers in ``fault_mg``, so the test runs the welded
tail (``split_levels=False``, the code it was written for) and the split one.
Both drop the coarse columns no fine unknown reaches, so "no empty column" is
true by construction and is not asserted.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1086_rotated_fault_transfer_rows.py
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900)]


@pytest.mark.parametrize("split_levels", [False, True])
def test_rotated_fine_transfer_zeroes_opening_rows_by_global_row(split_levels):
    from underworld3.utilities import rotated_bc

    tip_a, tip_b = np.array([0.30, 0.45]), np.array([0.70, 0.55])
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1, qdegree=2)
    pts = np.column_stack([np.linspace(tip_a[0], tip_b[0], 9),
                           np.linspace(tip_a[1], tip_b[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface("guide", base, pts)
    guide.discretize()
    child = base.adapt(guide.refinement_metric_function(
        h_near=0.0125, h_far=0.05, width=0.08, profile="linear"), max_levels=2)
    mesh = child.add_fault([("Flt", np.vstack([tip_a, tip_b]))], split_levels=split_levels)

    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
    stokes.add_fault_bc(0, "Flt")
    stokes._build()
    Q, Qt, normal_rows, _ = rotated_bc.build_rotation(stokes, stokes._rotated_freeslip_bcs)
    Pl = rotated_bc._build_rotated_custom_Pl(stokes, Q, normal_rows)
    assert Pl is not None, "no multigrid hierarchy on the add-on mesh"
    P = Pl[-1]

    # the opening rows, as global rows of the velocity block
    vel_is = stokes._subdict["velocity"][0]
    rstart = P.getOwnershipRange()[0]
    position = {int(g): rstart + k for k, g in enumerate(vel_is.getIndices())}
    opening = [position[g] for g in normal_rows if g in position]
    ai, aj, av = P.getValuesCSR()
    nonzero_in_opening = sum(
        int(np.any(av[ai[r - rstart]:ai[r - rstart + 1]] != 0.0)) for r in opening)
    assert uw.mpi.comm.allreduce(len(opening)) > 0
    assert uw.mpi.comm.allreduce(nonzero_in_opening) == 0
