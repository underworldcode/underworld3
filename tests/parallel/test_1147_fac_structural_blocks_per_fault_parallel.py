"""Two faults apart are two FAC blocks, each on the rank that owns most of it (#895).

With no ``fac_zone`` keyed, the finest level's FAC patch is the structural one: the
split faults' bands, their duplicates and a halo. The parallel branch gathers it
from every rank so that no partition boundary cuts it (#925), and it was one block
for all faults, factorised on the rank owning most of its rows. The add-on partition
gives each fault to the rank owning most of it, so two faults apart sat on two ranks
and both were factorised on one: on #895's stage 1 (kaiju, h_b/S 48, np 4 to 16)
that rank's factorisation and triangular solves did twice the work of one fault's,
and the solve took 19 to 28 % longer than with a block per fault. The slip agreed to
5e-14 in the same iterations.

The structural patch is now cut into connected pieces, rows that reference one coarse
DOF or are DOFs of one node being one piece, and each piece goes to the rank owning
most of its rows; the pieces one rank receives are one block. The fixture is #895's ``apart`` case at n_rungs 80:
UWTN 2026-022's problem A box with its fault shifted to (0.08, 0.20)-(0.48, 0.30) and
a second, equal fault at (0.52, 0.70)-(0.92, 0.80), both split; the near pair is the
problem A fault and a copy 0.12 below it.

- at np 2 and above the structural patch is two blocks, every patch row held once,
  each block on the rank owning most of its rows, and the faults' two blocks on two
  ranks where the partition gave the faults to two ranks;
- at np 1 the serial branch keeps one block;
- two faults 0.12 apart stay one block: their pieces land on one rank, and they are
  coupled in the operator, so cut apart they took 4 velocity iterations against 3;
- :func:`_connected_pieces` joins rows through the keys they share.

This file carries no ``mpi(min_size=2)`` mark: at np 1 it checks the serial
branch, in the parallel batches the parallel one.

    python -m pytest tests/parallel/test_1147_fac_structural_blocks_per_fault_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1147_fac_structural_blocks_per_fault_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import custom_mg
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80
S = float(np.linalg.norm(B - A)) / N_RUNGS
T_HAT = (B - A) / np.linalg.norm(B - A)
N_HAT = np.array([T_HAT[1], -T_HAT[0]])


def _solve(traces):
    """Solve the box with ``traces`` split; return the solver and, per rank, its
    finest-level blocks as (owned, subdomain) global rows."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(base, traces, 2 * S, spines=traces, margin_rings=1,
                                 clearance=0.7, split=True, mesher="ladder")[0]
    _x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    for label, _trace in traces:
        stokes.add_fault_bc(0, boundary=label)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    entry = stokes._custom_mg["hierarchy"].level_patch_rows[-1]
    pairs = [] if entry is None else entry if isinstance(entry, list) else [entry]
    mine = [(np.asarray(o, dtype=np.int64), np.asarray(s, dtype=np.int64))
            for o, s in pairs if np.size(s)]
    return stokes, uw.mpi.comm.allgather(mine)


def _trace(start):
    return start + (np.arange(N_RUNGS + 1) * S)[:, None] * T_HAT


@pytest.fixture(scope="module")
def apart():
    return _solve([("Fault", _trace(A + np.array([-0.22, -0.25]))),
                   ("Second", _trace(A + np.array([0.22, 0.25])))])


@pytest.fixture(scope="module")
def near():
    return _solve([("Fault", _trace(A)), ("Second", _trace(A + 0.12 * N_HAT))])


@pytest.mark.tier_b
def test_faults_apart_are_a_block_each_in_parallel(apart):
    stokes, per_rank = apart
    blocks = [s for rank in per_rank for _o, s in rank]
    expected = 1 if uw.mpi.size == 1 else 2
    assert len(blocks) == expected, (
        f"np {uw.mpi.size}: the structural patch is in blocks of "
        f"{sorted(b.size for b in blocks)}; expected {expected}, one per fault in "
        f"parallel (#895)")
    _rows, count = np.unique(np.concatenate(blocks), return_counts=True)
    assert count.max() == 1, (
        f"np {uw.mpi.size}: {int(np.sum(count > 1))} patch rows are held by two "
        f"blocks (#925)")
    # the gate counts every rank's patch and halo rows before they are gathered
    gate = stokes._custom_mg["hierarchy"].fac_gate[-1]
    assert count.size == gate["subdomain_rows"], (count.size, gate["subdomain_rows"])


@pytest.mark.tier_b
@pytest.mark.skipif(uw.mpi.size < 2, reason="ownership is a parallel question")
def test_each_block_is_on_the_rank_owning_most_of_it(apart):
    stokes, per_rank = apart
    # the split faults solve through the rotated workspace, whose KSP carries the
    # hierarchy; the SNES's own fieldsplit is never set up (PETSc error 73)
    cache = getattr(stokes, "_rotated_linear_cache", None)
    ksp = cache["ctx"]["ksp"] if cache else stokes.snes.getKSP()
    velocity = ksp.getPC().getFieldSplitSubKSP()[0].getPC()
    finest = velocity.getMGSmoother(velocity.getMGLevels() - 1).getOperators()[0]
    starts = np.array(uw.mpi.comm.allgather(finest.getOwnershipRange()[0]))
    holders = []
    for rank, blocks in enumerate(per_rank):
        for _owned, sub in blocks:
            owner = np.searchsorted(starts, sub, side="right") - 1
            majority = int(np.bincount(owner, minlength=len(starts)).argmax())
            assert majority == rank, (
                f"np {uw.mpi.size}: a block of {sub.size} rows is on rank {rank}, "
                f"rank {majority} owns most of it")
            holders.append(rank)
    # each fault's majority rank, read from the blocks: two ranks when the add-on
    # partition gave the faults to two ranks (it does at np 2 and 4 here)
    assert len(set(holders)) == 2, (
        f"np {uw.mpi.size}: both faults' blocks are on rank(s) {sorted(set(holders))} "
        f"(#895)")


@pytest.mark.tier_b
def test_faults_near_each_other_stay_one_block(near):
    stokes, per_rank = near
    blocks = [s for rank in per_rank for _o, s in rank]
    assert len(blocks) == 1, (
        f"np {uw.mpi.size}: two faults 0.12 apart are in blocks of "
        f"{sorted(b.size for b in blocks)}; their pieces land on one rank and are "
        f"coupled in the operator (#895 review, M1)")
    info = stokes._rotated_freeslip_info
    assert info["converged"]
    its = max(info["vel_its_last"])
    # 3 at np 1 to 4 as one block; cut into two pieces on one rank it took 4
    assert its <= 3, f"np {uw.mpi.size}: the velocity solve took {its} iterations"


@pytest.mark.tier_b
def test_connected_pieces_join_rows_through_shared_keys():
    rows = np.array([3, 4, 7, 8, 9, 12])
    # 3-4 share coarse column 0, 4-7 a fine row key, 8-9 coarse column 5; 12 alone;
    # the link from row 20 is outside the rows and ignored
    link_rows = np.array([3, 4, 4, 7, 8, 9, 12, 20])
    link_keys = np.array([-1, -1, 4, 4, -6, -6, 12, -6])
    pieces = custom_mg._connected_pieces(rows, link_rows, link_keys)
    assert [p.tolist() for p in pieces] == [[3, 4, 7], [8, 9], [12]]
