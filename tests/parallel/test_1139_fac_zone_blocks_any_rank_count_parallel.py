"""The finest level's FAC blocks are the same at any rank count (#925).

With ``fac_zone`` masks keyed, the finest level's FAC patch is cut into blocks:
the zone's rows segmented along strike at ``fac_block_rows``, and, on a level
that carries a split fault, the structural rows outside the zone as one more
block (#921). Basic ASM sums the correction of every block that holds a row.

In parallel each rank built its blocks from its own cells: its part of the zone,
with the ghost rows of its masked cells, segmented on its own, and its own
structural rows as one block. A row on a partition seam then sat in blocks on two
ranks, and rank boundaries cut both the zone and the structural block. On the
add-on matrix's row 5 (a TI band keyed beside a split fault, n_rungs 80) the
velocity solve took 4 iterations serially, 118 at np 2 and more than 200 at
np 4; at n_rungs 320 the outer solve never finished its first iteration at
np 4. Removing the rows held twice was not enough: the solve converged in
4-6 iterations at np 2 and 4 only once the blocks were the serial ones,
gathered across ranks.

The structural patch alone, with no zone keyed, was one block per rank: on the
same fixture its velocity solve took 3 iterations serially, 4 at np 2 and 6 at
np 4 (70, 96 and 143 summed over the solve). It converged, but a partition
boundary cut it all the same, and it too is built from the gathered rows. At
n_rungs 320, cut per rank, it took 67, 128, 131, 316 and 445 iterations at np 1,
2, 4, 8 and 16; as one block, 67 to 86.

The fixture is that problem at n_rungs 80: UWTN 2026-022's problem A box, the
main fault split, and a TI band (eta_1 = 0.01 along planes parallel to it) placed
0.08 below it as a ribbon, not cut, and keyed as the zone.

- split: every patch row is held by exactly one block over all ranks; there are
  two blocks, the zone's 1650 rows under the cap and the structural rows
  outside it, as in serial; and the velocity solve takes at most 5 iterations
  (4 measured at np 1, 2 and 4).
- placed but not split, the zone alone at ``fac_block_rows=500``: every row is
  held once, and the zone is cut into the fewest pieces of at most 500 rows,
  ``ceil(rows / 500)``, whatever the rank count;
- split with no zone keyed: the structural patch is one block holding every
  rank's patch and halo rows, as in serial, and the velocity solve takes at most
  3 iterations (3 measured at np 1, 2 and 4);
- placed but not split, a keyed mask holding no cell: no patch, as in serial. A
  keyed zone beats detection; the parallel branch fell back to the structural
  split when no rank held a zone cell.

This file carries no ``mpi(min_size=2)`` mark: at np 1 it checks the serial
branch, in the parallel batches the parallel one.

    python -m pytest tests/parallel/test_1139_fac_zone_blocks_any_rank_count_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1139_fac_zone_blocks_any_rank_count_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80
LENGTH = float(np.linalg.norm(B - A))
S = LENGTH / N_RUNGS
T_HAT = (B - A) / LENGTH
N_HAT = np.array([T_HAT[1], -T_HAT[0]])


def _solve_fixture(split, fac_block_rows, keyed="band"):
    """Solve the TI-band-beside-a-fault box with ``keyed`` as the zone: "band" (the TI
    band's cells), "empty" (a mask holding no cell) or None; return the solver
    and every rank's finest-level blocks (global rows)."""
    main = A + (np.arange(N_RUNGS + 1) * S)[:, None] * T_HAT
    band = A + 0.08 * N_HAT + np.linspace(0.0, 1.0, N_RUNGS + 1)[:, None] * (B - A)
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(
        base, [("Fault", main)], 2 * S, spines=[("Fault", main), ("TI", band)],
        margin_rings=1, clearance=0.7, split=split, mesher="ladder")[0]
    _x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

    zone = uw.discretisation.MeshVariable("zone", mesh, 1, degree=0, continuous=False)
    c = np.asarray(zone.coords)
    P, Q = band[0], band[-1]
    s = np.clip(((c - P) @ (Q - P)) / ((Q - P) @ (Q - P)), 0.0, 1.0)
    zone.array[:, 0, 0] = (np.linalg.norm(c - (P + s[:, None] * (Q - P)), axis=1)
                           < S).astype(float)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_viscosity_1 = 1.0 - 0.99 * zone.sym[0]
    stokes.constitutive_model.Parameters.director = sympy.Matrix([[N_HAT[0], N_HAT[1]]])
    stokes.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    if split:
        stokes.add_fault_bc(0, boundary="Fault")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6

    band_cells = None
    if keyed is not None:
        band_cells = np.asarray(mesh.cells_labelled("Band", 72), dtype=bool)
        if keyed == "empty":
            band_cells = np.zeros_like(band_cells)
    set_custom_fmg(stokes, levels, field_id=0, fac_zone=band_cells,
                   fac_block_rows=fac_block_rows)
    stokes.solve()
    entry = stokes._custom_mg["hierarchy"].level_patch_rows[-1]
    pairs = [] if entry is None else entry if isinstance(entry, list) else [entry]
    mine = [np.asarray(sub, dtype=np.int64) for _own, sub in pairs]
    blocks = [b for rank in uw.mpi.comm.allgather(mine) for b in rank if b.size]
    return stokes, blocks


def _times_held(blocks):
    """How many blocks hold each row, over all ranks."""
    _rows, count = np.unique(np.concatenate(blocks), return_counts=True)
    return count


@pytest.fixture(scope="module")
def split_level():
    return _solve_fixture(split=True, fac_block_rows=2000)


@pytest.mark.tier_b
def test_each_patch_row_is_in_one_block(split_level):
    _stokes, blocks = split_level
    count = _times_held(blocks)
    assert count.max() == 1, (
        f"{int(np.sum(count > 1))} of {count.size} patch rows are held by two or more "
        f"blocks at np {uw.mpi.size}; basic ASM adds every block's correction there (#925)")


@pytest.mark.tier_b
def test_the_blocks_are_the_serial_ones(split_level):
    _stokes, blocks = split_level
    sizes = sorted(int(b.size) for b in blocks)
    # serial: the zone's 1650 rows (macOS and Linux alike) fit under the cap as
    # one block, and the structural rows outside the zone are one more
    assert len(sizes) == 2 and sizes[0] == 1650, (
        f"np {uw.mpi.size}: block sizes {sizes}, serial has the zone's 1650 rows "
        f"and one structural block (#925)")


@pytest.mark.tier_b
def test_the_velocity_solve_takes_the_serial_iterations(split_level):
    stokes, _blocks = split_level
    info = stokes._rotated_freeslip_info
    assert info["converged"]
    its = max(info["vel_its_last"])
    # 4 measured at np 1, 2 and 4; rank-cut blocks took 118 at np 2 and
    # more than 200 at np 4
    assert its <= 5, f"np {uw.mpi.size}: the velocity solve took {its} iterations (#925)"


@pytest.mark.tier_b
def test_an_unsplit_zone_is_cut_into_the_same_pieces_at_any_rank_count():
    _stokes, blocks = _solve_fixture(split=False, fac_block_rows=500)
    count = _times_held(blocks)
    assert count.max() == 1, (
        f"{int(np.sum(count > 1))} of {count.size} zone rows are held by two or more "
        f"blocks at np {uw.mpi.size} (#925)")
    expected = int(np.ceil(count.size / 500))
    sizes = sorted(int(b.size) for b in blocks)
    assert len(blocks) == expected and sizes[-1] <= 500, (
        f"np {uw.mpi.size}: the zone's {count.size} rows are in blocks of {sizes}; "
        f"segmented along strike at 500 rows they are {expected} blocks of at most "
        f"500 (#925)")


@pytest.fixture(scope="module")
def structural_alone():
    return _solve_fixture(split=True, fac_block_rows=2000, keyed=None)


@pytest.mark.tier_b
def test_the_structural_patch_alone_is_one_block(structural_alone):
    stokes, blocks = structural_alone
    sizes = sorted(int(b.size) for b in blocks)
    assert len(sizes) == 1, (
        f"np {uw.mpi.size}: the structural patch is in blocks of {sizes}; serial "
        f"has one (#925)")
    # the gate counts every rank's patch and halo rows before they are gathered
    gate = stokes._custom_mg["hierarchy"].fac_gate[-1]
    assert sizes[0] == gate["subdomain_rows"], (
        f"np {uw.mpi.size}: the block holds {sizes[0]} rows, the ranks' patch and "
        f"halo rows are {gate['subdomain_rows']} (#925)")


@pytest.mark.tier_b
def test_the_structural_patch_alone_takes_the_serial_iterations(structural_alone):
    stokes, _blocks = structural_alone
    info = stokes._rotated_freeslip_info
    assert info["converged"]
    its = max(info["vel_its_last"])
    # 3 measured at np 1, 2 and 4 (kaiju) and np 1 (macOS); one block per rank
    # took 4 at np 2 and 6 at np 4
    assert its <= 3, f"np {uw.mpi.size}: the velocity solve took {its} iterations (#925)"


@pytest.mark.tier_b
def test_an_empty_keyed_zone_installs_no_patch():
    _stokes, blocks = _solve_fixture(split=False, fac_block_rows=2000, keyed="empty")
    sizes = sorted(int(b.size) for b in blocks)
    assert not sizes, (
        f"np {uw.mpi.size}: a keyed zone with no cells installed blocks of {sizes}; "
        f"serial installs none (#925)")
