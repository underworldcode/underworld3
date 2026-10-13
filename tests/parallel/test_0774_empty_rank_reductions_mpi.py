"""Parallel regression tests for issue #405 — reductions on a zero-cell rank.

A rank that owns NO CELLS used to raise a rank-local ``ValueError`` from an
unguarded local reduction (``self._radii.min()`` and friends) while its
populated peers sat in the matching collective. The job then hung or aborted
asymmetrically. Every global quantity computed from rank-local data must
instead reduce across ranks, with the starved rank contributing the identity
element (+inf for a MIN, -inf for a MAX, 0 for a SUM) — so a rank with no
cells returns the SAME global answer as everyone else.

Run under MPI, e.g.::

    mpirun -np 2 python -m pytest --with-mpi \
        tests/parallel/test_0774_empty_rank_reductions_mpi.py
    mpirun -np 4 python -m pytest --with-mpi \
        tests/parallel/test_0774_empty_rank_reductions_mpi.py

The fixture is a 1x2 quad box — TWO cells for two-or-more ranks — so PETSc
must leave at least one rank empty at both np=2 and np=4. Every test is
timeout-guarded: the pre-fix failure mode is a hang, not an exception.
"""

import math

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(120),
    pytest.mark.level_1,
    pytest.mark.tier_a,
]

# Cell is 1.0 wide x 0.5 tall; the characteristic length UW3 reports is the
# centroid-to-corner half-diagonal. Analytic, so this is a real oracle rather
# than a recorded number.
SERIAL_RADIUS = math.sqrt(0.5 ** 2 + 0.25 ** 2)


def _starved_box():
    """A 2-cell mesh: at np >= 2 some rank is guaranteed to own no cells."""
    return uw.meshing.StructuredQuadBox(elementRes=(1, 2))


def _local_cell_count(mesh):
    cStart, cEnd = mesh.dm.getHeightStratum(0)
    return cEnd - cStart


def test_premise_some_rank_owns_no_cells():
    """Test premise: without a genuinely empty rank nothing here is tested."""
    mesh = _starved_box()
    counts = uw.mpi.comm.allgather(_local_cell_count(mesh))

    assert sum(counts) >= 2, f"fixture lost cells: {counts}"
    assert min(counts) == 0, (
        f"no rank was starved at np={uw.mpi.size}: cells per rank {counts} — "
        "this test cannot detect the #405 defect on this partition")
    assert max(counts) > 0, f"every rank starved: {counts}"


def test_radius_accessors_are_global_on_a_zero_cell_rank():
    """min/max/mean radius: same value on every rank, equal to the serial one."""
    mesh = _starved_box()

    r_min = mesh.get_min_radius()
    r_max = mesh.get_max_radius()
    r_mean = mesh.get_mean_radius()

    for name, value in (("min", r_min), ("max", r_max), ("mean", r_mean)):
        gathered = uw.mpi.comm.allgather(value)
        assert max(gathered) - min(gathered) < 1.0e-12, (
            f"get_{name}_radius disagrees across ranks: {gathered} — a "
            "starved rank must not return a rank-local answer")
        assert np.isclose(value, SERIAL_RADIUS, rtol=1.0e-9), (
            f"get_{name}_radius = {value}, expected the serial "
            f"{SERIAL_RADIUS}")


def test_negative_control_rank_local_minimum_would_be_caught():
    """Prove the cross-rank agreement assertion above has teeth.

    If ``get_min_radius`` returned this rank's own minimum (the pre-fix
    intent, minus the raise), the ranks would NOT agree — so the assertion
    in the previous test is not true by construction.
    """
    mesh = _starved_box()

    radii = np.asarray(mesh._radii).reshape(-1)
    rank_local_min = float(radii.min()) if radii.size else float("inf")
    gathered = uw.mpi.comm.allgather(rank_local_min)

    assert max(gathered) - min(gathered) > 1.0e-12, (
        f"rank-local minima {gathered} happen to agree, so this partition "
        "cannot distinguish a global answer from a rank-local one")
    assert math.isinf(max(gathered)), (
        "expected the starved rank's rank-local minimum to be the identity "
        f"element, got {gathered}")


def test_points_in_domain_answers_false_on_a_zero_cell_rank():
    """A rank with no cells contains no points — and must not raise."""
    mesh = _starved_box()

    # Interior of the lower cell (avoids the internal face at y = 0.5).
    query = np.array([[0.5, 0.25]])
    in_or_not = mesh.points_in_domain(query)

    n_local = _local_cell_count(mesh)
    if n_local == 0:
        assert not in_or_not.any(), (
            f"rank {uw.mpi.rank} owns no cells but claimed {query}")

    claims = uw.mpi.comm.allreduce(int(in_or_not.any()), op=uw.MPI.SUM)
    assert claims >= 1, "no rank claimed an interior point"


def test_quality_diagnostic_agrees_across_ranks():
    """quality() reduces globally; the branch choice must be rank-symmetric."""
    mesh = _starved_box()

    n_cells = mesh.quality()["n_cells"]
    gathered = uw.mpi.comm.allgather(n_cells)

    assert len(set(gathered)) == 1, (
        f"quality()['n_cells'] disagrees across ranks: {gathered}")
    assert n_cells == 2, f"expected the fixture's 2 cells, got {n_cells}"


def test_estimate_dt_agrees_across_ranks():
    """estimate_dt is the reason #405 was raised in priority: it feeds every
    time-stepping loop through get_min_radius and the diffusivity reduction."""
    mesh = _starved_box()
    T = uw.discretisation.MeshVariable("T405", mesh, 1, degree=1)

    solver = uw.systems.Diffusion(mesh, u_Field=T)
    solver.constitutive_model = uw.constitutive_models.DiffusionModel
    solver.constitutive_model.Parameters.diffusivity = 1.0

    dt_const = float(solver.estimate_dt())
    gathered = uw.mpi.comm.allgather(dt_const)
    assert max(gathered) - min(gathered) < 1.0e-12, (
        f"estimate_dt disagrees across ranks: {gathered}")
    # Diffusive CFL with unit diffusivity is exactly h_min**2.
    assert np.isclose(dt_const, mesh.get_min_radius() ** 2, rtol=1.0e-12)

    # A spatially varying diffusivity is sampled at THIS rank's cell
    # centroids — a starved rank samples nothing, which is where the
    # unguarded `.max()` used to raise.
    x, _y = mesh.X
    solver.constitutive_model.Parameters.diffusivity = 1.0 + x
    dt_varying = float(solver.estimate_dt())
    gathered = uw.mpi.comm.allgather(dt_varying)
    assert max(gathered) - min(gathered) < 1.0e-12, (
        f"estimate_dt (varying K) disagrees across ranks: {gathered}")
    assert dt_varying < dt_const, (
        "a larger diffusivity must shorten the diffusive timestep")


def test_gather_data_keeps_one_row_per_rank():
    """#405 item 3: NaN rows must survive, so row index still equals rank."""
    contribution = np.array(
        [float("nan") if uw.mpi.rank % 2 else float(uw.mpi.rank)])

    table = uw.utilities.gather_data(contribution, bcast=True)

    assert table.shape[0] == uw.mpi.size, (
        f"gather_data returned {table.shape[0]} rows for {uw.mpi.size} "
        "ranks — a dropped row renumbers every rank after it")
    for r in range(uw.mpi.size):
        if r % 2:
            assert np.isnan(table[r]), f"row {r} should be this rank's NaN"
        else:
            assert table[r] == r, f"row {r} came from the wrong rank"

    # The old behaviour is still available, explicitly.
    stripped = uw.utilities.gather_data(contribution, bcast=True,
                                        strip_nan=True)
    assert stripped.shape[0] == (uw.mpi.size + 1) // 2
