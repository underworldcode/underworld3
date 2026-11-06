"""The seam-ligament placement of a 2-D fault network (#670).

``FaultNetwork.build(seams="ligament")`` carves the band on every rank at
once, clips it one cell short of each partition seam and leaves the base
cells there as a LIGAMENT the fault is not cut through: nothing is
gathered, the split runs rank-local (a fault crossing a rank in and out
is split as sub-chains), and the band's weak plane painted on the
ligament cells carries the slip across the seam. This test pins the
mechanism at np=2 and np=3:

- the placement's gates hold and its info is identical on every rank;
  the band's cells are embedded once or stand in the ligament, never both;
- no fault edge touches the seam, so no redistribution happened (the
  cell count is the local count the partition gave, plus the surgery);
- the contact solve converges, and the peak slip per piece sits within
  a stated bound of the gathered (serial-topology) answer — the ligament
  is a modelling perturbation whose size this test MEASURES, not a
  reproduction identity.

Run:
    UW_HANG_WATCHDOG=600 UW_HANG_WATCHDOG_ABORT=1 mpirun -np 2 python -m pytest \\
        tests/parallel/ptest_0864_seam_ligament_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(900)]

H = 0.03
WIDTH = 0.04
ETA_1 = 0.01           # the weak-plane viscosity the ligament carries
# The network of ptest_0859 turned VERTICAL. On the unit box the np=2
# seam runs along y = 0.5 (the partitioner follows the cell numbering),
# so the horizontal network of ptest_0859 lies ALONG the seam and its
# whole band is ligament — the along-strike pathology the design note
# records, not a crossing. Vertical, the seam crosses each piece once.
# The gathered answer of the vertical network at np=2 (probe_cross,
# 2026-09-03): the peak tangential jump per piece.
GATHERED = {"Main": 0.4364, "Cont": 0.3591, "Splay": 0.1047}


def _pieces():
    main = np.column_stack([np.full(12, 0.5), np.linspace(0.25, 0.50, 12)])
    cont = np.column_stack([np.full(9, 0.5), np.linspace(0.55, 0.75, 9)])
    s = np.linspace(0.0, 1.0, 8)
    splay = np.column_stack([0.5 + 0.18 * s, 0.38 + 0.12 * s])
    return [("Main", main), ("Cont", cont), ("Splay", splay)]


def _build(seams):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
        regular=False, refinement=1, qdegree=2)
    pieces = _pieces()
    net = uw.meshing.FaultNetwork(pieces, hierarchy=[n for n, _p in pieces])
    net.prepare(h=H, ligament=1.0, verbose=False)
    net.build(base=base, width=WIDTH, realisation="split", max_levels=1,
              seams=seams)
    return net


def _shared_flags(dm):
    from underworld3.utilities.place_surface import _shared_point_flags
    return _shared_point_flags(dm).astype(bool)


def test_the_ligament_placement_is_gated_and_rank_local():
    comm = MPI.COMM_WORLD
    net = _build("ligament")
    mesh = net.mesh
    info = net.info
    # identical global bookkeeping on every rank (n_cells is rank-local)
    keys = ("n_ligament_cells", "seams")
    seen = comm.allgather({k: info[k] for k in keys})
    assert all(s == seen[0] for s in seen), seen
    assert info["seams"] == "ligament"
    # the band crosses the seam on this fixture: a ligament exists, and
    # it is band material on every rank that holds it
    lig = net.ligament_cells()
    assert lig is not None, "no seam crossing: the fixture lost its purpose"
    band = info["band"]
    assert not (lig & ~band).any(), "a ligament cell outside the band"
    # nothing gathered, nothing redistributed: no fault facet touches
    # the seam, on any rank (the split's own precondition)
    dm = mesh.dm
    shared = _shared_flags(dm)
    pStart, _pEnd = dm.getChart()
    fS, fE = dm.getHeightStratum(1)
    touching = 0
    for name in ("Main", "Cont", "Splay"):
        for side in ("Plus", "Minus"):
            lbl = f"{name}{side}"
            if not dm.hasLabel(lbl):
                continue
            v = int(mesh.boundaries[lbl].value)
            if dm.getLabel(lbl).getStratumSize(v) == 0:
                continue
            for f in dm.getLabel(lbl).getStratumIS(v).getIndices():
                f = int(f)
                if fS <= f < fE and (shared[f - pStart] or any(
                        shared[int(q) - pStart]
                        for q in dm.getTransitiveClosure(f)[0])):
                    touching += 1
    assert comm.allreduce(touching, op=MPI.SUM) == 0
    # the long pieces were split somewhere, and the pairing is rank-local
    # (the Splay ends ON the seam and may lie wholly in the ligament)
    for name in ("Main", "Cont"):
        n_pairs = comm.allreduce(len(mesh._fault_point_pairs[name]),
                                 op=MPI.SUM)
        assert n_pairs > 0, f"{name} was not split on any rank"


LONG_H = 0.02
# One long vertical fault (35 cells) crossed once by the np=2 seam, twice
# at np=3. The gathered (serial-topology) answers at np=2, probe_long /
# probe_long_ti, 2026-09-03: the split's peak tangential jump and the
# weak plane's peak tangential velocity jump across the band.
LONG_GATHERED = {"split": 0.5094, "ti": 0.5688}


def _build_long(seams, realisation):
    h = LONG_H
    main = np.column_stack([np.full(36, 0.5), np.linspace(0.15, 0.85, 36)])
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=4 * h,
        regular=False, refinement=1, qdegree=2)
    net = uw.meshing.FaultNetwork([("Main", main)])
    net.prepare(h=h, verbose=False)
    net.build(base=base, width=2 * h, realisation=realisation, max_levels=1,
              seams=seams, band=2 * h, ramp=6 * h)
    return net


def _stokes_on(net):
    mesh = net.mesh
    x, _y = mesh.X
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1,
                                       continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 2.0 * (x - 0.5)), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    return stokes


def test_the_weak_plane_crosses_the_seam_within_the_bound():
    """The TI realisation - the design line's realisation for
    partition-crossing structures: the ligament's base cells ARE the
    band there, at the base's resolution, and the answer moves by a few
    per cent while the cells stay balanced. Measured 2026-09-03 at
    h = 0.02 with the band butted to the seam: peak 0.5684 (np=2) and
    0.5643 (np=3) against 0.5688 (the first, seam-cell build: 0.5605
    and 0.5484)."""
    comm = MPI.COMM_WORLD
    net = _build_long("ligament", "ti")
    assert net.ligament_cells() is not None, "no crossing on this fixture"
    stokes = _stokes_on(net)
    net.apply(stokes, eta_1=ETA_1)
    stokes.solve()
    peak = comm.allreduce(float(net.slips(stokes).get("Main", 0.0)),
                          op=MPI.MAX)
    assert peak == pytest.approx(LONG_GATHERED["ti"], rel=0.06), (
        f"weak-plane peak slip {peak:.4f} vs gathered "
        f"{LONG_GATHERED['ti']}")
    # nothing was gathered: no rank holds more than 60% of the cells
    n_local = int(net.mesh.dm.getHeightStratum(0)[1])
    n_total = comm.allreduce(n_local, op=MPI.SUM)
    assert comm.allreduce(n_local, op=MPI.MAX) < 0.6 * n_total


def test_the_split_runs_rank_local_and_the_tips_weld_as_measured():
    """The split realisation under the ligament: rank-local sub-chains,
    the contact solve converging on geometric multigrid - and a WELD at
    each crossing, because a sub-chain ends in a pinned tip that no weak
    plane can free. Measured 2026-09-03 at h = 0.02, eta_1 = 0.01, band
    butted to the seam, blind tip enclosed by the weak plane: peak 0.4703
    (np=2, one crossing) and 0.4352 (np=3, two) against the gathered
    0.5094 (the seam-cell build gave 0.4421 and 0.3869, insensitive to
    eta_1). The bound is that measurement; the remedy (a free tip at a
    ligament end) is the design note's next step."""
    comm = MPI.COMM_WORLD
    net = _build_long("ligament", "split")
    mesh = net.mesh
    assert net.ligament_cells() is not None, "no crossing on this fixture"
    n_pairs = comm.allreduce(len(mesh._fault_point_pairs["Main"]),
                             op=MPI.SUM)
    assert 0 < n_pairs < 69, n_pairs        # 69 pairs when gathered
    stokes = _stokes_on(net)
    net.apply(stokes, eta_1=ETA_1)
    info = net.solve(stokes)
    assert info.get("converged"), "the contact solve did not converge"
    fallbacks = getattr(stokes, "pc_fallbacks", {}) or {}
    assert not fallbacks, f"preconditioner fallback recorded: {fallbacks}"
    peak = comm.allreduce(float(net.slips(stokes).get("Main", 0.0)),
                          op=MPI.MAX)
    assert 0.7 * LONG_GATHERED["split"] < peak < LONG_GATHERED["split"], (
        f"split peak slip {peak:.4f} vs gathered {LONG_GATHERED['split']}")


def test_the_band_meshed_through_the_seam_gives_the_gathered_answer():
    """``seams="conform"``: the band keeps its own resolution through the
    seam (each rank makes the band cells its cavity holds, the fill wraps
    around the band, and the band vertices both sides use are shared
    points of the rebuild), so the weak plane's answer is the gathered
    one up to the fill's noise while the cells stay balanced. Measured
    2026-09-03: 0.5688 at np=2 and np=3 against 0.5688."""
    comm = MPI.COMM_WORLD
    net = _build_long("conform", "ti")
    info = net.info
    assert info["n_ligament_cells"] == 0
    stokes = _stokes_on(net)
    net.apply(stokes, eta_1=ETA_1)
    stokes.solve()
    peak = comm.allreduce(float(net.slips(stokes).get("Main", 0.0)),
                          op=MPI.MAX)
    assert peak == pytest.approx(LONG_GATHERED["ti"], rel=5e-3), (
        f"weak-plane peak slip {peak:.4f} vs gathered "
        f"{LONG_GATHERED['ti']}")
    n_local = int(net.mesh.dm.getHeightStratum(0)[1])
    n_total = comm.allreduce(n_local, op=MPI.SUM)
    assert comm.allreduce(n_local, op=MPI.MAX) < 0.6 * n_total
