"""The split THROUGH a partition seam (#670, the conform build's cut).

``FaultNetwork.build(seams="conform", realisation="split")`` meshes the
band through every seam the fault crosses and then cuts the spine through
the seam as well: the chain is assembled globally, the vertex on the seam
is duplicated on both ranks with the replica owned where the original is,
and the seam edges re-homed onto the replica keep their star-forest
entries. Nothing is gathered. This test pins the mechanism at np=2 and
np=3 against the gathered (serial-topology) answer:

- the pairing, counted on the owning ranks, is the serial pairing (one
  pair per interior chain vertex, the seam vertex included);
- the star forest is consistent (every leaf's coordinates agree with its
  root, the owned-point Euler characteristic is that of a slit disc);
- the contact solve converges on the inherited geometric tail, the pairs
  do not open, and the peak slip is the gathered answer up to the fill's
  noise — no weld at the seam;
- the cells stay balanced (no rank holds more than 60%).

Run:
    UW_HANG_WATCHDOG=600 UW_HANG_WATCHDOG_ABORT=1 mpirun -np 2 python -m pytest \\
        tests/parallel/ptest_0865_seam_split_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(900)]

ETA_1 = 0.01
LONG_H = 0.02
# The long vertical fault of ptest_0864 (35 cells at h = 0.02), crossed
# once by the np=2 seam and twice at np=3. The gathered split answer at
# np=2 (probe_long, 2026-09-03): peak tangential jump 0.5094, 69 pairs.
LONG_GATHERED_SPLIT = 0.5094
LONG_PAIRS = 69


def _build_long(realisation, seams="conform"):
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


def _global(x, op=MPI.SUM):
    return MPI.COMM_WORLD.allreduce(x, op=op)


def _leaves(dm):
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        return set()
    return set() if ilocal is None else {int(p) for p in ilocal}


def _owned_euler(dm):
    leaves = _leaves(dm)
    counts = []
    for lo, hi in (dm.getDepthStratum(0), dm.getDepthStratum(1),
                   dm.getHeightStratum(0)):
        counts.append(_global(sum(1 for p in range(lo, hi)
                                  if p not in leaves)))
    nv, ne, nc = counts
    return nv - ne + nc


def _sf_coordinate_drift(dm):
    """Broadcast every vertex's coordinates root-to-leaf; leaves must agree."""
    pStart, pEnd = dm.getChart()
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    sf = dm.getPointSF()
    worst = 0.0
    for comp in range(2):
        root = np.zeros(pEnd - pStart, dtype=np.float64)
        root[vS - pStart: vE - pStart] = X[: vE - vS, comp]
        leaf = np.full(pEnd - pStart, np.nan, dtype=np.float64)
        sf.bcastBegin(MPI.DOUBLE, root, leaf, MPI.REPLACE)
        sf.bcastEnd(MPI.DOUBLE, root, leaf, MPI.REPLACE)
        seen = np.isfinite(leaf[vS - pStart: vE - pStart])
        if seen.any():
            worst = max(worst, float(np.abs(
                leaf[vS - pStart: vE - pStart][seen]
                - X[: vE - vS, comp][seen]).max()))
    return _global(worst, op=MPI.MAX)


def _seam_pairs(mesh, name):
    """The pairs whose points are shared (on the seam), and the owned
    pair count — the pair is owned where its Plus point is a root."""
    leaves = _leaves(mesh.dm)
    pairs = mesh._fault_point_pairs[name]
    from underworld3.utilities.place_surface import _shared_point_flags
    shared = _shared_point_flags(mesh.dm).astype(bool)
    pStart = mesh.dm.getChart()[0]
    on_seam = [(qm, qp) for qm, qp in pairs.items()
               if shared[qp - pStart] or shared[qm - pStart]]
    owned = sum(1 for _qm, qp in pairs.items() if qp not in leaves)
    return on_seam, owned


def test_the_split_through_the_seam_is_the_gathered_split():
    comm = MPI.COMM_WORLD
    net = _build_long("split")
    mesh = net.mesh
    dm = mesh.dm
    assert net.info["n_ligament_cells"] == 0
    assert net.ligament_cells() is None

    # the star forest is consistent after the cut through the seam
    assert _sf_coordinate_drift(dm) == 0.0, "a leaf's coordinates drifted"
    assert _owned_euler(dm) == 0, "the slit did not open once, globally"

    # the pairing is the serial pairing, the seam vertex included: each
    # rank the seam crosses holds the pair at the crossing, and exactly
    # one of them owns it
    on_seam, owned = _seam_pairs(mesh, "Main")
    n_seam_pairs = _global(len(on_seam))
    assert n_seam_pairs >= 1, "no pair on the seam: the fixture lost its purpose"
    assert _global(owned) == LONG_PAIRS, (
        f"{_global(owned)} owned pairs vs {LONG_PAIRS} gathered")
    # a seam pair's two points are owned by the same rank (the contact
    # solve's pair block lives in one rank's diagonal portion)
    leaves = _leaves(dm)
    for qm, qp in on_seam:
        assert (qm in leaves) == (qp in leaves), "a seam pair straddles ranks"
    # the whole Plus->Minus normal was recorded for the seam vertices
    normals = getattr(mesh, "_fault_seam_normals", {}).get("Main", {})
    assert _global(len(normals)) >= 1
    for qp, n in normals.items():
        assert np.linalg.norm(n) > 0

    # nothing gathered: the cells stay balanced
    n_local = int(dm.getHeightStratum(0)[1])
    n_total = _global(n_local)
    assert _global(n_local, op=MPI.MAX) < 0.6 * n_total

    stokes = _stokes_on(net)
    net.apply(stokes, eta_1=ETA_1)
    info = net.solve(stokes)
    assert info.get("converged"), "the contact solve did not converge"
    fallbacks = getattr(stokes, "pc_fallbacks", {}) or {}
    assert not fallbacks, f"preconditioner fallback recorded: {fallbacks}"
    from underworld3.utilities.fault_contact import fault_slip
    _s, V, leak = fault_slip(stokes, "Main", info)
    leak_max = _global(float(np.abs(leak).max()) if len(leak) else 0.0,
                       op=MPI.MAX)
    assert leak_max < 1e-9, f"a coincident pair opened: leak {leak_max:.2e}"
    peak = _global(float(net.slips(stokes).get("Main", 0.0)), op=MPI.MAX)
    # the gathered answer up to the fill's noise (the conform TI build sat
    # within 0.5% of gathered; the ligament split lost 8-15% per crossing)
    assert peak == pytest.approx(LONG_GATHERED_SPLIT, rel=0.03), (
        f"split peak slip {peak:.4f} vs gathered {LONG_GATHERED_SPLIT}")


# A three-piece vertical network: Main crosses the np=2 seam (y = 0.5 on
# the unit box) mid-fault, Cont continues above it, the Splay branches
# below it. The pieces of ptest_0864 END exactly on that seam, and a tip
# on the seam beside a junction is a fill the conform placement refuses
# today (its boundary walk doubles back) — a limitation of the placement
# recorded in the design note, so this fixture keeps the crossing mid-fault.
H = 0.03
WIDTH = 0.04
# The gathered (serial-topology) peaks, measured 2026-09-05 (net_ref probe):
# 29 / 15 / 13 pairs.
GATHERED = {"Main": 0.3438, "Cont": 0.2130, "Splay": 0.0433}


def _pieces():
    main = np.column_stack([np.full(16, 0.5), np.linspace(0.25, 0.62, 16)])
    cont = np.column_stack([np.full(9, 0.5), np.linspace(0.67, 0.86, 9)])
    s = np.linspace(0.0, 1.0, 8)
    splay = np.column_stack([0.5 + 0.18 * s, 0.33 - 0.12 * s])
    return [("Main", main), ("Cont", cont), ("Splay", splay)]


def test_the_network_split_through_the_seams_matches_the_gathered_peaks():
    comm = MPI.COMM_WORLD
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
        regular=False, refinement=1, qdegree=2)
    pieces = _pieces()
    net = uw.meshing.FaultNetwork(pieces, hierarchy=[n for n, _p in pieces])
    net.prepare(h=H, ligament=1.0, verbose=False)
    net.build(base=base, width=WIDTH, realisation="split", max_levels=1,
              seams="conform")
    mesh = net.mesh
    assert _sf_coordinate_drift(mesh.dm) == 0.0
    # three slits: the owned Euler characteristic of a disc with three
    # disjoint cuts (a fault wholly in no rank is still cut somewhere)
    n_cut = sum(1 for name in ("Main", "Cont", "Splay")
                if _global(len(mesh._fault_point_pairs[name])) > 0)
    assert _owned_euler(mesh.dm) == 1 - n_cut
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
    net.apply(stokes, eta_1=ETA_1)
    info = net.solve(stokes)
    assert info.get("converged")
    slips = net.slips(stokes)
    for name, ref in GATHERED.items():
        peak = _global(float(slips.get(name, 0.0)), op=MPI.MAX)
        assert peak == pytest.approx(ref, rel=0.05), (
            f"{name}: peak slip {peak:.4f} vs gathered {ref}")
