"""The pair jump's sense agrees with the rock either side, at a junction.

A split fault's slip is read from its node pairs as :math:`v^+ - v^-`,
and the sign of that is only meaningful if every vertex took its Plus
and Minus sides the same way round. Away from anything else a vertex's
fan is a closed ring and the walk cannot go wrong. Where a cut stops
one rung short of another — a kissing Y — the tip vertices' fans touch
the other cut, and that is where the side-taking was found reversed
(S-fault rig, superfine, 2026-09-04 dump: the splay's first samples
read +0.0070, +0.0084, +0.0098 where the current build and the velocity
field both say -0.0070, -0.0084, -0.0098; the rest of the strand and
every peak were identical).

The oracle is the velocity itself, off the cut in the intact rock at
±w/2, projected on the trace: ``(v_left - v_right) · t`` with ``left``
the polyline's own left, which never goes through the pair machinery.
Wherever both measures are clearly non-zero they must agree in sign,
on every strand, including the one that kisses.

The control is the assertion's sensitivity: swap the sides on the pairs
nearest the junction and the check fails there — a reversed tip is
seen, not averaged away.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite")]

H = 0.02
WIDTH = 0.04          # two rungs across; the Y kisses at one rung
FLOOR = 0.08          # a sample counts once it is this fraction of the
                      # strand's peak, in BOTH measures (tips read ~0)


def _pieces():
    """A straight main strand on the diagonal and a splay leaving its
    upper limb at ~22 degrees, ending on it (prepare() trims the splay
    back to a one-rung kissing junction)."""
    s = np.linspace(-0.28, 0.28, 29)
    d = np.array([np.cos(np.deg2rad(40.0)), np.sin(np.deg2rad(40.0))])
    main = 0.5 + s[:, None] * d
    phi = np.deg2rad(62.0)
    e = np.array([np.cos(phi), np.sin(phi)])
    start = 0.5 + 0.03 * d
    u = np.linspace(0.0, 0.16, 9)
    splay = start + u[:, None] * e
    return [("Main", main), ("Splay", splay)]


def _solve():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
        regular=False, refinement=1, qdegree=2)
    pieces = _pieces()
    net = uw.meshing.FaultNetwork(pieces,
                                  hierarchy=[n for n, _p in pieces])
    net.prepare(h=H, ligament=1.0, verbose=False)
    mesh = net.build(base=base, width=WIDTH, realisation="split",
                     max_levels=1)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1,
                                       continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, 0.0]
    # simple shear parallel to the main strand: dextral on it
    d = np.array([np.cos(np.deg2rad(40.0)), np.sin(np.deg2rad(40.0))])
    n = np.array([-d[1], d[0]])
    t_sym = (x - 0.5) * float(n[0]) + (y - 0.5) * float(n[1])
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((2.0 * t_sym * float(d[0]),
                                 2.0 * t_sym * float(d[1])), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    for name, _p in net.prepared:
        stokes.add_fault_bc(0, boundary=name)
    from underworld3.utilities import fault_contact
    info = fault_contact.solve_with_fault(stokes)
    assert info.get("converged")
    return net, stokes, v


def _both_measures(net, stokes, v, name):
    """(gauge, rock) at every pair of ``name``, ordered along the
    trace: the pair jump on the trace tangent, and the rock velocity
    difference across ±w/2 on the same tangent."""
    from scipy.spatial import cKDTree
    from underworld3.utilities.fault_contact import fault_pair_jumps

    P = np.asarray(dict(net.prepared)[name], dtype=float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    left = np.column_stack([-t[:, 1], t[:, 0]])
    coords, jumps, _normals = fault_pair_jumps(
        stokes, name, stokes._rotated_freeslip_info, gather=True)
    near = cKDTree(P).query(coords)[1]
    gauge = np.einsum("ij,ij->i", jumps, t[near])
    Qp = coords + 0.5 * WIDTH * left[near]
    Qm = coords - 0.5 * WIDTH * left[near]
    vp = np.asarray(uw.function.evaluate(v.sym, Qp)).reshape(len(coords), -1)
    vm = np.asarray(uw.function.evaluate(v.sym, Qm)).reshape(len(coords), -1)
    rock = np.einsum("ij,ij->i", vp[:, :2] - vm[:, :2], t[near])
    order = np.argsort((coords - P[0]) @ (P[-1] - P[0]))
    return gauge[order], rock[order]


def _disagreements(gauge, rock):
    clear = ((np.abs(gauge) > FLOOR * np.abs(gauge).max())
             & (np.abs(rock) > FLOOR * np.abs(rock).max()))
    assert clear.sum() >= 4, "too few samples above the floor to judge"
    return np.flatnonzero(clear & (np.sign(gauge) != np.sign(rock)))


@pytest.fixture(scope="module")
def solved():
    return _solve()


def test_every_strand_slips_the_way_the_rock_says(solved):
    net, stokes, v = solved
    for name, _p in net.prepared:
        gauge, rock = _both_measures(net, stokes, v, name)
        bad = _disagreements(gauge, rock)
        assert len(bad) == 0, (
            f"{name}: the pair jump and the rock disagree in sign at "
            f"samples {bad.tolist()} — gauge {gauge[bad].round(4).tolist()}, "
            f"rock {rock[bad].round(4).tolist()}")


def test_the_main_is_dextral_under_a_dextral_drive(solved):
    net, stokes, v = solved
    gauge, rock = _both_measures(net, stokes, v, "Main")
    assert gauge[len(gauge) // 2] > 0 and rock[len(rock) // 2] > 0


def test_a_reversed_tip_is_seen(solved):
    """The control: swap the sides on the three pairs nearest the Y and
    the check must fail THERE — the floor does not hide a tip."""
    net, stokes, v = solved
    gauge, rock = _both_measures(net, stokes, v, "Splay")
    assert len(_disagreements(gauge, rock)) == 0
    k = np.flatnonzero(np.abs(gauge) > FLOOR * np.abs(gauge).max())[:3]
    wrong = gauge.copy()
    wrong[k] *= -1.0
    bad = _disagreements(wrong, rock)
    assert set(k.tolist()) <= set(bad.tolist())
