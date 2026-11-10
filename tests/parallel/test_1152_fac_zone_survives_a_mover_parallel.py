"""A ``fac_zone`` FAC patch survives a mover, and the solution does not change (#1017).

A move of the mesh kills its FAC patch (Louis's ruling, 2026-11-08): the coincidence
test that finds a split fault's structural rows reads moved fine nodes against coarse
levels that did not move. A zone-keyed patch is cut from the masked cells' rows instead,
which a node move leaves in place, and it only preconditions; it is kept (Louis,
2026-11-10: "this is mostly about efficiency and I am good with what works but I don't
want to compromise accuracy"). On a split level the move still drops the structural rows
outside the zone. A repartition or a topology change kills the whole patch.

The accuracy check: after each of several MMPDE moves, the solve with the kept patch
matches a solve with the hierarchy registered again and the zone recomputed from the
moved field, in velocity, pressure and fault slip, to the solver tolerance (solved at
1e-8, so that convergence noise sits well inside the bounds), and takes no more than
one velocity iteration more. Hyperion measured 2-3 velocity iterations at
every move with the patch kept, the same as a fresh registration, and 14-28 with it
killed, at np 1, 2 and 4. A kept patch that is stale (its rows away from the band)
fails the iteration bound.

The fixture is test_1139's: a transverse-isotropic weak band beside a placed fault,
the band's cells as the zone.

    python -m pytest tests/parallel/test_1152_fac_zone_survives_a_mover_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1152_fac_zone_survives_a_mover_parallel.py
"""
import warnings

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.fault_contact import fault_slip
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(900)]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80
LENGTH = float(np.linalg.norm(B - A))
S = LENGTH / N_RUNGS
T_HAT = (B - A) / LENGTH
N_HAT = np.array([T_HAT[1], -T_HAT[0]])
MOVES = 3
TOLERANCE = 1.0e-8


def _fixture(split):
    main = A + (np.arange(N_RUNGS + 1) * S)[:, None] * T_HAT
    band = A + 0.08 * N_HAT + np.linspace(0.0, 1.0, N_RUNGS + 1)[:, None] * (B - A)
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(
        base, [("Fault", main)], 2 * S, spines=[("Fault", main), ("TI", band)],
        margin_rings=1, clearance=0.7, split=split, mesher="ladder")[0]
    zone = uw.discretisation.MeshVariable("zone1152", mesh, 1, degree=0, continuous=False)
    c = np.asarray(zone.coords)
    P, Q = band[0], band[-1]
    t = np.clip(((c - P) @ (Q - P)) / ((Q - P) @ (Q - P)), 0.0, 1.0)
    zone.array[:, 0, 0] = (np.linalg.norm(c - (P + t[:, None] * (Q - P)), axis=1)
                           < S).astype(float)
    return mesh, levels, zone


def _stokes(mesh, zone, split, tag):
    _x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"V1152{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1152{tag}", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.constitutive_model.Parameters.shear_viscosity_1 = 1.0 - 0.99 * zone.sym[0]
    s.constitutive_model.Parameters.director = sympy.Matrix([[N_HAT[0], N_HAT[1]]])
    s.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    if split:
        s.add_fault_bc(0, boundary="Fault")
    s.petsc_use_pressure_nullspace = True
    # well inside the comparison's bounds: at 1e-6, kept and fresh differ by the
    # convergence noise itself, 1.5e-6 in velocity at np 2 (hyperion, round 3)
    s.tolerance = TOLERANCE
    return s, v, p


def _velocity_iterations(s, split):
    if split:
        return max(s._rotated_freeslip_info["vel_its_last"])
    return s.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getIterationNumber()


def _norm(a):
    return np.sqrt(uw.mpi.comm.allreduce(float(np.sum(np.asarray(a) ** 2))))


def _gauged(a):
    a = np.asarray(a).ravel()
    return a - uw.mpi.comm.allreduce(float(a.sum())) / uw.mpi.comm.allreduce(a.size)


def _moves(split):
    """Solve after each of MOVES MMPDE moves with the patch kept and with a fresh
    registration; yield (kept, fresh, errors, warnings) per move."""
    mesh, levels, zone = _fixture(split)
    kept, vk, pk = _stokes(mesh, zone, split, "k")
    set_custom_fmg(kept, levels, field_id=0,
                   fac_zone=np.asarray(mesh.cells_labelled("Band", 72), dtype=bool))
    fresh, vf, pf = _stokes(mesh, zone, split, "f")
    kept.solve(zero_init_guess=True)
    x, y = mesh.X
    for k in range(1, MOVES + 1):
        cx = 0.35 + 0.1 * k
        uw.meshing.node_redistribution(
            mesh, 1 + 6 * sympy.exp(-(((x - cx) ** 2 + (y - 0.5) ** 2) / 0.03)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kept.solve(zero_init_guess=True)
        said = [str(w.message) for w in caught if "FAC patch" in str(w.message)]
        set_custom_fmg(fresh, levels, field_id=0,
                       fac_zone=np.asarray(zone.array[:, 0, 0] > 0.5))
        fresh.solve(zero_init_guess=True)
        errors = {
            "velocity": _norm(np.asarray(vk.array) - np.asarray(vf.array)) / _norm(vf.array),
            "pressure": (_norm(_gauged(pk.array) - _gauged(pf.array))
                         / _norm(_gauged(pf.array))),
        }
        if split:
            _s, slip_k, _l = fault_slip(kept, "Fault", kept._rotated_freeslip_info)
            _s, slip_f, _l = fault_slip(fresh, "Fault", fresh._rotated_freeslip_info)
            errors["slip"] = _norm(slip_k - slip_f) / _norm(slip_f)
        yield k, kept, fresh, errors, said


@pytest.mark.parametrize("split", [False, True], ids=["zone", "zone_on_a_split_level"])
def test_a_kept_zone_patch_matches_a_fresh_registration(split):
    for k, kept, fresh, errors, said in _moves(split):
        # measured at np 1: 6 and 6 on the zone, 13 and 13 on the split level; a
        # killed or stale zone patch took 12 and 15
        its_k = _velocity_iterations(kept, split)
        its_f = _velocity_iterations(fresh, split)
        assert its_k <= its_f + 1, (k, its_k, its_f)
        # to the solver tolerance: at 1e-8 kept and fresh agree to ~4e-9 (hyperion,
        # np 1, 2 and 4); a patch only preconditions, and the outer solve converges
        # on the true residual, so a stale one costs iterations and not accuracy
        assert errors["velocity"] < 100 * TOLERANCE, (k, errors)
        assert errors["pressure"] < 1000 * TOLERANCE, (k, errors)
        if split:
            assert errors["slip"] < 1000 * TOLERANCE, (k, errors)
        record = kept.pc_fallbacks.get("custom_mg.fac_patch")
        if not split:
            # nothing was dropped: no warning, no record
            assert record is None and not said, (k, record, said)
        else:
            # the structural rows outside the zone are gone, said once
            assert record is not None and "fac_zone blocks" in record["installed"]
            assert len(said) == (1 if k == 1 else 0), (k, said)
