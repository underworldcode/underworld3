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
1e-8, so that convergence noise sits well inside the bounds).

The iteration check. The mover holds the zone field where it is in space while the
cells move under it, so the kept patch's cells hold less of the band at each move, and
against a patch on the recomputed zone the kept one costs what that drift costs on the
host's mesh. The kept solver must carry nothing else from before the move: on the zone,
a registration made after the move with the kept solver's own mask takes the same
velocity count, and both take fewer than a registration with no zone, which installs no
patch. On a split level a move also drops the structural rows outside the zone, and the
kept solver takes no more than one velocity iteration more than a fresh registration.
The counts measured are beside the assertions.

The fixture is test_1139's: a transverse-isotropic weak band beside a placed fault,
the band's cells as the zone.

    python -m pytest tests/parallel/test_1152_fac_zone_survives_a_mover_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1152_fac_zone_survives_a_mover_parallel.py
"""
import warnings

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems.solve_report import ksp_reason_string
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


def _told(s, split):
    """How a solve ended and where it started from, for a failed assertion's message."""
    if split:
        ksp_reason = s._rotated_freeslip_info["ksp_reason"]
    else:
        ksp_reason = s.snes.getKSP().getConvergedReason()
    report = s.solve_report
    return {"reason": report.reason_str, "ksp_reason": ksp_reason_string(ksp_reason),
            "first_residual": report.fnorm0, "last_residual": report.fnorm}


_PC_FAILED = {int(getattr(PETSc.PC.FailedReason, name)): name
              for name in dir(PETSc.PC.FailedReason) if name.isupper()}


def _pc_failed(pc):
    return _PC_FAILED.get(int(pc.getFailedReason()), str(pc.getFailedReason()))


def _ksp_lines(ksp, name, lines):
    """One line for ``ksp`` and for each solver under it: how it ended, its iteration
    count, why its preconditioner failed, and the norm of its operator (NaN where the
    assembly left one) with its count of entries that are not finite. They tell a
    failed factorisation (``DIVERGED_PC_FAILED`` with a zero pivot on one block and
    ``SUBPC_ERROR`` above it) from a NaN in the operator (a NaN norm,
    ``DIVERGED_NANORINF``) from a sub-solve that diverged by itself (its own reason,
    no failed preconditioner).

    Nothing here is collective: a rank whose preconditioner never set up takes fewer
    lines than the others, and a collective norm would then hang. A line carries this
    rank's rows of the operator, and :func:`_record` adds the ranks up."""
    pc, A = ksp.getPC(), ksp.getOperators()[0]
    rows = None
    if "aij" in A.getType():
        values = A.getValuesCSR()[2]
        rows = (float(np.sum(values ** 2)), int(np.sum(~np.isfinite(values))))
    lines.append((name, f"{ksp.getType()}/{pc.getType()} "
                        f"reason={ksp_reason_string(ksp.getConvergedReason())} "
                        f"its={ksp.getIterationNumber()} pc_failed={_pc_failed(pc)}",
                  rows))
    _pc_lines(pc, name, lines)


def _pc_lines(pc, name, lines):
    kind = pc.getType()
    try:
        if kind == "fieldsplit":
            for i, sub in enumerate(pc.getFieldSplitSubKSP()):
                _ksp_lines(sub, f"{name}.split{i}", lines)
        elif kind == "mg":
            for level in range(pc.getMGLevels()):
                _ksp_lines(pc.getMGSmoother(level), f"{name}.level{level}", lines)
        elif kind == "composite":
            # the FAC smoother: the level's SOR, then the ASM over the patch blocks
            for i in (0, 1):
                member = pc.getCompositePC(i)
                lines.append((f"{name}.pc{i}",
                              f"{member.getType()} pc_failed={_pc_failed(member)}", None))
                _pc_lines(member, f"{name}.pc{i}", lines)
        elif kind == "asm":
            # this rank's blocks; the ranks' lines are set side by side when printed
            blocks = pc.getASMSubKSP()
            lines.append((f"{name}.blocks", f"{len(blocks)} "
                          f"reasons={sorted({ksp_reason_string(b.getConvergedReason()) for b in blocks})} "
                          f"pc_failed={sorted({_pc_failed(b.getPC()) for b in blocks})}",
                          None))
    except PETSc.Error as err:
        # a preconditioner that never set up has no members to ask
        lines.append((f"{name}.{kind}", f"not set up (PETSc error {err.ierr})", None))


def _record(tag, k, errors, told, solvers, split):
    """The whole record of a failed move, one key a line, for the test to print: an
    assertion message is cut at a few hundred characters, and the linear solvers'
    reasons were the part cut from the first red run that carried one."""
    def flat(value):
        if isinstance(value, dict):
            return ", ".join(f"{key}={item}" for key, item in value.items())
        return str(value)

    lines = [("errors", flat({key: float(value) for key, value in errors.items()}), None)]
    lines += [(key, flat(value), None) for key, value in told.items()]
    for which, s in solvers.items():
        if split:
            cache = getattr(s, "_rotated_linear_cache", None)
            ksp = cache["ctx"]["ksp"] if cache else None
        else:
            ksp = s.snes.getKSP()
        if ksp is None:
            lines.append((f"{which}.ksp", "no linear solver kept", None))
        else:
            _ksp_lines(ksp, f"{which}.ksp", lines)
    # the ranks' lines are matched by key: a rank may hold fewer
    gathered = [{key: (text, rows) for key, text, rows in g}
                for g in uw.mpi.comm.allgather(lines)]
    keys = list(dict.fromkeys(key for g in gathered for key in g))
    record = [f"test_1152[{tag}] move {k}: the record of a failed move"]
    for key in keys:
        held = [g[key] for g in gathered if key in g]
        text = held[0][0]
        rows = [r for _t, r in held if r is not None]
        if rows:
            text += (f" operator_norm={np.sqrt(sum(r[0] for r in rows))}"
                     f" not_finite={sum(r[1] for r in rows)}")
        others = sorted({t for t, _r in held} - {held[0][0]})
        absent = len(gathered) - len(held)
        record.append(f"  {key}: {text}"
                      + (f" | on other ranks: {others}" if others else "")
                      + (f" | absent on {absent} rank(s)" if absent else ""))
    return record


def _norm(a):
    return np.sqrt(uw.mpi.comm.allreduce(float(np.sum(np.asarray(a) ** 2))))


def _gauged(a):
    a = np.asarray(a).ravel()
    return a - uw.mpi.comm.allreduce(float(a.sum())) / uw.mpi.comm.allreduce(a.size)


def _moves(split):
    """Solve after each of MOVES MMPDE moves with the patch kept and with a fresh
    registration; yield (kept, fresh, others, errors, warnings, told) per move. On the
    zone, ``others`` holds two more registrations made after the move: ``same`` with
    the kept solver's mask and ``none`` with no zone. It is empty on a split level.
    ``told`` is what a red run needs to explain itself: how each solve ended, the
    residual it started from, and the worst cell the move left."""
    mesh, levels, zone = _fixture(split)
    kept, vk, pk = _stokes(mesh, zone, split, "k")
    kept_mask = np.asarray(mesh.cells_labelled("Band", 72), dtype=bool)
    set_custom_fmg(kept, levels, field_id=0, fac_zone=kept_mask)
    fresh, vf, pf = _stokes(mesh, zone, split, "f")
    others = {} if split else {"same": (_stokes(mesh, zone, split, "s")[0], kept_mask),
                               "none": (_stokes(mesh, zone, split, "n")[0], None)}
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
        told = {"q_min": mesh.quality()["q_min"], "kept": _told(kept, split)}
        set_custom_fmg(fresh, levels, field_id=0,
                       fac_zone=np.asarray(zone.array[:, 0, 0] > 0.5))
        fresh.solve(zero_init_guess=True)
        told["fresh"] = _told(fresh, split)
        for which, (solver, mask) in others.items():
            set_custom_fmg(solver, levels, field_id=0, fac_zone=mask)
            solver.solve(zero_init_guess=True)
            told[which] = _told(solver, split)
        errors = {
            "velocity": _norm(np.asarray(vk.array) - np.asarray(vf.array)) / _norm(vf.array),
            "pressure": (_norm(_gauged(pk.array) - _gauged(pf.array))
                         / _norm(_gauged(pf.array))),
        }
        if split:
            _s, slip_k, _l = fault_slip(kept, "Fault", kept._rotated_freeslip_info)
            _s, slip_f, _l = fault_slip(fresh, "Fault", fresh._rotated_freeslip_info)
            errors["slip"] = _norm(slip_k - slip_f) / _norm(slip_f)
        yield (k, kept, fresh, {which: solver for which, (solver, _m) in others.items()},
               errors, said, told)


@pytest.mark.parametrize("split", [False, True], ids=["zone", "zone_on_a_split_level"])
def test_a_kept_zone_patch_matches_a_fresh_registration(split, capsys):
    for k, kept, fresh, others, errors, said, told in _moves(split):
        its_k = _velocity_iterations(kept, split)
        its_f = _velocity_iterations(fresh, split)
        its = {which: _velocity_iterations(solver, split)
               for which, solver in others.items()}
        told["velocity_its"] = {"kept": its_k, "fresh": its_f, **its}
        # one line per move, past pytest's capture: on a loaded host a move takes over a
        # minute at np 2 (hyperion at load 60: 157 s and 254 s for the two cases, 3-4
        # times CI), and the MPI supervisor kills a job silent for 300 s. It carries the
        # counts, so a green run records them for its host too.
        with capsys.disabled():
            uw.pprint(f"test_1152[{'split' if split else 'zone'}] move {k}/{MOVES}: "
                      f"velocity iterations kept {its_k}, zone recomputed {its_f}"
                      + ("" if split else f", kept mask registered again {its['same']}, "
                                          f"no zone {its['none']}"),
                      flush=True)
        counts_hold = (its_k <= its_f + 1 if split
                       else its_k == its["same"] and its_k < its["none"])
        if not (counts_hold and errors["velocity"] < 100 * TOLERANCE
                and all(errors[key] < 1000 * TOLERANCE for key in errors)):
            record = _record("split" if split else "zone", k, errors, told,
                             {"kept": kept, "fresh": fresh, **others}, split)
            # twice: past the capture for a run under mpirun, where the report of a
            # failure may never be written, and into it for a run under pytest-xdist,
            # whose workers' own output does not reach the log
            with capsys.disabled():
                for line in record:
                    uw.pprint(line, flush=True)
            for line in record:
                uw.pprint(line, flush=True)
        if split:
            # np 1 on macOS (12-core Apple M4), kept and fresh at each move: 17 and 16,
            # 17 and 17, 17 and 17. With no zone registered: 18, 18, 17, so the bound
            # tells a lost patch at the first move only.
            assert its_k <= its_f + 1, (k, told["velocity_its"])
        else:
            # Velocity iterations at moves 1, 2 and 3, np 1 on macOS (12-core Apple M4):
            # kept 7, 7, 7; the kept mask registered again 7, 7, 7; the zone recomputed
            # 6, 6, 6; no zone 12, 10, 9. The kept patch's cells hold 135 of the 330
            # weak cells after the first move, 54 of 260 after the second and 31 of
            # 208 after the third, which is the one iteration against the recomputed
            # zone; on a GitHub-hosted Ubuntu runner that gap was two at the second
            # move (8 against 6), so it is printed and not bounded.
            assert its_k == its["same"], (k, told["velocity_its"])
            assert its_k < its["none"], (k, told["velocity_its"])
            # no zone and a moved mesh: nothing is installed on the finest level
            none_rows = others["none"]._custom_mg["hierarchy"].level_patch_rows[-1]
            assert none_rows is None, (k, none_rows)
        # to the solver tolerance: at 1e-8 kept and fresh agree to ~4e-9 (hyperion,
        # np 1, 2 and 4); a patch only preconditions, and the outer solve converges
        # on the true residual, so a stale one costs iterations and not accuracy
        assert errors["velocity"] < 100 * TOLERANCE, (k, errors, told)
        assert errors["pressure"] < 1000 * TOLERANCE, (k, errors, told)
        if split:
            assert errors["slip"] < 1000 * TOLERANCE, (k, errors, told)
        record = kept.pc_fallbacks.get("custom_mg.fac_patch")
        if not split:
            # nothing was dropped: no warning, no record
            assert record is None and not said, (k, record, said)
        else:
            # the structural rows outside the zone are gone, said once
            assert record is not None and "fac_zone blocks" in record["installed"]
            assert len(said) == (1 if k == 1 else 0), (k, said)
