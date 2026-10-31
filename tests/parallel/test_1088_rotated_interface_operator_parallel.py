"""The rotated Newton loop smooths every iterate with that iterate's operator (#894).

With a fault interface law, ``solve_rotated_freeslip`` added the interface
tangent into a NEW matrix at every Newton iterate (``Atot = Ahat.copy()``, then
``axpy``) and handed it to the cached KSP. PCMG keeps the finest smoother's
operator once it is set unless the new operator is the same Mat object (the
``opsset`` check in PETSc's ``PCSetUp_MG``), so the finest smoother, and the
Galerkin coarse operators built from it, stayed at iterate 0. The answer was
unaffected; the cost was iterations. Measured on this fixture with a viscous
fault, velocity iterations per Newton increment:

| path | viscous fault, before | viscous fault, after | frictionless |
|---|---|---|---|
| saddle point | 2, 32, 31, 26, 27 | 2, 2, 2, 3, 3 | 3 at every increment |
| thin sheet | 2, 27, 21, 21, 21, 20 | 2 at every increment | 2 at every increment |

The frictionless contact refreshes its operator in place, and the fix makes the
interface-law branch do the same.

The same holds on the split mesh's own multigrid tail (no ``set_custom_fmg``):
4, 30, 23, 24, 23, 22 on the thin sheet before, 4-5 after. GAMG was never
affected.

The test records, after every linear solve of the rotated loop, the operator the
multigrid was handed (the velocity block on the saddle-point path, the whole
rotated operator on the thin-sheet path) and the finest smoother's operator, and
checks that they agree at every iterate. Agreement alone cannot see an operator
that is the right object holding iterate 0's values, so the test also bounds the
Newton increments and the velocity iterations of every increment. A viscosity
that depends on the velocity gives the loop several iterates. The frictionless
contact and GAMG are controls: they agreed before the fix.

The fixture is test_1090's: a fault split in place over a welded base hierarchy
(1/8, 1/16), so the finest level is the split mesh and carries the FAC patch.
This file carries no ``mpi(min_size=2)`` mark: it runs in the serial suite and
in the parallel batches.

    python -m pytest tests/parallel/test_1088_rotated_interface_operator_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1088_rotated_interface_operator_parallel.py
"""
import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg, rotated_bc

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])


@pytest.fixture(scope="module")
def base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=1)


@pytest.fixture(scope="module")
def flat_base():
    """The same box at the finest spacing with no refinement hierarchy, so the
    split mesh owns no multigrid tail and the velocity block falls back to GAMG."""
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 16, regular=False, qdegree=2)


def _operator_gap(mg):
    """|finest smoother's operator - the operator ``mg`` was handed|_F, relative."""
    handed = mg.getOperators()[1]
    finest = mg.getMGSmoother(mg.getMGLevels() - 1).getOperators()[1]
    gap = handed.copy()
    gap.axpy(-1.0, finest, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    value = gap.norm(PETSc.NormType.FROBENIUS) / handed.norm(PETSc.NormType.FROBENIUS)
    gap.destroy()
    return value


def _solve_recording(base, path, law, hierarchy, monkeypatch):
    """Solve on the split mesh by ``path`` with fault law ``law``; return one
    ``(operator gap, velocity iterations, velocity PC)`` record per Newton
    increment. ``hierarchy`` is how the velocity block gets its multigrid:
    ``set_custom_fmg`` over the base levels, the split mesh's own tail (none
    registered), or GAMG (``base`` has no hierarchy)."""
    records = []

    def recorded(original, saddle):
        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            ctx = out[2]
            if saddle:
                vel_ksp = ctx["pc"].getFieldSplitSubKSP()[0]
                records.append((_operator_gap(vel_ksp.getPC()),
                                vel_ksp.getIterationNumber(), ctx["velocity_pc"]))
            else:
                records.append((_operator_gap(ctx["pc"]),
                                ctx["ksp"].getIterationNumber(), ctx["velocity_pc"]))
            return out
        return wrapper

    monkeypatch.setattr(rotated_bc, "_solve_rotated_iterative",
                        recorded(rotated_bc._solve_rotated_iterative, True))
    monkeypatch.setattr(rotated_bc, "_solve_rotated_vector",
                        recorded(rotated_bc._solve_rotated_vector, False))

    mesh = base.add_fault(("Flt", TRACE))
    x, y = mesh.X
    if path == "stokes":
        v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
        solver = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        for wall in ("Bottom", "Top", "Left", "Right"):
            solver.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
        solver.petsc_use_pressure_nullspace = True
        u = v
    else:
        solver = uw.systems.ThinSheet(mesh)
        for wall in ("Bottom", "Top", "Left", "Right"):
            solver.add_essential_bc([y - sympy.Rational(1, 2), 0], wall)
        u = solver.u
    solver.add_fault_bc(law, "Flt")
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = (
        1 + 50 * (u.sym[0] ** 2 + u.sym[1] ** 2))
    solver.tolerance = 1e-6
    if hierarchy == "set_custom_fmg":
        levels = base._coarse_level_meshes()[:-1] + [base]
        custom_mg.set_custom_fmg(solver, levels, field_id=0 if path == "stokes" else None)
    solver.solve()
    assert solver._rotated_freeslip_info["converged"], f"{path}: did not converge"
    return records


# Bounds asserted, from the fixed build on this fixture. Velocity iterations per
# Newton increment, largest measured: 3 with set_custom_fmg, 6 on the mesh's own
# tail, 44 on GAMG. The stale operator of #894 took 20-34 on both multigrid arms.
# GAMG rebuilds its hierarchy from the live operator at every set-up, so it was
# never stale (the same 31-44 before and after) and is a control here, like the
# frictionless contact. Newton increments: 5 (saddle point) and 6 (thin sheet) on
# every arm; an operator holding iterate 0's values takes 49 or stalls.
VELOCITY_ITS_MAX = {"set_custom_fmg": 5, "mesh_tail": 8, "gamg": 60}
INCREMENTS_MAX = 8


@pytest.mark.parametrize("hierarchy", ["set_custom_fmg", "mesh_tail", "gamg"])
@pytest.mark.parametrize("law", [1.0, 0], ids=["viscous_fault", "frictionless"])
@pytest.mark.parametrize("path", ["stokes", "sheet"])
def test_the_finest_smoother_follows_every_iterate(base, flat_base, path, law, hierarchy,
                                                   monkeypatch):
    mesh = flat_base if hierarchy == "gamg" else base
    records = _solve_recording(mesh, path, law, hierarchy, monkeypatch)
    installed = {pc for _gap, _its, pc in records}
    assert installed == {"GAMG" if hierarchy == "gamg" else "custom-FMG"}, installed
    # one increment would leave nothing to go stale: the test needs the iterates
    assert 3 <= len(records) <= INCREMENTS_MAX, (
        f"{path}, law {law}, {hierarchy}: {len(records)} Newton increments")
    gaps = [gap for gap, _its, _pc in records]
    assert max(gaps) < 1e-12, (
        f"{path}, law {law}, {hierarchy}: the finest smoother's operator differs "
        f"from the operator the multigrid was handed, by {gaps} (relative, per "
        f"increment; #894)")
    its = [its for _gap, its, _pc in records]
    assert max(its) <= VELOCITY_ITS_MAX[hierarchy], (
        f"{path}, law {law}, {hierarchy}: velocity iterations per increment {its} "
        f"(20-34 with the stale operator of #894)")


def _viscous_fault_solve(base, use_lu):
    """The viscous-fault saddle-point solve by the direct-LU rotated path or by the
    multigrid one, solved tight enough to compare: ``(converged, peak slip)``."""
    from underworld3.utilities.fault_contact import fault_slip
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = base.add_fault(("Flt", TRACE))
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    stokes.add_fault_bc(1.0, "Flt")
    stokes.petsc_use_pressure_nullspace = True
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        1 + 50 * (v.sym[0] ** 2 + v.sym[1] ** 2))
    stokes.tolerance = 1e-8
    if use_lu:
        stokes._rotated_use_lu = True
    else:
        custom_mg.set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["nonlinear_iterations"] >= 3, info["nonlinear_iterations"]
    _s, slip, _leak = fault_slip(stokes, "Flt", info)
    peak = uw.mpi.comm.allreduce(float(np.abs(slip).max()) if len(slip) else 0.0, op=max)
    return bool(info["converged"]), peak


def test_the_direct_lu_path_takes_an_interface_law(base):
    """The direct-LU path keeps a fresh operator at every iterate: its pressure pin
    (``zeroRows``) drops the pinned row's entries, so the in-place refresh of the
    multigrid paths would not fit it. An in-place refresh there raised PETSc's
    "new nonzero caused a malloc" at iterate 1, which fails this test.

    The path does not converge on a fault contact at all (#899, on development
    too), so non-convergence alone is an expected failure. A fix to #899 that
    converges to a different slip fails the test; one that agrees passes it, and
    the xfail line should then go."""
    lu_converged, lu = _viscous_fault_solve(base, use_lu=True)
    mg_converged, mg = _viscous_fault_solve(base, use_lu=False)
    assert mg_converged
    if not lu_converged:
        pytest.xfail("#899: the direct-LU rotated path does not converge on a fault contact")
    assert abs(lu - mg) < 1e-6 * mg, (lu, mg)
