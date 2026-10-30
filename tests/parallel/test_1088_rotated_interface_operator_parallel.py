"""The rotated Newton loop smooths every iterate with that iterate's operator (#894).

With a fault interface law, ``solve_rotated_freeslip`` added the interface
tangent into a NEW matrix at every Newton iterate (``Atot = Ahat.copy()``, then
``axpy``) and handed it to the cached KSP. PCMG keeps the finest smoother's
operator once it is set unless the new operator is the same Mat object (the
``opsset`` check in PETSc's ``PCSetUp_MG``), so the finest smoother, and the
Galerkin coarse operators built from it, stayed at iterate 0. The answer was
unaffected; the cost was iterations. Measured on this fixture with a viscous
fault: 2, 32, 31, 26, 27 velocity iterations per increment, against 3 for the
frictionless contact, whose operator is refreshed in place.

The test records, after every linear solve of the rotated loop, the operator the
multigrid was handed (the velocity block on the saddle-point path, the whole
rotated operator on the thin-sheet path) and the finest smoother's operator, and
checks that they agree at every iterate. A viscosity that depends on the
velocity gives the loop several iterates. The frictionless contact is the
control: it takes the other branch of the loop, and agreed before the fix.

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


def _operator_gap(mg):
    """|finest smoother's operator - the operator ``mg`` was handed|_F, relative."""
    handed = mg.getOperators()[1]
    finest = mg.getMGSmoother(mg.getMGLevels() - 1).getOperators()[1]
    gap = handed.copy()
    gap.axpy(-1.0, finest, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    value = gap.norm(PETSc.NormType.FROBENIUS) / handed.norm(PETSc.NormType.FROBENIUS)
    gap.destroy()
    return value


def _solve_recording(base, path, law, monkeypatch):
    """Solve on the split mesh by ``path`` with fault law ``law``; return one
    ``(operator gap, velocity iterations)`` record per Newton increment."""
    records = []

    def recorded(original, saddle):
        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            ctx = out[2]
            if saddle:
                vel_ksp = ctx["pc"].getFieldSplitSubKSP()[0]
                records.append((_operator_gap(vel_ksp.getPC()),
                                vel_ksp.getIterationNumber()))
            else:
                records.append((_operator_gap(ctx["pc"]),
                                ctx["ksp"].getIterationNumber()))
            return out
        return wrapper

    monkeypatch.setattr(rotated_bc, "_solve_rotated_iterative",
                        recorded(rotated_bc._solve_rotated_iterative, True))
    monkeypatch.setattr(rotated_bc, "_solve_rotated_vector",
                        recorded(rotated_bc._solve_rotated_vector, False))

    levels = base._coarse_level_meshes()[:-1] + [base]
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
    custom_mg.set_custom_fmg(solver, levels, field_id=0 if path == "stokes" else None)
    solver.solve()
    assert solver._rotated_freeslip_info["converged"], f"{path}: did not converge"
    return records


@pytest.mark.parametrize("law", [1.0, 0], ids=["viscous_fault", "frictionless"])
@pytest.mark.parametrize("path", ["stokes", "sheet"])
def test_the_finest_smoother_follows_every_iterate(base, path, law, monkeypatch):
    records = _solve_recording(base, path, law, monkeypatch)
    # one increment would leave nothing to go stale: the test needs the iterates
    assert len(records) >= 3, f"{path}: only {len(records)} Newton increments"
    gaps = [gap for gap, _its in records]
    assert max(gaps) < 1e-12, (
        f"{path}, law {law}: the finest smoother's operator differs from the "
        f"operator the multigrid was handed, by {gaps} (relative, per increment; #894)")
