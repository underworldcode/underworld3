"""A declared rotation mode gets one verdict on every rank (#906).

Rotated free slip exempts a mode the mesh declares from the constraint test only while
every rotated wall's vertices are equidistant from the rotation's centre. Each rank
holds part of each wall, so the nearest and furthest distances are reduced over all
ranks before the verdict. The deformed fixture moves the outer wall's vertices off
their circle only near (1, 0), so at np >= 2 a rank can hold an untouched part of the
wall; a rank-local verdict would keep the mode there and drop it elsewhere.

- An annulus as built keeps its declared mode on every rank, and v.n is zero on every
  constrained row.
- With 1e-6 of radial deformation near (1, 0) on its outer wall, every rank drops the
  mode, rank 0 warns, and the constraint test then rejects it on every rank.
- An add_fault child keeps the declaration and the exemption on every rank. At np > 1
  the add-on partition can move the level the fault is cut on into a new mesh, which
  must inherit the declaration too: without that, a conforming sheet on a shell gave
  v_rms 1.79e-3 at np 1 and 3.45e-3 at np 4 (hyperion, on b7030602).
- The fixture has its power: at each rank count it was measured at, some rank holds no
  deformed wall vertex. Measured on Linux (hyperion, 2026-11-06), the ranks untouched
  were 1, 2, 2, 3, 4, 6 at np 2, 3, 4, 5, 6, 8. With the reduction made rank-local, the
  ranks disagree and the solve hangs into the supervisor at np 2, 3, 4 and 8. If a
  partitioner change makes the power assert fail, the fixture has lost its power, not
  the library its fix: widen the patch's complement (raise the x > 0.9 cut).

The serial file, tests/test_1131_rotated_nullspace_from_mesh.py, carries the nodal
measurements, which read local arrays.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1131_rotated_nullspace_from_mesh_parallel.py
"""
import warnings

import numpy as np
import pytest
import sympy
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

# the rank counts at which some rank was measured to hold no deformed wall vertex
HOLDS_AT = (2, 3, 4, 5, 6, 8)


def _annulus(deform):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1, qdegree=3)
    touched = 0
    if deform:
        X = np.array(mesh.X.coords)
        r = np.linalg.norm(X, axis=1)
        patch = np.isclose(r, 1.0, atol=1.0e-9) & (X[:, 0] > 0.9)
        X[patch] *= 1.0 + 1.0e-6
        touched = int(np.count_nonzero(patch))
        mesh.deform(X)
    v = uw.discretisation.MeshVariable(f"V{int(deform)}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{int(deform)}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    stokes.bodyforce = (sympy.cos(3 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([x / r, y / r]).T)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    return stokes, touched


def _normal_velocity(info):
    """The largest |v.n| on the constrained rows, over max |U|, over all ranks."""
    Q, U, rows = info["Q"], info["U"], info["normal_rows"]
    Uhat = U.duplicate()
    Q.mult(U, Uhat)
    rs, re = Uhat.getOwnershipRange()
    local = [g - rs for g in rows if rs <= g < re]
    vn = float(np.abs(Uhat.getArray()[local]).max()) if local else 0.0
    Uhat.destroy()
    return uw.mpi.comm.allreduce(vn, op=MPI.MAX) / U.norm(PETSc.NormType.NORM_INFINITY)


def _solve(deform):
    stokes, touched = _annulus(deform)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    info = stokes._rotated_freeslip_info
    warned = any("is not a sphere about the rotation centre" in str(w.message) for w in record)
    verdicts = uw.mpi.comm.allgather((tuple(info["declared_modes_exempt"]),
                                      info["rotation_gauge"]["orthonormal"]))
    return dict(warned=bool(uw.mpi.comm.allreduce(int(warned), op=MPI.MAX)),
                verdicts=verdicts, vn=_normal_velocity(info),
                untouched=uw.mpi.comm.allreduce(int(touched == 0)))


def test_the_annulus_as_built_keeps_its_mode_on_every_rank():
    out = _solve(deform=False)
    assert not out["warned"] and set(out["verdicts"]) == {((True,), 1)}, out
    assert out["vn"] < 1.0e-12, out


def test_a_wall_deformed_on_some_ranks_drops_the_mode_on_every_rank():
    out = _solve(deform=True)
    assert out["warned"] and set(out["verdicts"]) == {((False,), 0)}, out
    assert out["vn"] < 1.0e-12, out
    if uw.mpi.size in HOLDS_AT:
        assert out["untouched"] >= 1, ("no rank misses the deformation: the fixture has "
                                       "lost its power at this rank count", out)


def test_an_add_fault_child_keeps_the_declaration_on_every_rank():
    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1, refinement=1,
                              qdegree=2)
    child = base.add_fault([("Flt", np.array([[0.62, 0.10], [0.85, 0.30]]))])
    moved = bool(uw.mpi.comm.allreduce(int(getattr(child, "parent", base) is not base), op=MPI.MAX))
    declared = uw.mpi.comm.allgather(None if child._nullspace_rotations is None
                                     else len(child._nullspace_rotations))
    v = uw.discretisation.MeshVariable("Vf", child, 2, degree=2)
    p = uw.discretisation.MeshVariable("Pf", child, 1, degree=1)
    stokes = uw.systems.Stokes(child, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = child.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    stokes.bodyforce = (sympy.cos(3 * sympy.atan2(y, x)) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([x / r, y / r]).T)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    stokes.solve()
    info = stokes._rotated_freeslip_info
    verdicts = uw.mpi.comm.allgather((info["converged"], tuple(info["declared_modes_exempt"] or ()),
                                      info["rotation_gauge"]["orthonormal"]))
    assert set(declared) == {1}, (declared, moved)
    assert set(verdicts) == {(True, (True,), 1)}, (verdicts, moved)
