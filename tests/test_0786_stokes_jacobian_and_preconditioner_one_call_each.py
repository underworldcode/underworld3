"""The Stokes Jacobian and the matrix its preconditioner is built from, assembled
one to a call, are the matrices PETSc assembles in one pass (#962).

PETSc's one pass leaves early on a rank without cells and skips an assembly of
the Jacobian that the other ranks make, so the solver's local Jacobian routine
hands PETSc one matrix and one set of kernels at a time
(``UW_DMPlexSNESComputeJacobianFEM``). On one rank PETSc's own routine is still
right, and it is the baseline here: both matrices must equal it in structure and
bit for bit in value, for a plain solve, for Nitsche boundary terms (boundary
kernels in both sets) and for a constrained solve (a multiplier block whose
preconditioner kernel is not its Jacobian kernel).

One rank only: the baseline is PETSc's routine, which is the defect on a rank
without cells. ``tests/parallel/test_0786_stokes_rank_without_cells_parallel.py``
has the solves that complete there.
"""
import ctypes
import os
import sys

import petsc4py
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1]


def _solver(solver_class, tag):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5, refinement=1, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V0786{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P0786{tag}", mesh, 1, degree=1)
    stokes = solver_class(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x = mesh.X[0]
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])
    stokes.petsc_options["snes_type"] = "ksponly"
    return stokes


def _plain():
    stokes = _solver(uw.systems.Stokes, "a")
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_essential_bc((0.0, 0.0), wall)
    return stokes


def _nitsche():
    stokes = _solver(uw.systems.Stokes, "b")
    stokes.add_essential_bc((0.0, 0.0), "Left")
    stokes.add_essential_bc((0.0, 0.0), "Right")
    stokes.add_nitsche_bc(0.0, "Top")
    stokes.add_nitsche_bc(0.0, "Bottom")
    return stokes


def _constrained():
    stokes = _solver(uw.systems.Stokes_Constrained, "c")
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
    stokes.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))
    return stokes


def _petsc():
    """PETSc's library, for the two calls below that petsc4py does not wrap."""
    config = petsc4py.get_config()
    name = "libpetsc.dylib" if sys.platform == "darwin" else "libpetsc.so"
    return ctypes.CDLL(
        os.path.join(config["PETSC_DIR"], config.get("PETSC_ARCH", ""), "lib", name))


def _use_the_one_pass_assembly_of_petsc(dm):
    """Put PETSc's own local routines back on ``dm``."""
    set_local_fem = _petsc().DMPlexSetSNESLocalFEM
    set_local_fem.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p]
    set_local_fem.restype = ctypes.c_int
    assert set_local_fem(dm.handle, False, None) == 0


def _has_the_one_pass_assembly_of_petsc(dm):
    """Whether the local Jacobian routine of ``dm`` is PETSc's own."""
    library = _petsc()
    get_local = library.DMSNESGetJacobianLocal
    get_local.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                          ctypes.POINTER(ctypes.c_void_p)]
    get_local.restype = ctypes.c_int
    routine, context = ctypes.c_void_p(), ctypes.c_void_p()
    assert get_local(dm.handle, ctypes.byref(routine), ctypes.byref(context)) == 0
    return routine.value == ctypes.cast(library.DMPlexSNESComputeJacobianFEM, ctypes.c_void_p).value


def _assembled(stokes):
    """Copies of the Jacobian and of the preconditioner's matrix at the solution."""
    J, P, _ = stokes.snes.getJacobian()
    stokes.snes.computeJacobian(stokes.snes.getSolution(), J, P)
    return J.copy(), P.copy()


@pytest.mark.tier_b
@pytest.mark.parametrize("build", [_plain, _nitsche, _constrained])
def test_each_matrix_is_the_one_petsc_assembles_in_one_pass(build):
    stokes = build()
    stokes.solve()

    # the premise: the solver's own routine made these, and they are two matrices
    # that differ, so neither stands in for the other
    assert not _has_the_one_pass_assembly_of_petsc(stokes.dm)
    J, P = _assembled(stokes)
    assert J.handle != P.handle
    assert J.norm() > 0.0 and P.norm() > 0.0
    assert not J.equal(P)

    _use_the_one_pass_assembly_of_petsc(stokes.dm)
    assert _has_the_one_pass_assembly_of_petsc(stokes.dm)
    J_one_pass, P_one_pass = _assembled(stokes)

    assert J.equal(J_one_pass)
    assert P.equal(P_one_pass)
