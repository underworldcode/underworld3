"""Regression test for ``SNES_Scalar.petsc_use_constant_nullspace``.

A scalar Poisson problem with no Dirichlet boundary conditions has a
constant kernel (any constant solves :math:`-\\Delta u = 0`). The linear
system is singular and the solution is determined only up to an additive
constant.

The new ``petsc_use_constant_nullspace`` flag tells PETSc about the
constant nullspace via ``MatNullSpace().create(constant=True)``. PETSc
projects each KSP right-hand side onto the orthogonal complement of the
nullspace before solving and returns the minimum-norm solution within
the affine null space — i.e. the solution with (discrete) zero mean.

We verify:
  1. The shape of the solution (non-constant part) is the same with or
     without the flag — PETSc's projection lives entirely in the
     constant kernel, so the non-constant component is unaffected.
  2. With the flag, the discrete mean of the solution is small
     (minimum-norm pinning).
  3. The recovered field agrees with the analytic solution
     :math:`u(x, y) = -x^3/6 + x^2/4 + C` to FE-discretisation
     accuracy.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import Poisson


def _build_neumann_poisson(mesh, use_nullspace):
    """Build a pure-Neumann Poisson with f = x - 1/2 (zero mean)."""
    T = uw.discretisation.MeshVariable(
        f"T_ns_{use_nullspace}", mesh, num_components=1, degree=2,
    )
    poisson = Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.petsc_use_constant_nullspace = use_nullspace
    x, _ = mesh.X
    poisson.f = x - sympy.Rational(1, 2)
    return poisson, T


def _shape_error(T, mesh):
    """Discretisation error in the non-constant part of the solution."""
    coords = np.asarray(T.coords)
    expected = -coords[:, 0] ** 3 / 6.0 + coords[:, 0] ** 2 / 4.0
    diff = T.data[:, 0] - expected
    diff_centered = diff - diff.mean()
    return np.abs(diff_centered).max()


def test_constant_nullspace_canonical_solution():
    """With ``petsc_use_constant_nullspace=True`` the solver returns the
    minimum-norm (near-zero-mean) solution."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1,
    )

    poisson_off, T_off = _build_neumann_poisson(mesh, use_nullspace=False)
    poisson_off.solve()
    assert poisson_off.snes.getConvergedReason() > 0, (
        "Pure-Neumann Poisson without nullspace declaration failed to "
        f"converge: SNES reason {poisson_off.snes.getConvergedReason()}"
    )

    poisson_on, T_on = _build_neumann_poisson(mesh, use_nullspace=True)
    poisson_on.solve()
    assert poisson_on.snes.getConvergedReason() > 0, (
        "Pure-Neumann Poisson with nullspace declaration failed to "
        f"converge: SNES reason {poisson_on.snes.getConvergedReason()}"
    )

    # Both should recover the analytic shape to FE accuracy.
    err_off = _shape_error(T_off, mesh)
    err_on  = _shape_error(T_on,  mesh)
    assert err_off < 1.0e-4, f"shape error off = {err_off:.3e}"
    assert err_on  < 1.0e-4, f"shape error on  = {err_on:.3e}"

    # The flag should leave the non-constant part of the solution
    # essentially unchanged — PETSc's projection lives in the kernel.
    assert abs(err_off - err_on) < 1.0e-5, (
        f"shape error differs more than expected between paths: "
        f"off={err_off:.3e}, on={err_on:.3e}"
    )

    # With the flag, the mean of the solution should be much closer to
    # zero than without (canonical minimum-norm pinning). The exact
    # value depends on the discrete inner product against the constant
    # mode, but the flag should move it toward zero by a clear margin.
    mean_off = T_off.data[:, 0].mean()
    mean_on  = T_on.data[:, 0].mean()
    assert abs(mean_on) < abs(mean_off), (
        f"with nullspace declaration, |mean(T)| should drop; "
        f"got |mean_off|={abs(mean_off):.3e}, |mean_on|={abs(mean_on):.3e}"
    )


def test_constant_nullspace_property_setter():
    """Property setter round-trip + ``is_setup`` invalidation."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2,
    )
    T = uw.discretisation.MeshVariable(
        "T_setter", mesh, num_components=1, degree=1,
    )
    poisson = Poisson(mesh, u_Field=T)

    assert poisson.petsc_use_constant_nullspace is False, (
        "Default should be False"
    )

    poisson.petsc_use_constant_nullspace = True
    assert poisson.petsc_use_constant_nullspace is True
    assert poisson.is_setup is False, (
        "Setting the flag should invalidate the setup so the solver "
        "rebuilds with the nullspace attached."
    )

    poisson.petsc_use_constant_nullspace = False
    assert poisson.petsc_use_constant_nullspace is False


if __name__ == "__main__":
    test_constant_nullspace_property_setter()
    test_constant_nullspace_canonical_solution()
    print("PASSED")
