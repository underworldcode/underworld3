"""A Stokes solve with a velocity null-space mode completes when a rank holds no cells (#1179).

``_build_velocity_nullspace_vector`` evaluates the mode at the velocity nodes and
reshaped the result with ``reshape(n, -1)``. On a rank with no velocity nodes ``n``
is 0, numpy cannot infer ``-1`` from an empty array, and that rank alone raised
("cannot reshape array of size 0 into shape (0,newaxis)") while the others went on
into collectives: the run aborted on mismatched reductions or hung.
``mesh.nullspace_rotations`` sets such a mode by default on an annulus or a sphere.

The box of two cells puts both cells on one rank at every np >= 2 (asserted). Its
top and bottom walls fix u_y only, and its sides are free, so a translation along x
is a true null mode of the operator.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1179_velocity_nullspace_rank_without_cells_parallel.py
"""
import pytest
import sympy

import underworld3 as uw

from empty_rank_fixtures import cells_per_rank

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


def test_a_velocity_nullspace_mode_with_a_rank_without_cells():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(1, 2), qdegree=3)
    counts = cells_per_rank(mesh)
    assert 0 in counts, f"premise: some rank holds no cells ({counts})"

    v = uw.discretisation.MeshVariable("V1179", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1179", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    stokes.add_essential_bc((None, 0.0), "Top")
    stokes.add_essential_bc((None, 0.0), "Bottom")
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_velocity_nullspace_basis = [sympy.Matrix([1, 0])]
    stokes.tolerance = 1.0e-8

    stokes.solve()
    reasons = uw.mpi.comm.allgather(stokes.snes.getConvergedReason())
    assert all(r > 0 for r in reasons), reasons
