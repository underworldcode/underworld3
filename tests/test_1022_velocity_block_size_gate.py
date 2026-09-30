"""`mat_block_size` is declared only when the velocity block really is node-blocked.

The GAMG bundle declares ``mat_block_size`` so coarsening aggregates nodes rather
than scalars (#579). ``MatSetFromOptions`` hands that to
``PetscLayoutSetBlockSize``, which is a hard error — not a hint it may decline —
when a rank's local row count is not divisible:

    Arguments are incompatible
    Local size 67 not compatible with block size 2

Whether it divides is a property of the particular COMBINATION of boundary
conditions, not of their kind. Constrained degrees of freedom are absent from the
field's global section, so on a 3x3 P2 velocity field:

    no velocity BCs                       98    even
    full vector Dirichlet, every wall     50    even
    component-wise free slip              70    even
    (0, 0) Bottom, (0, None) elsewhere    67    ODD

which is why the free-slip measurements behind #579 never met it and
``test_1010_stokesCart`` did.

The second test is the one that matters. A gate that withdrew the option
unconditionally would fix the crash and silently discard everything #579 was
for, and the first test alone cannot tell the two apart.
"""

import pytest
import sympy

import underworld3 as uw
from petsc4py import PETSc

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _stokes(tag, bcs):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3))
    v = uw.discretisation.MeshVariable(f"Vb{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"Pb{tag}", mesh, 1, degree=1, continuous=True)

    solver = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = 1
    solver.bodyforce = sympy.Matrix([0, -1])
    for value, boundary in bcs:
        solver.add_dirichlet_bc(value, boundary)

    return solver


def _block_size_option(solver):
    key = f"{solver.petsc_options_prefix}fieldsplit_velocity_mat_block_size"
    options = PETSc.Options()

    return options.getInt(key) if key in options else None


def test_an_indivisible_velocity_block_solves():
    """The asymmetric mix that made PCSetUp_FieldSplit refuse the sub-matrix."""

    solver = _stokes(
        "odd",
        [
            ((0.0, 0.0), "Bottom"),
            ((0.0, None), "Top"),
            ((0.0, None), "Left"),
            ((0.0, None), "Right"),
        ],
    )

    solver.solve()

    assert _block_size_option(solver) is None, (
        "the velocity block is not node-blocked here, so mat_block_size must "
        "have been withdrawn"
    )


def test_a_divisible_velocity_block_keeps_the_block_size():
    """The control: where the declaration is valid it survives.

    Without this, withdrawing `mat_block_size` on every solve would pass the
    test above and quietly undo #579.
    """

    solver = _stokes(
        "even",
        [
            ((0.0, 0.0), "Bottom"),
            ((0.0, 0.0), "Top"),
            ((0.0, 0.0), "Left"),
            ((0.0, 0.0), "Right"),
        ],
    )

    solver.solve()

    assert _block_size_option(solver) == solver.mesh.dim, (
        "a fully Dirichlet velocity block is node-blocked, so GAMG should still "
        "be told its block size"
    )
