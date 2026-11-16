"""Scoped resets of a solver: ``solver.reset.conditions()``, ``.nullspace()``,
``.discretisation()`` and ``.all()`` (#1169).

``_reset()`` emptied the essential and natural conditions and dropped the
discretisation, and kept the Stokes nullspace objects (built on the old walls'
numbering: the next solve with other walls diverged or stopped with PETSc error
75), the rotated free-slip conditions and the constraint conditions with their
multipliers. Each scope now clears exactly what it names; the configuration
(constitutive model, tolerances, options, nullspace settings) is never reset;
every scope is idempotent and the same on every rank. ``_reset()`` is a
deprecated alias of ``reset.all()``.

Serial and parallel; the empty-rank case runs from np 2.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1169_solver_reset_scopes_parallel.py
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

SLIP_ENDS = {"Top": (None, 0.0), "Bottom": (None, 0.0), "Left": (0.0, 0.0), "Right": (0.0, 0.0)}
SLIP_SIDES = {"Top": (0.0, 0.0), "Bottom": (0.0, 0.0), "Left": (0.0, None), "Right": (0.0, None)}
NO_SLIP = dict.fromkeys(SLIP_ENDS, (0.0, 0.0))
FREE_SLIP = {"Top": (None, 0.0), "Bottom": (None, 0.0), "Left": (0.0, None), "Right": (0.0, None)}
_count = [0]


def _box(walls, solver_class=uw.systems.Stokes, elementRes=(4, 4)):
    _count[0] += 1
    mesh = uw.meshing.StructuredQuadBox(elementRes=elementRes, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V1169_{_count[0]}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1169_{_count[0]}", mesh, 1, degree=1)
    stokes = solver_class(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x) * (1 + y)]])
    stokes.tolerance = 1.0e-10
    stokes.petsc_options.delValue("ksp_monitor")
    for wall, velocity in walls.items():
        stokes.add_essential_bc(velocity, wall)
    return stokes, v


def _speed(v):
    return uw.mpi.comm.allreduce(
        float(np.linalg.norm(np.asarray(v.array)[:, 0, :], axis=1).max(initial=0.0)), op=MPI.MAX)


def _configuration(stokes):
    return (stokes.tolerance, stokes.petsc_use_pressure_nullspace,
            tuple(stokes.petsc_velocity_nullspace_basis),
            stokes.petsc_options.getString("snes_type", ""),
            stokes.constitutive_model.Parameters.shear_viscosity_0.sym)


def test_the_scopes_each_solver_has():
    stokes, _ = _box(NO_SLIP)
    assert stokes.reset.scopes == ("conditions", "nullspace", "discretisation", "all")
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    T = uw.discretisation.MeshVariable("T1169", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    assert poisson.reset.scopes == ("conditions", "discretisation", "all")


def test_each_scope_clears_what_it_names_and_the_configuration_survives():
    stokes, v = _box({"Bottom": (0.0, 0.0)}, uw.systems.Stokes_Constrained)
    stokes.petsc_use_pressure_nullspace = True
    stokes.add_natural_bc((0.0, 0.0), "Top")
    stokes.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
    stokes.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))
    stokes.solve()
    configuration = _configuration(stokes)

    stokes.reset.nullspace()
    assert stokes._stokes_nullspace is None
    assert len(stokes.essential_bcs) == 1 and len(stokes._block_constraint_bcs) == 2
    assert stokes.dm is not None

    stokes.reset.discretisation()
    assert stokes.dm is None and stokes.snes is None
    assert len(stokes.essential_bcs) == 1 and len(stokes.natural_bcs) == 1
    assert len(stokes._block_constraint_bcs) == 2

    stokes.reset.conditions()
    stokes.reset.conditions()          # idempotent
    assert stokes.essential_bcs == [] and stokes.natural_bcs == []
    assert stokes._block_constraint_bcs == [] and stokes._multipliers == []
    assert _configuration(stokes) == configuration


def test_rotated_free_slip_conditions_are_conditions():
    stokes, _ = _box({"Bottom": (0.0, 0.0)})
    stokes.add_rotated_freeslip_bc(0.0, "Top")
    stokes.reset.discretisation()
    assert len(stokes._rotated_freeslip_bcs) == 1
    stokes.reset.conditions()
    assert stokes._rotated_freeslip_bcs == []


@pytest.mark.parametrize("first, second", [(SLIP_ENDS, SLIP_SIDES), (NO_SLIP, FREE_SLIP)],
                         ids=["other walls, same unknowns", "more unknowns"])
def test_after_reset_all_the_solve_is_a_new_solvers(first, second):
    """#1169: diverged (reason -6) and PETSc error 75 before."""
    fresh, v_fresh = _box(second)
    fresh.petsc_use_pressure_nullspace = True
    fresh.solve()
    stokes, v = _box(first)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    stokes.reset.all()
    for wall, velocity in second.items():
        stokes.add_essential_bc(velocity, wall)
    stokes.solve(zero_init_guess=True)
    assert stokes.snes.getConvergedReason() > 0
    assert _speed(v) == pytest.approx(_speed(v_fresh), rel=1e-8)


def test_constraints_registered_again_after_reset_all():
    def constrain(stokes):
        stokes.add_essential_bc((0.0, 0.0), "Bottom")
        stokes.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
        stokes.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))

    fresh, v_fresh = _box({}, uw.systems.Stokes_Constrained)
    constrain(fresh)
    fresh.solve()
    stokes, v = _box({}, uw.systems.Stokes_Constrained)
    constrain(stokes)
    stokes.solve()
    stokes.reset.all()
    constrain(stokes)
    assert len(stokes._multipliers) == 2
    stokes.solve(zero_init_guess=True)
    assert _speed(v) == pytest.approx(_speed(v_fresh), rel=1e-10)


def test_the_old_name_is_a_deprecated_alias_of_reset_all():
    stokes, _ = _box({"Bottom": (0.0, 0.0)})
    stokes.add_rotated_freeslip_bc(0.0, "Top")
    with pytest.warns(FutureWarning, match=r"reset\.all\(\)"):
        stokes._reset()
    assert stokes.essential_bcs == [] and stokes._rotated_freeslip_bcs == []


def test_reset_all_with_a_rank_without_cells():
    """The box of two cells puts both on one rank at np >= 2."""
    if uw.mpi.size < 2:
        pytest.skip("needs a rank without cells")
    from empty_rank_fixtures import cells_per_rank

    stokes, v = _box({"Top": (None, 0.0), "Bottom": (None, 0.0)}, elementRes=(1, 2))
    assert 0 in cells_per_rank(stokes.mesh)
    stokes.solve()
    first = _speed(v)
    stokes.reset.all()
    stokes.add_essential_bc((None, 0.0), "Top")
    stokes.add_essential_bc((None, 0.0), "Bottom")
    v.array[...] = 0.0
    stokes.solve(zero_init_guess=True)
    assert stokes.snes.getConvergedReason() > 0
    assert _speed(v) == pytest.approx(first, rel=1e-8)
