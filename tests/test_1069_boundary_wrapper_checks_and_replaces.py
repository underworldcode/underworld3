"""The boundary-condition wrapper checks what it is given and replaces kernels (#1138, #1069).

``PetscDSAddBoundary_UW`` looked a condition's label up by name and handed the
result to ``DMAddBoundary`` unchecked, and returned the boundary index as its
error code:

- #1138: a mesh wrapped around a bare DMPlex lists ``All_Boundaries`` and the DM
  holds no such label. An essential condition on it went in on a null label,
  and the solve ended in a segmentation fault (exit 59).
- the second face of #1138, found in the review of PR #1170: a field number
  the solver's DM does not have was refused by ``DMAddBoundary``, the refusal
  was discarded, and the solve converged without the condition.

Both now raise when the conditions are set up: the first with a ValueError
naming the label and the labels the DM holds, the second with PETSc's error
code (PETSc prints its message on stderr).

#1069: the boundary kernels were registered with ``PetscWeakFormAdd*``, which
appends. A narrow rewire (kernels only, conditions kept) registers them again,
so after N rewires each boundary key listed N + 1 kernels, every one called on
every boundary quadrature point and the last overwriting the rest. They are
registered with ``PetscWeakFormSet*`` now, which replaces: one kernel per key
after any number of rewires, and the rewired answer is a fresh build's.
"""

import os
import re

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


@pytest.mark.tier_b
def test_an_essential_condition_on_a_label_the_dm_lacks_is_refused():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    cells = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]], dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=PETSc.COMM_WORLD)
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    assert "All_Boundaries" in [b.name for b in mesh.boundaries]
    assert not mesh.dm.hasLabel("All_Boundaries")
    v = uw.discretisation.MeshVariable("V1138", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1138", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([[0.0, 1.0]])
    stokes.add_essential_bc((0.0, 0.0), "All_Boundaries")
    stokes.petsc_use_pressure_nullspace = True
    with pytest.raises(ValueError, match="All_Boundaries"):
        stokes.solve()


@pytest.mark.tier_b
def test_a_condition_on_a_field_the_solver_lacks_is_refused():
    # A third variable made first puts the pressure at position 2 on the mesh;
    # the solver's DM has fields 0 and 1 only (#1170, the Charter's field rule).
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    uw.discretisation.MeshVariable("other1170", mesh, 1, degree=1)
    v = uw.discretisation.MeshVariable("V1170", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1170", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([[0.0, 1.0]])
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    assert p.field_id == 2
    stokes.add_condition(p.field_id, "dirichlet", 0.0, "Top")
    with pytest.raises(ValueError, match="field 2, and the solver's DM has fields 0 to 1"):
        stokes.solve()


@pytest.mark.tier_b
def test_a_natural_condition_on_a_boundary_with_no_faces_is_refused():
    # Natural conditions are keyed on the consolidated UW_Boundaries label by
    # the boundary's value; a boundary the mesh lists and has no faces for
    # marked nothing, and the flux was dropped without a word.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    cells = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]], dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=PETSc.COMM_WORLD)
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    T = uw.discretisation.MeshVariable("T1211n", mesh, 1, degree=1)
    s = uw.systems.Poisson(mesh, u_Field=T)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1.0
    s.f = 1.0
    s.add_natural_bc([5.0], "All_Boundaries")
    with pytest.raises(ValueError, match="mark no point"):
        s.solve()


def _boundary_kernels(solver, tmp_path, tag):
    # One file for every rank: the viewer is the DS's communicator's, and
    # pytest gives each process a tmp_path of its own.
    """The kernel names PETSc lists for the boundary residual f0 of the solver's DS.

    Read off PETSc's ASCII view of the DS: the line after the
    ``boundary_residual_f0`` header lists the key and its kernels. The fixture
    has one natural condition, so one key. A change of the view's format fails
    here (no header, or no kernel found), it cannot pass."""
    path = uw.mpi.comm.bcast(os.path.join(tmp_path, f"ds_{tag}.txt"), root=0)
    viewer = PETSc.Viewer().createASCII(path, comm=solver.dm.getComm())
    solver.dm.getDS().view(viewer)
    viewer.destroy()
    uw.mpi.comm.barrier()
    text = open(path).read()
    entry = text.split("boundary_residual_f0")[1].split("\n", 2)[1]
    return re.findall(r"\w+_petsc_eqn_\d+", entry)


def _robin(tag, coefficient):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, qdegree=3)
    T = uw.discretisation.MeshVariable(f"T1069{tag}", mesh, 1, degree=2)
    k = uw.expression(rf"k_{{1069{tag}}}", coefficient(mesh))
    s = uw.systems.Poisson(mesh, u_Field=T)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1.0
    s.f = 1.0
    s.add_dirichlet_bc(0.0, "Bottom")
    s.add_natural_bc([-k * T.sym[0] + 0.2], "Top")
    s.tolerance = 1.0e-10
    return s, T, k


@pytest.mark.tier_b
def test_a_rewire_replaces_the_boundary_kernels(tmp_path):
    s, T, k = _robin("a", lambda m: sympy.Float(-1.0))
    s.solve(zero_init_guess=True)
    assert len(_boundary_kernels(s, tmp_path, "build")) == 1
    for r, value in enumerate((lambda m: -(1 + m.X[0]), lambda m: -(2 + m.X[1]))):
        k.sym = value(s.mesh)
        s._needs_function_rewire = True          # the narrow route: kernels only
        s.solve(zero_init_guess=True)
        assert len(_boundary_kernels(s, tmp_path, f"rewire{r}")) == 1

    # The answer was right before the fix too (the last kernel overwrote the
    # rest); the count above is what the fix changes. This holds the answer.
    fresh, T_fresh, _ = _robin("b", lambda m: -(2 + m.X[1]))
    fresh.solve(zero_init_guess=True)
    np.testing.assert_allclose(np.asarray(T.array), np.asarray(T_fresh.array), rtol=0, atol=1e-9)
