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


# Library code that puts one condition on the whole boundary loops over the
# boundaries that cover the exterior once: an Annulus lists Centre and has no
# faces for it, so a condition on Centre is refused, and it lists Lower, Upper
# and All_Boundaries, which share every exterior face (review of #1211).

def _annulus():
    return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.3, cellSize=0.1, qdegree=3)


@pytest.mark.tier_b
def test_the_boundaries_covering_the_exterior_once_on_an_annulus():
    mesh = _annulus()
    assert [b.name for b in mesh.boundaries] == ["Lower", "Upper", "Centre", "All_Boundaries"]
    assert [b.name for b in mesh._boundaries_that_mark_faces()] == ["Lower", "Upper", "All_Boundaries"]
    assert [b.name for b in mesh._boundaries_covering_the_exterior_once()] == ["All_Boundaries"]


def _inflow_after_two_steps(names=None):
    """The Eulerian history of a zero field under radial outflow, inflow value 1.

    ``names``, when given, replaces the mesh's choice of boundaries for the
    inflow condition."""
    mesh = _annulus()
    if names is not None:
        chosen = [b for b in mesh.boundaries if b.name in names]
        mesh._boundaries_covering_the_exterior_once = lambda: chosen
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T1069in", mesh, 1, degree=2)
    T.array[...] = 0.0
    manager = uw.systems.ddt.EulerianSUPG(mesh, T, sympy.Matrix([[x, y]]), vtype=uw.VarType.SCALAR,
                                          degree=2, continuous=True, order=1, transport_on_update=True)
    manager.inflow_value = sympy.Matrix([[1.0]])
    manager.initialise_history()
    for _ in range(2):
        manager.update_pre_solve(0.05)
    value = np.asarray(manager.psi_star[0].array).ravel()
    r = np.hypot(*np.asarray(manager.psi_star[0].coords).T)
    return value, r


@pytest.mark.tier_b
def test_the_eulerian_inflow_is_applied_once_per_face_on_an_annulus():
    # Lower and Upper hold different faces, so the run on them applies the
    # term once per face by construction; adding All_Boundaries applies it
    # twice. The default must be the first and differ from the second.
    default, r = _inflow_after_two_steps()
    once, _ = _inflow_after_two_steps(("Lower", "Upper"))
    twice, _ = _inflow_after_two_steps(("Lower", "Upper", "All_Boundaries"))
    assert default[r < 0.31].mean() > 0.5            # the inflow value entered at Lower
    assert np.abs(default[r > 0.99]).max() < 1e-8    # nothing reaches Upper, the outflow
    np.testing.assert_allclose(default, once, rtol=0, atol=1e-10)
    assert np.abs(twice - once).max() > 1e-3


@pytest.mark.tier_b
@pytest.mark.parametrize("advection", ["supg", "slcn"])
def test_the_level_set_far_field_on_an_annulus(advection):
    # A disc of radius 0.15 at (0.65, 0) turned about the origin for one step.
    # On development this was a segmentation fault (the condition on Centre).
    from underworld3.systems import level_set
    mesh = _annulus()
    x, y = mesh.X
    psi = uw.discretisation.MeshVariable(f"psi1069{advection}", mesh, 1, degree=2)
    eps = level_set.interface_thickness(mesh, psi, scale=2.0)
    X = np.asarray(psi.coords)
    level_set.initialise_psi(psi, eps, signed_distance=0.15 - np.hypot(X[:, 0] - 0.65, X[:, 1]))
    solver = uw.systems.LevelSetSolver(psi, velocity=sympy.Matrix([[-y, x]]), epsilon=eps,
                                       advection=advection, far_field=0.0, reini_steps=1,
                                       reini_frequency=5)
    before = solver.interface_volume()
    solver.solve(0.05)
    r = np.hypot(X[:, 0], X[:, 1])
    on_the_walls = (r < 0.3 + 1e-6) | (r > 1.0 - 1e-6)
    assert on_the_walls.sum() > 0
    # The far field holds the walls at 0; SLCN's mass correction then shifts
    # the whole field by one constant, so its walls hold one value.
    walls = np.asarray(psi.array).ravel()[on_the_walls]
    assert walls.max() == walls.min()
    if not solver.conserve_mass:
        assert walls.max() == 0.0
    # SLCN's mass correction holds the volume; SUPG's measured drift is 0.7 %
    assert solver.interface_volume() == pytest.approx(before, rel=1e-10 if advection == "slcn" else 0.02)
