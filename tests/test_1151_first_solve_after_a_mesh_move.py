"""The first solve after a mesh move assembles with the moved fields.

``remesh_with_field_transfer`` writes every transferred variable through a plain view
of its local vector, which does not mark the mesh's assembly vector stale as a user's
write does. A solver that had solved before the move then assembled its first solve
after it with the transferred fields at their pre-move values; a second solve, or any
solve elsewhere, set the mark and was right. On a viscosity read from a DG0 field the
first solve was 6 % off in velocity and 80 % in pressure after ``mesh.deform``, and
0.2 % and 15 % after one MMPDE move. The transfer now marks the mesh stale.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _fixture():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 12, regular=False, qdegree=2)
    # varying in x, so that an x move changes every interior value (hyperion's
    # review: a DG0 band whose centroids do not cross it, or a field an x move does
    # not change, passes whatever the assembly reads)
    weak = uw.discretisation.MeshVariable("weak1151", mesh, 1, degree=1)
    c = np.asarray(weak.coords)
    weak.array[:, 0, 0] = 0.5 + 0.5 * np.sin(2 * np.pi * c[:, 0])
    return mesh, weak


def _stokes(mesh, weak, tag):
    _x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"V1151{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1151{tag}", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0 - 0.99 * weak.sym[0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    s.bodyforce = sympy.Matrix([[0, sympy.sin(sympy.pi * mesh.X[0])]])
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-6
    return s, v, p


def _relative(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _move_deform(mesh):
    c = np.asarray(mesh.X.coords).copy()
    c[:, 0] += 0.02 * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])   # zero on the walls
    mesh.deform(c)


def _move_mmpde(mesh):
    x, y = mesh.X
    uw.meshing.node_redistribution(
        mesh, 1 + 6 * sympy.exp(-(((x - 0.45) ** 2 + (y - 0.5) ** 2) / 0.03)))


@pytest.mark.parametrize("move", [_move_deform, _move_mmpde], ids=["deform", "mmpde"])
def test_the_first_solve_after_a_move_reads_the_moved_fields(move):
    mesh, weak = _fixture()
    s, v, p = _stokes(mesh, weak, "a")
    s.solve(zero_init_guess=True)
    before = np.asarray(weak.array).copy()
    move(mesh)
    # the transfer must change the field, or the comparison passes whatever the
    # assembly reads
    assert np.count_nonzero(np.asarray(weak.array) != before) > before.shape[0] // 2
    s.solve(zero_init_guess=True)              # the solve that read pre-move values
    built_after, vb, pb = _stokes(mesh, weak, "b")
    built_after.solve(zero_init_guess=True)
    pa = np.asarray(p.array).ravel()
    pr = np.asarray(pb.array).ravel()
    # velocity, without the mark: 2.1e-2 (deform) and 4.1e-2 (MMPDE)
    assert _relative(v.array, vb.array) < 1.0e-5
    assert _relative(pa - pa.mean(), pr - pr.mean()) < 1.0e-4
