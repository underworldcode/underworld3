"""A rotated free-slip wall meeting an essential wall must still hold its corner.

On a flat, axis-aligned lid the rotated constraint and `add_dirichlet_bc((None,
0.0), "Top")` are the same discrete constraint: the measure-weighted node normal
is exactly (0,1), so striking the rotated normal row is striking u_y.

They were not the same. `build_rotation` skipped any node with a velocity DOF
constrained out of the global vector — which is every node where the lid meets a
wall held by an essential condition — so the wall-normal component was left FREE
at those nodes. The lid leaked at its own end points (max|u_y| = 4.0e-3 against
|u|max 2.5e-2, entirely at the two corners) and the solve differed from the
component-Dirichlet one by 2e-3 globally, with an exact linear solve on both
sides. Issue #616; the corner reaction of #608 is the same node.

Run: pixi run python -m pytest tests/test_1066_rotated_meets_essential.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1]

import numpy as np
import sympy
import underworld3 as uw

RES = 16


def _solve(lid, direct=False):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=3
    )
    x, z = mesh.X
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.saddle_preconditioner = 1.0
    s.bodyforce = sympy.Matrix([[0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * z)]])
    s.tolerance = 1.0e-9
    s.petsc_use_pressure_nullspace = True
    s.add_dirichlet_bc((0.0, None), "Left")
    s.add_dirichlet_bc((0.0, None), "Right")
    s.add_dirichlet_bc((None, 0.0), "Bottom")
    if lid == "dirichlet":
        s.add_dirichlet_bc((None, 0.0), "Top")
        if direct:
            s.petsc_options["ksp_type"] = "preonly"
            s.petsc_options["pc_type"] = "lu"
    else:
        s.add_rotated_freeslip_bc(0.0, "Top")
        if direct:
            s._rotated_use_lu = True
    s.solve()
    assert s.snes.getConvergedReason() > 0
    return np.squeeze(np.asarray(v.array)).copy(), v.coords.copy()


def test_rotated_lid_holds_its_corners():
    """u.n = 0 on the whole lid, corners included, to round-off."""
    u, coords = _solve("rotated")
    top = np.abs(coords[:, 1] - 1.0) < 1.0e-9
    leak = np.abs(u[top, 1]).max() / np.abs(u).max()
    # It was 1.6e-1, at the two corner nodes and nowhere else.
    assert leak < 1.0e-12, f"the lid leaks at u.n = {leak:.2e} of |u|max"


def test_rotated_lid_reproduces_the_component_condition():
    """The two are the same discrete constraint on a flat wall, so the same answer.

    Judged against a monolithic DIRECT solve of the component-Dirichlet problem, so
    a difference cannot be blamed on either iterative solve.
    """
    reference, _c = _solve("dirichlet", direct=True)
    for lid in ("dirichlet", "rotated"):
        u, _c = _solve(lid)
        rel = np.linalg.norm(u - reference) / np.linalg.norm(reference)
        assert rel < 1.0e-7, f"{lid} lid differs from the direct reference by {rel:.2e}"
