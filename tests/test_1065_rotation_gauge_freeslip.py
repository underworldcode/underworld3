"""Regression test for the free-slip velocity rigid-rotation gauge fix.

On a fully free-slip (no Dirichlet velocity BC) domain the rigid-body rotations
are a true nullspace of the velocity block (A_uu·rotation = 0). PETSc's
fieldsplit/Schur inner velocity solve has no rotation nullspace attached, so it
leaves an UNCONSTRAINED rigid rotation in the converged velocity. The operator is
blind to it (the true residual still converges to machine precision), but the
rotation amplitude is partition-dependent — so the tangential velocity differs
serial-vs-parallel even though the physical (radial) flow does not.

``SNES_Stokes_SaddlePt.solve`` projects this rotation gauge out of the converged
solution (``_remove_velocity_rotation_gauge``). This test pins that behaviour: on
a fully free-slip annulus driven by a PURELY RADIAL body force (which does zero
work against a rigid rotation, so the physical solution carries no net rotation),
the converged velocity's projection onto the rigid-rotation mode must be ~0.

Without the fix the solve leaves a non-trivial rotation component (~1e-3 of the
flow); with the fix it is at round-off. Serial, fast — no MPI required.
"""

import numpy as np
import sympy
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.timeout(120)]


def _freeslip_rotation_coefficient():
    """Solve a fully free-slip annulus with a radial body force and return the
    normalised projection of the velocity onto the rigid-rotation mode
    t(x,y) = (-y, x): |<v, t>| / (|v| |t|).  ~0 iff the rotation gauge is removed."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=0.12, qdegree=4)
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    X = mesh.CoordinateSystem.X
    unit_r = mesh.CoordinateSystem.unit_e_0

    st = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.tolerance = 1.0e-10
    st.petsc_use_nullspace = True            # build the rotation (+ pressure) nullspace
    # Fully free-slip: BOTH boundaries are constraint BCs, NO essential velocity
    # BC -> the rigid-rotation nullspace is ACTIVE (the case the fix targets).
    st.add_constraint_bc("Upper", g=0.0, normal=unit_r)
    st.add_constraint_bc("Lower", g=0.0, normal=unit_r)
    # purely radial drive: does zero work against a rigid rotation, so the
    # physical solution has no net rotation component.
    st.bodyforce = 1.0e2 * sympy.sin(3 * sympy.atan2(X[1], X[0])) * unit_r
    st.solve(zero_init_guess=True)

    # rigid-rotation field t = (-y, x); project the converged velocity onto it.
    rot = sympy.Matrix([-X[1], X[0]])
    v_dot_rot = float(uw.maths.Integral(mesh, v.sym.dot(rot)).evaluate())
    rot_dot_rot = float(uw.maths.Integral(mesh, rot.dot(rot)).evaluate())
    v_dot_v = float(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate())
    # normalised rotation coefficient: |<v,t>| / (|v| |t|)
    return abs(v_dot_rot) / (np.sqrt(v_dot_v) * np.sqrt(rot_dot_rot))


@pytest.mark.tier_a
@pytest.mark.level_1
def test_freeslip_velocity_has_no_rotation_gauge():
    """The converged free-slip velocity must carry no rigid-rotation component:
    the solver projects the rotation gauge out. Guards
    ``_remove_velocity_rotation_gauge`` — without it this coefficient is ~1e-3."""
    coeff = _freeslip_rotation_coefficient()
    assert coeff < 1.0e-8, (
        "free-slip velocity retains a rigid-rotation gauge component "
        f"(|<v,rot>|/(|v||rot|) = {coeff:.3e}); the post-solve rotation "
        "projection (_remove_velocity_rotation_gauge) did not run / is ineffective"
    )


if __name__ == "__main__":
    print(f"rotation coefficient = {_freeslip_rotation_coefficient():.6e}")
