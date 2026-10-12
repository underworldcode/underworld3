"""Custom (non-nested) FMG on a velocity split that also holds a coupled field.

``add_coupled_field`` groups the coupled field with velocity in fieldsplit 0, so
that split's operator is [u, chi], interleaved point by point. A custom-P
hierarchy built for velocity alone does not match it: the transfer has the
velocity block's size and PETSc's Galerkin product fails. ``set_custom_fmg(...,
field_id=0)`` now builds one transfer per field of the split (each at its own
degree, BC-reduced per level) and scatters them into the split's numbering.

The hierarchy here is the shape the notch campaign uses: an INDEPENDENT coarse
mesh (a separate box, not nested) under the solver mesh's own nested levels.
The custom-P solve must converge and give the solution of the native nested FMG.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

SMOOTHING = 0.05    # l^2 of the screened strain-rate field


def coupled_driven_stokes(mesh, tag):
    """Lid-driven box, viscosity 1 + chi^2 / 2 with chi - l^2 lap chi = edot_II:
    two-way coupled, so the Newton operator's [u, chi] split is genuinely mixed."""
    v = uw.discretisation.MeshVariable(f"v_{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"p_{tag}", mesh, 1, degree=1, continuous=True)
    chi = uw.discretisation.MeshVariable(f"chi_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1 + 0.5 * chi.sym[0] ** 2
    zero = (0.0,) * mesh.dim
    stokes.add_dirichlet_bc((1.0,) + zero[1:], "Top")
    stokes.add_dirichlet_bc(zero, "Bottom")
    stokes.add_dirichlet_bc(zero, "Left")          # right wall traction-free
    stokes.add_coupled_field(
        chi, F0=chi.sym[0] - stokes.Unknowns.Einv2,
        F1=SMOOTHING * chi.sym.jacobian(mesh.CoordinateSystem.N))
    stokes.consistent_jacobian = True
    stokes.petsc_options["snes_rtol"] = 1.0e-10
    return stokes, v, chi


def rel_l2(a, b):
    num = uw.mpi.comm.allreduce(float(np.sum((np.asarray(a) - np.asarray(b)) ** 2)))
    den = uw.mpi.comm.allreduce(float(np.sum(np.asarray(b) ** 2)))
    return np.sqrt(num / den)


def check_custom_fmg_on_coupled_split():
    fine = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8,
        refinement=1, qdegree=3)
    coarse = uw.meshing.UnstructuredSimplexBox(          # independent, non-nested
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 3, qdegree=3)

    ref, v_ref, chi_ref = coupled_driven_stokes(fine, "ref")
    ref.solve()                                          # native nested FMG
    assert ref.snes.getConvergedReason() > 0

    sc, v_c, chi_c = coupled_driven_stokes(fine, "custom")
    tail = [coarse] + fine._coarse_level_meshes()[:-1]
    sc.set_custom_fmg(tail, field_id=0)
    sc.solve()

    assert sc.snes.getConvergedReason() > 0
    assert sc.snes.getFunctionNorm() < 1.0e-6
    h = sc._custom_mg["hierarchy"]
    chi_id = int(chi_c._solver_field_id)
    assert h.split_fields == [0, chi_id]                 # velocity AND chi
    vpc = sc.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC()
    assert vpc.getType() == "mg"
    assert vpc.getMGLevels() == len(tail) + 1
    A = vpc.getOperators()[0]
    assert h.transfers[-1].getSize()[0] == A.getSize()[0] == h.split_size
    # the same Newton solution as the native hierarchy
    assert rel_l2(v_c.data, v_ref.data) < 1.0e-6
    assert rel_l2(chi_c.data, chi_ref.data) < 1.0e-6
    assert chi_c.max() > 1.0                             # driven, not trivially zero


@pytest.mark.skipif(uw.mpi.size > 1, reason="serial test; see tests/parallel")
def test_custom_fmg_on_a_coupled_split_matches_the_native_hierarchy():
    check_custom_fmg_on_coupled_split()
