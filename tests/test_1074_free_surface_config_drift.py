"""FreeSurface's derived lids must not drift from the free solve.

`held` and `consistent` are separate Stokes solvers, and the free solve's
configuration is copied into them ONCE, when the manager is built. Changing the
free solve afterwards leaves them stale — and `h_inf`, the equilibrium the
surface relaxes toward, is recovered from the HELD solve. So the surface would
relax toward an equilibrium computed with the old rheology while the free solve
uses the new one: a wrong answer, not a crash.
"""

import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _model():
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    v = uw.discretisation.MeshVariable("V_fs", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_fs", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, -1.0])
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.petsc_options.delValue("ksp_monitor")
    return mesh, stokes


def test_changing_the_rheology_after_construction_is_refused():
    mesh, stokes = _model()
    fs = uw.systems.FreeSurface(stokes, "Top", buoyancy_scale=1.0)

    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1000.0

    with pytest.raises(RuntimeError, match="no longer match the free solve"):
        fs.solve()


def test_the_message_names_what_drifted_and_how_to_recover():
    mesh, stokes = _model()
    fs = uw.systems.FreeSurface(stokes, "Top", buoyancy_scale=1.0)
    stokes.bodyforce = sympy.Matrix([0, -5.0])

    with pytest.raises(RuntimeError) as excinfo:
        fs.solve()
    message = str(excinfo.value)
    assert "bodyforce" in message
    assert "BEFORE constructing" in message


def test_an_untouched_manager_solves():
    """The negative control: the guard must not refuse an ordinary run."""
    mesh, stokes = _model()
    fs = uw.systems.FreeSurface(stokes, "Top", buoyancy_scale=1.0)
    fs.solve()
    assert np.all(np.isfinite(np.asarray(stokes.u.data)))


def test_a_shared_symbolic_parameter_still_tracks():
    """A parameter whose value is an expression over mesh variables is shared
    symbolically and must NOT trip the guard — only re-assignment drifts."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=3
    )
    v = uw.discretisation.MeshVariable("V_sym", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_sym", mesh, 1, degree=1)
    T = uw.discretisation.MeshVariable("T_sym", mesh, 1, degree=2)
    T.array[:, 0, 0] = 0.5
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sympy.exp(-T.sym[0])
    stokes.bodyforce = sympy.Matrix([0, -T.sym[0]])
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    stokes.petsc_options.delValue("ksp_monitor")

    fs = uw.systems.FreeSurface(stokes, "Top", buoyancy_scale=1.0)
    T.array[:, 0, 0] = 0.9        # the FIELD changes, not the expression
    fs.solve()                    # must not raise
    assert np.all(np.isfinite(np.asarray(stokes.u.data)))
