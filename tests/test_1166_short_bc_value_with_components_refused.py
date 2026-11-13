"""A boundary value shorter than the components it names is refused (#1129).

``add_dirichlet_bc(0.05, "Left", 1)``, the deprecated ``components=`` form with a
one-entry value for component 1, prescribed garbage on the wall: u_y on Left
read -2.7386, 0 and 0.866 in 2-D (down to -11.9 in 3-D) where 0.05 was asked,
and the solve converged. PETSc's essential-field callback writes the value array
from the generated function, which here sets only slot 0 (the free component);
the constrained slot 1 was never written. ``add_natural_bc`` with a short value
failed later with "Expecting reshape size to 1 but got ... 2". Both now refuse at
the call, naming the full-length form. The full-length forms are unchanged.
"""
import warnings

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


def _stokes(dim=2):
    if dim == 2:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    else:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.5)
    v = uw.discretisation.MeshVariable(f"V1166_{dim}", mesh, dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P1166_{dim}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.petsc_options.delValue("ksp_monitor")
    return mesh, v, stokes


def _u_y_on_left(v):
    X = np.asarray(v.coords)
    on = np.abs(X[:, 0]) < 1e-9
    return np.asarray(v.array)[on, 0, 1]


@pytest.mark.tier_b
@pytest.mark.parametrize("conds, components", [(0.05, 1), ((0.05,), (1,))])
def test_a_short_essential_value_with_components_is_refused(conds, components):
    mesh, v, stokes = _stokes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=r"\(None, 0\.05\)"):
            stokes.add_dirichlet_bc(conds, "Left", components)


@pytest.mark.tier_b
def test_a_short_essential_value_is_refused_in_three_dimensions():
    mesh, v, stokes = _stokes(dim=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="component 1"):
            stokes.add_dirichlet_bc(0.05, "Left", 1)


@pytest.mark.tier_b
def test_a_short_natural_value_with_components_is_refused():
    mesh, v, stokes = _stokes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="component 1"):
            stokes.add_natural_bc(0.05, "Top", 1)


@pytest.mark.tier_b
@pytest.mark.parametrize("conds, components", [((None, 0.05), None), ((0.0, 0.05), (1,))])
def test_full_length_values_still_prescribe_what_they_say(conds, components):
    mesh, v, stokes = _stokes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if components is None:
            stokes.add_dirichlet_bc(conds, "Left")
        else:
            stokes.add_dirichlet_bc(conds, "Left", components)
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    u_y = _u_y_on_left(v)
    # the Bottom corner takes Bottom's value, registered last
    X = np.asarray(v.coords)
    on = np.abs(X[:, 0]) < 1e-9
    away = np.asarray(X[on, 1]) > 1e-9
    assert np.abs(u_y[away] - 0.05).max() < 1e-12
