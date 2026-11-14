"""A transverse-isotropic model stores its director as a column, whatever shape
it is given (#1185).

``TransverseIsotropicVEPFlowModel`` with a yield stress reads the director as a
column (``T = Edot * n``) and raised "Matrix size mismatch: (2, 2) * (1, 2)" for
a row, which is what a mesh variable's ``.sym`` is. The setter now stores a
column, so every reader sees one shape.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

R2 = 1 / sympy.sqrt(2)


def _vep(director):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel
    P = st.constitutive_model.Parameters
    P.shear_viscosity_0 = 1.0
    P.shear_viscosity_1 = 0.1
    P.yield_stress = 1.0
    P.director = director(mesh)
    st.add_dirichlet_bc((1.0, 0.0), "Top")
    st.add_dirichlet_bc((0.0, 0.0), "Bottom")
    return st


@pytest.mark.level_2
@pytest.mark.tier_b
def test_a_row_director_solves_as_a_column_does():
    row = _vep(lambda m: sympy.Matrix([[R2, R2]]))
    col = _vep(lambda m: sympy.Matrix([[R2], [R2]]))
    row.solve(timestep=0.01)
    col.solve(timestep=0.01)
    assert np.allclose(np.asarray(row.Unknowns.u.data), np.asarray(col.Unknowns.u.data),
                       atol=1e-10)


@pytest.mark.level_1
@pytest.mark.tier_b
@pytest.mark.parametrize("model", ["TransverseIsotropicFlowModel",
                                   "TransverseIsotropicVEPFlowModel"])
def test_the_director_is_stored_as_a_column(model):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5)
    n = uw.discretisation.MeshVariable(f"n_1185_{model}", mesh, 2, degree=1)
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = getattr(uw.constitutive_models, model)
    P = st.constitutive_model.Parameters
    for given in (sympy.Matrix([[R2, R2]]), sympy.Matrix([[R2], [R2]]), n.sym):
        P.director = given
        assert P.director.sym.shape == (2, 1)
