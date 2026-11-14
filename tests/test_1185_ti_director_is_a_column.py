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


@pytest.mark.level_1
@pytest.mark.tier_b
@pytest.mark.parametrize("model", ["TransverseIsotropicFlowModel",
                                   "TransverseIsotropicVEPFlowModel"])
def test_numpy_directors_become_columns_and_bad_ones_are_refused(model):
    """A numpy array of any orientation becomes a column. A matrix-valued named
    expression is refused at the setter: on development a row solved 1.24e-4 off
    the column's answer, and a column went stale in plain TI (a changed value left
    the next solve unchanged, 0.31 off a fresh solve), both with no error. A
    variable passed itself is refused, naming ``.sym``."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.5)
    n = uw.discretisation.MeshVariable(f"n_1185b_{model}", mesh, 2, degree=1)
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = getattr(uw.constitutive_models, model)
    P = st.constitutive_model.Parameters
    r = float(R2)
    for given in (np.array([r, r]), np.array([[r, r]]), np.array([[r], [r]])):
        P.director = given
        assert P.director.sym.shape == (2, 1)
    with pytest.raises(ValueError, match="matrix"):
        P.director = uw.expression(rf"n_{{row,{model[:4]}}}", sympy.Matrix([[R2, R2]]), "row")
    with pytest.raises(ValueError, match="scalar expressions"):
        P.director = uw.expression(rf"n_{{col,{model[:4]}}}", sympy.Matrix([[R2], [R2]]), "column")
    with pytest.raises(TypeError, match=r"\.sym"):
        P.director = n


@pytest.mark.level_2
@pytest.mark.tier_b
def test_a_matrix_of_scalar_expressions_stays_live():
    """What the refusal points to: a Matrix whose entries are uw.expression
    scalars. Changing an entry's value changes the next solve."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    P = st.constitutive_model.Parameters
    P.shear_viscosity_0 = 1.0
    P.shear_viscosity_1 = 0.1
    nx = uw.expression(r"n_{x,1185}", 1.0, "director x")
    ny = uw.expression(r"n_{y,1185}", 0.0, "director y")
    P.director = sympy.Matrix([[nx], [ny]])
    st.add_dirichlet_bc((1.0, 0.0), "Top")
    st.add_dirichlet_bc((0.0, 0.0), "Bottom")
    st.solve()
    first = np.array(st.Unknowns.u.data)
    nx.sym, ny.sym = R2, R2
    st.solve()
    assert np.abs(np.asarray(st.Unknowns.u.data) - first).max() > 1.0e-3
