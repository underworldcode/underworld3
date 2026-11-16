"""Two meshes in one process, in one model or two: the second is not handed the
first's symbols.

unwrap returns raw coordinate BaseScalars, and SymPy's global cache returns a
structurally equal expression built earlier: the same names and hashes, but the
first mesh's coordinate scalars and variable functions, which are not equal to
the new mesh's. After a model with a coordinate-dependent expression on its own
mesh, a reset, and then a FreeSurface whose viscosity depends on the coordinates
of a new mesh, the second solve was refused: its rotated normal "not mesh
coordinates", or its solver "foreign mesh". With and without units.
"""
import pytest
import sympy

import underworld3 as uw
from _units_helpers import model_units

R_IN, R_OUT = 0.5, 1.0


def _first_model():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    x, y = mesh.X
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = (
        model_units(1.0, "Pa*s") * (1 + 20 * (x ** 2 + y ** 2)))
    st.bodyforce = sympy.Matrix([0, -sympy.sin(sympy.pi * x)])
    st.add_dirichlet_bc((0.0, 0.0), "Bottom")
    st.add_dirichlet_bc((0.0, 0.0), "Top")
    st.solve()


def _free_surface_built_but_not_solved():
    """A first model that only builds a FreeSurface (H1's case): its derived
    solves copy a coordinate-dependent viscosity through unwrap."""
    _free_surface_on_a_new_mesh(solve=False)


def _free_surface_on_a_new_mesh(solve=True):
    mesh = uw.meshing.Annulus(radiusInner=R_IN, radiusOuter=R_OUT, cellSize=0.1, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    rhat = sympy.Matrix([[x / r, y / r]])
    T = uw.discretisation.MeshVariable(f"T_{id(mesh)}", mesh, 1, degree=2, continuous=True)
    T.data[:, 0] = 0.5
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        model_units(1.0, "Pa*s") * (1 + 20 * (x ** 2 + y ** 2)))
    stokes.bodyforce = 1.0e3 * T.sym[0] * rhat
    stokes.add_essential_bc((0.0, 0.0), "Lower")
    stokes.tolerance = 1.0e-5
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=5.0e5, normal=rhat,
                                composition=T, tangent_advect="shape", surface_filter=20,
                                smooth_length=0.2, consistent_constraint="strong")
    fs._comp_adv.constitutive_model.Parameters.diffusivity = model_units(1.0, "m**2/s")
    fs._comp_adv.add_dirichlet_bc(1.0, "Lower")
    fs._comp_adv.add_dirichlet_bc(0.0, "Upper")
    if solve:
        fs.solve()


def _references():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), time=uw.quantity(1, "Myr"),
        mass=uw.quantity(1e24, "kg"), temperature=uw.quantity(1000, "K"))


@pytest.mark.level_2
@pytest.mark.tier_b
@pytest.mark.parametrize("first", ["stokes_solved", "free_surface_built"])
@pytest.mark.parametrize("reset", [True, False], ids=["reset_between", "one_model"])
@pytest.mark.parametrize("units", [False, True], ids=["plain", "units"])
def test_a_second_mesh_is_not_handed_the_first_mesh_symbols(units, reset, first):
    """Two meshes' coordinates compare equal by hash and structure, so this does
    not need a reset between the two: two meshes in one model meet it too."""
    try:
        uw.reset_default_model()
        if units:
            _references()
        {"stokes_solved": _first_model,
         "free_surface_built": _free_surface_built_but_not_solved}[first]()
        if reset:
            uw.reset_default_model()
            if units:
                _references()
        _free_surface_on_a_new_mesh()
    finally:
        uw.reset_default_model()
