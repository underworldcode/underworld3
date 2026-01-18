# %%
import underworld3 as uw
import sympy
import math
import pytest

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before each test to prevent pollution from other tests."""
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


# %% [markdown]
# ### Test Semi-Lagrangian method in advecting vector fields
# ### Scalar field advection together diffusion tested in a different pytest

# %%
# ### Set up variables of the model

# %%
res = 16
nsteps = 2
velocity = 1
dt = 0.1  # do large time steps

# ### mesh coordinates
xmin, xmax = 0.0, 0.5
ymin, ymax = 0.0, 1.0

sdev = 0.03
x0 = 0.5 * xmax
y0 = 0.3 * ymax

# ### Set up the mesh
# NOTE: Meshes MUST be created inside fixtures or tests, not at module level!
# Module-level creation happens at import time, before fixtures reset state,
# which can cause "model has no reference quantities" errors if another test
# has polluted global state.

def create_mesh(mesh_type):
    """Factory function to create meshes inside tests for proper isolation."""
    if mesh_type == "mesh0":
        return uw.meshing.StructuredQuadBox(
            elementRes=(int(res), int(res)),
            minCoords=(xmin, ymin),
            maxCoords=(xmax, ymax),
            qdegree=3,
        )
    elif mesh_type == "mesh1":
        return uw.meshing.UnstructuredSimplexBox(
            cellSize=1 / res, regular=False, qdegree=3, refinement=0
        )
    elif mesh_type == "mesh2":
        return uw.meshing.UnstructuredSimplexBox(
            cellSize=1 / res, regular=True, qdegree=3, refinement=0
        )
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")


@pytest.mark.parametrize("mesh_type", ["mesh0", "mesh1", "mesh2"])
def test_SLVec_boxmesh(mesh_type):
    """Test Semi-Lagrangian vector field advection."""
    # Create mesh INSIDE test function to ensure proper isolation
    mesh = create_mesh(mesh_type)
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")

    # Create mesh vars
    Vdeg = 2
    sl_order = 2

    v = uw.discretisation.MeshVariable(r"V", mesh, mesh.dim, degree=Vdeg)
    vec_ana = uw.discretisation.MeshVariable(r"$V_{ana}$", mesh, mesh.dim, degree=Vdeg)
    vec_tst = uw.discretisation.MeshVariable(r"$V_{num}$", mesh, mesh.dim, degree=Vdeg)

    # #### Create the SL object
    DuDt = uw.systems.ddt.SemiLagrangian(
        mesh,
        vec_tst.sym,
        v.sym,
        vtype=uw.VarType.VECTOR,
        degree=Vdeg,
        continuous=vec_tst.continuous,
        varsymbol=vec_tst.symbol,
        verbose=False,
        bcs=None,
        order=sl_order,
        smoothing=0.0,
    )

    # ### Set up:
    # - Velocity field
    # - Initial vector distribution
    # TODO: DELETE remove swarm.access / data, replace with direct array assignment
    # with mesh.access(v):
    #     v.data[:, 1] = velocity

    v.array[:, 0, 1] = velocity

    # distance field will travel
    dist_travel = velocity * dt * nsteps

    # use rigid body vortex with a Gaussian envelope
    x, y = mesh.X
    gauss_fn = lambda alpha, xC, yC: sympy.exp(
        -alpha * ((x - xC) ** 2 + (y - yC) ** 2 + 0.000001)
    )  # Gaussian envelope

    vec_tst.array = uw.function.evaluate(
        sympy.Matrix(
            [
                -gauss_fn(33, x0, y0) * (y - y0),
                gauss_fn(33, x0, y0) * (x - x0),
            ]
        ).T,
        vec_tst.coords,
    )
    vec_ana.array = uw.function.evaluate(
        sympy.Matrix(
            [
                -gauss_fn(33, x0, y0 + dist_travel) * (y - (y0 + dist_travel)),
                gauss_fn(33, x0, y0 + dist_travel) * (x - x0),
            ]
        ).T,
        vec_ana.coords,
    )

    for i in range(nsteps):
        DuDt.update_pre_solve(dt, verbose=False, evalf=False)
        vec_tst.array[...] = DuDt.psi_star[0].array[...]

    ### compare UW3 and analytical solution
    min_dom = 0.1 * x0 + dist_travel
    max_dom = x0 + dist_travel + 0.9 * x0

    x, y = mesh.X

    mask_fn = sympy.Piecewise((1, (x > min_dom) & (x < max_dom)), (0.0, True))

    # sympy functions corresponding to integrals
    vec_diff = vec_tst.sym - vec_ana.sym
    vec_diff_mag = vec_diff.dot(vec_diff)

    vec_ana_mag = vec_ana.sym.dot(vec_ana.sym)

    vec_diff_mag_integ = uw.maths.Integral(mesh, mask_fn * vec_diff_mag).evaluate()
    vec_ana_mag_integ = uw.maths.Integral(mesh, mask_fn * vec_ana_mag).evaluate()
    vec_norm = math.sqrt(vec_diff_mag_integ / vec_ana_mag_integ)

    # assume second order relationship between relative norm and mesh resolution
    # (relative norm = K * min_radius**2)
    assert (
        math.log10(vec_norm) / math.log10(mesh.get_min_radius())
    ) <= 2  # right hand side is 2 + (log10(k)/log10(delta_x)) so it's at least 2

    del mesh
    del DuDt


# Meshes are now created inside test functions, no need to delete them here
