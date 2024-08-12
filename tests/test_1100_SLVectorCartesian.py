# %%
import underworld3 as uw
import sympy
import math
import pytest

# %% [markdown]
# ### Test Semi-Lagrangian method in advecting vector fields
# ### Scalar field advection together diffusion tested in a different pytest

# %%
# ### Set up variables of the model

# %%
res = 16
nsteps = 2
velocity = 1
dt = 0.1 # do large time steps

# ### mesh coordinates
xmin, xmax = 0., 0.5
ymin, ymax = 0., 1.

sdev = 0.03
x0 = 0.5 * xmax
y0 = 0.3 * ymax

# ### Set up the mesh
### Quads
meshStructuredQuadBox = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)),
    minCoords=(xmin, ymin),
    maxCoords=(xmax, ymax),
    qdegree=3,
)

unstructured_simplex_box_irregular = uw.meshing.UnstructuredSimplexBox(
    cellSize=1 / res, regular=False, qdegree=3, refinement=0
)

unstructured_simplex_box_regular = uw.meshing.UnstructuredSimplexBox(
    cellSize=1 / res, regular=True, qdegree=3, refinement=0
)

@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        unstructured_simplex_box_irregular,
        unstructured_simplex_box_regular,
    ],
)

# test function
def test_SLVec_boxmesh(mesh):
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")

    # Create mesh vars
    Vdeg        = 2
    sl_order    = 2

    v           = uw.discretisation.MeshVariable(r"V", mesh, mesh.dim, degree = Vdeg)
    vec_ana     = uw.discretisation.MeshVariable(r"$V_{ana}$", mesh, mesh.dim, degree = Vdeg)
    vec_tst     = uw.discretisation.MeshVariable(r"$V_{num}$", mesh, mesh.dim, degree = Vdeg)

    # #### Create the SL object
    DuDt = uw.systems.ddt.SemiLagrangian(
                                            mesh,
                                            vec_tst.sym,
                                            v.sym,
                                            vtype = uw.VarType.VECTOR,
                                            degree = Vdeg,
                                            continuous = vec_tst.continuous,
                                            varsymbol = vec_tst.symbol,
                                            verbose = False,
                                            bcs = None,
                                            order = sl_order,
                                            smoothing = 0.0,
                                        )

    # ### Set up:
    # - Velocity field
    # - Initial vector distribution
    with mesh.access(v):
        v.data[:, 1] = velocity

    # distance field will travel
    dist_travel = velocity * dt * nsteps

    # use rigid body vortex with a Gaussian envelope 
    x,y = mesh.X    
    gauss_fn = lambda alpha, xC, yC : sympy.exp(-alpha * ((x - xC)**2 + (y - yC)**2 + 0.000001)) # Gaussian envelope
    with mesh.access(vec_tst, vec_ana):
        vec_tst.data[:, :] = uw.function.evaluate(sympy.Matrix([-gauss_fn(33, x0, y0) * (y - y0), 
                                                                gauss_fn(33, x0, y0) * (x - x0)]), 
                                                 vec_tst.coords)
        vec_ana.data[:, :] = uw.function.evaluate(sympy.Matrix([-gauss_fn(33, x0, y0 + dist_travel) * (y - (y0 + dist_travel)), 
                                                                gauss_fn(33, x0, y0 + dist_travel) * (x - x0)]), 
                                                 vec_ana.coords) 


    for i in range(nsteps):
        DuDt.update_pre_solve(dt, verbose = False, evalf = False)
        with mesh.access(vec_tst): # update
            vec_tst.data[...] = DuDt.psi_star[0].data[...]

    ### compare UW3 and analytical solution
    min_dom = 0.1 * x0 + dist_travel
    max_dom = x0 + dist_travel + 0.9 * x0

    x,y = mesh.X

    mask_fn = sympy.Piecewise((1, (x > min_dom) &  (x < max_dom)), (0., True))

    # sympy functions corresponding to integrals
    vec_diff        = vec_tst.sym - vec_ana.sym
    vec_diff_mag    = vec_diff.dot(vec_diff)

    vec_ana_mag     = vec_ana.sym.dot(vec_ana.sym)

    vec_diff_mag_integ  = uw.maths.Integral(mesh, mask_fn * vec_diff_mag).evaluate()
    vec_ana_mag_integ   = uw.maths.Integral(mesh, mask_fn * vec_ana_mag).evaluate()
    vec_norm            = math.sqrt(vec_diff_mag_integ / vec_ana_mag_integ)
 
    # assume second order relationship between relative norm and mesh resolution 
    # (relative norm = K * min_radius**2)
    assert (math.log10(vec_norm) / math.log10(mesh.get_min_radius())) <= 2 # right hand side is 2 + (log10(k)/log10(delta_x)) so it's at least 2    
    
    del mesh
    del DuDt

del meshStructuredQuadBox
del unstructured_simplex_box_irregular
del unstructured_simplex_box_regular


