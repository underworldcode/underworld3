# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# %% [markdown]
# # Structural Optimisation Tests
#
# ## 1 - Shape Recovery (Stokes flow)
#
# Set up a Stokes flow with obstructions, solve and then try to recover obstructions
#

# %%
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import sympy


# %%
width = 4.0
height = 1.0
resolution = 14

csize = 1.0 / resolution
csize_circle = 0.66 * csize
res = csize_circle

width = 4.0
height = 1.0

rows = 2
columns = int((width-1)*rows)
radius_0 = 0.1
variation = 0.25

U0 = 1.0

write_file = True



# %%
obs_penalty = uw.function.expression(r"\beta_\textrm{max}", 100000, r"Obstruction penalty factor")


# %%
## Pure gmsh version

if write_file:

    import pygmsh
    from enum import Enum
    
    ## NOTE: stop using pygmsh, then we can just define boundary labels ourselves and not second guess pygmsh
    
    class boundaries(Enum):
        bottom = 1
        right = 2
        left  = 3
        top = 4
        inclusion = 5
        All_Boundaries = 1001 
    
    # Mesh a 2D pipe with circular holes
    
    ## Restore inflow samples to inflow points
    def pipemesh_return_coords_to_bounds(coords):
        lefty_troublemakers = coords[:, 0] < 0.0
        coords[lefty_troublemakers, 0] = 0.0001
    
        return coords
    
    if uw.mpi.rank == 0:
        import gmsh
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("Domain")
        
        inclusions = []
        inclusion_curves = []
    
        # Repeatable random numbers 
        rrand = np.random.default_rng(66667)
        
        dy = 1.0/(rows+0.5)
        dx = dy*1.2
        
        for row in range(0,rows):
            for col in range(0,columns):
        
                y = dy*(row+0.75) 
                x = 0.25 + dx * col + ( row%2 ) * 0.5 * dx
                r = radius_0  +  variation * (rrand.random()-0.5)
        
                i_points = [
                    gmsh.model.occ.add_point(x,y,0.0,   meshSize=csize_circle),
                    gmsh.model.occ.add_point(x,y+r,0.0, meshSize=csize_circle),
                    gmsh.model.occ.add_point(x-r,y,0.0, meshSize=csize_circle),
                    gmsh.model.occ.add_point(x,y-r,0.0, meshSize=csize_circle),
                    gmsh.model.occ.add_point(x+r,y,0.0, meshSize=csize_circle)
                ]
                
                i_quarter_circles = [
                    gmsh.model.occ.add_circle_arc(i_points[1], i_points[0], i_points[2]),
                    gmsh.model.occ.add_circle_arc(i_points[2], i_points[0], i_points[3]),
                    gmsh.model.occ.add_circle_arc(i_points[3], i_points[0], i_points[4]),
                    gmsh.model.occ.add_circle_arc(i_points[4], i_points[0], i_points[1]),
                ]
               
                inclusion_loop = gmsh.model.occ.add_curve_loop(i_quarter_circles)
                inclusion = gmsh.model.occ.add_plane_surface([inclusion_loop])            
        
                inclusions.append((2,inclusion))
                inclusion_curves.append(i_quarter_circles[0])
                inclusion_curves.append(i_quarter_circles[1])
                inclusion_curves.append(i_quarter_circles[2])
                inclusion_curves.append(i_quarter_circles[3])
        
                gmsh.model.occ.synchronize()
        
        corner_points = []
        corner_points.append(gmsh.model.occ.add_point(0.0, 0.0, 0.0,  csize))
        corner_points.append(gmsh.model.occ.add_point(width, 0.0, 0.0, csize))
        corner_points.append(gmsh.model.occ.add_point(width, 1.0, 0.0,  csize))
        corner_points.append(gmsh.model.occ.add_point(0.0, 1.0, 0.0, csize))
        
        bottom = gmsh.model.occ.add_line(corner_points[0], corner_points[1])
        right = gmsh.model.occ.add_line(corner_points[1], corner_points[2])
        top = gmsh.model.occ.add_line(corner_points[2], corner_points[3])
        left =  gmsh.model.occ.add_line(corner_points[3], corner_points[0])
        
        # gmsh.model.occ.synchronize()
        
        domain_loop = gmsh.model.occ.add_curve_loop((bottom, right, top, left))
        gmsh.model.occ.add_plane_surface([domain_loop])
        
        gmsh.model.occ.synchronize()
        
        # The ordering of the boundaries is scrambled in the 
        # occ.cut stage, save the bb and match the boundaries afterwards.
        
        brtl_bboxes = [ 
                   gmsh.model.get_bounding_box(1,bottom),
                   gmsh.model.get_bounding_box(1,right),
                   gmsh.model.get_bounding_box(1,top),
                   gmsh.model.get_bounding_box(1,left) 
                ]
        
        brtl_indices = [bottom, right, top, left]
         
        domain_cut, index = gmsh.model.occ.cut([(2,domain_loop)], inclusions)
        domain = domain_cut[0]
        gmsh.model.occ.synchronize()
    
        ## There is surely a better way !
      
        brtl_indices = [bottom, right, top, left]
        brtl_map = [
            brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,bottom)), 
            brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,right)),
            brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,top)), 
            brtl_bboxes.index(gmsh.model.occ.get_bounding_box(1,left))
        ]
        
        new_bottom = brtl_indices[brtl_map.index(0)]
        new_right  = brtl_indices[brtl_map.index(1)]
        new_top    = brtl_indices[brtl_map.index(2)]
        new_left   = brtl_indices[brtl_map.index(3)]
          
        gmsh.model.addPhysicalGroup(1, [new_bottom], boundaries.bottom.value, name=boundaries.bottom.name)
        gmsh.model.addPhysicalGroup(1, [new_right], boundaries.right.value, name=boundaries.right.name)
        gmsh.model.addPhysicalGroup(1, [new_top], boundaries.top.value, name=boundaries.top.name)
        gmsh.model.addPhysicalGroup(1, [new_left], boundaries.left.value, name=boundaries.left.name)
        gmsh.model.addPhysicalGroup(1, inclusion_curves, boundaries.inclusion.value, name=boundaries.inclusion.name)
        gmsh.model.addPhysicalGroup(2, [domain[1]], 666666, "Elements")
        
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(dim=2)
        gmsh.write(f".meshes/ns_pipe_flow_{resolution}.msh")
        gmsh.finalize()
    
    pipemesh = uw.discretisation.Mesh(
        f".meshes/ns_pipe_flow_{resolution}.msh",
        markVertices=True,
        useMultipleTags=True,
        useRegions=True,
        refinement=0,
        refinement_callback=None,
        return_coords_to_bounds= pipemesh_return_coords_to_bounds,
        boundaries=boundaries,
        qdegree=3)
    
    pipemesh.dm.view()
    
    # Some useful coordinate stuff
    
    x = pipemesh.N.x
    y = pipemesh.N.y


# %%
# def deferred_derivative(expr, diff_variable):

#     from underworld3.function.expressions import UWDerivativeExpression as _derivative_expression
#     from underworld3.function.expressions import UWexpression as _expression
#     import sympy

#     latex_expr = sympy.latex(expr)
#     latex_diff_variable = sympy.latex(diff_variable)
#     latex = (
#         r"\partial \left[" + latex_expr + r"\right] / \partial " + latex_diff_variable
#     )

#     if isinstance(diff_variable, _expression):
#         diff_variable = diff_variable.sym

#     try:
#         rows, cols = sympy.Matrix(diff_variable).shape
#     except TypeError:
#         rows, cols = (1,1)

#     # Return expression if scalars 
#     if rows==1 and cols==1:
#         # ddx = sympy.Matrix((_derivative_expression(latex, expr, diff_variable)))
#         ddx = _derivative_expression(latex, expr, diff_variable)
#     else:
#         ddx = sympy.Matrix.zeros(rows=rows, cols=cols)
#         for i in range(rows):
#             for j in range(cols):
#                 latex = (
#                     r"\partial \left["
#                     + sympy.latex(expr)
#                     + r"\right] / \partial "
#                     + sympy.latex(diff_variable[i, j])
#                 )
#                 ddx[i, j] = _derivative_expression(latex, expr, diff_variable[i, j])
#     return ddx


# %%
dX = uw.function.deferred_derivative(pipemesh.CoordinateSystem.X[0], pipemesh.CoordinateSystem.X[0])

# %%
uw.function.derivative(pipemesh.CoordinateSystem.X, pipemesh.CoordinateSystem.X, evaluate=False).doit()

# %%

# %%

# %%
dX

# %%
dX2 = pipemesh.CoordinateSystem.X.diff(pipemesh.CoordinateSystem.X, evaluate=False)

# %%
dX2

# %%
V = uw.discretisation.MeshVariable("VV", pipemesh, pipemesh.dim, degree=2)

# %%
V.view()

# %%
dVdX = uw.function.deferred_derivative(V.sym, pipemesh.CoordinateSystem.X)
dVdX

# %%
dVdV = uw.function.deferred_derivative(V.sym, V.sym)
dVdV[0,0].sym

# %%
J = uw.function.expression("J", V.sym +  pipemesh.CoordinateSystem.X, "Residual")

# %%
Jx = J.diff(pipemesh.CoordinateSystem.X[0], evaluate=False)

# %%
type(dVdX)

# %%
dJdX = uw.function.deferred_derivative(J, pipemesh.CoordinateSystem.X)
dJdX[0,0].sym

# %%
J.sym = V.sym.dot(V.sym) * sympy.Matrix([[1,1]]) +  pipemesh.CoordinateSystem.X
dJdX

# %%
0/0

# %%
if write_file:

    v_soln = uw.discretisation.MeshVariable("V0", pipemesh, pipemesh.dim, degree=2)
    p_soln = uw.discretisation.MeshVariable("P0", pipemesh, 1, degree=1, continuous=True)
    
    # Set solve options here (or remove default values
    # stokes.petsc_options.getAll()
    
    stokes = uw.systems.Stokes(
        pipemesh,
        velocityField=v_soln,
        pressureField=p_soln,
        verbose=False)
    
    stokes.petsc_options["snes_monitor"] = None
    stokes.petsc_options["ksp_monitor"] = None
    
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
    
    stokes.tolerance = 0.00001
    
    
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    
    # Constant visc
    
    stokes.penalty = 1
    stokes.bodyforce = sympy.Matrix([0, 0]).T
    
    
    # Velocity boundary conditions
    
    stokes.add_dirichlet_bc(
        (0.0, 0.0),
        "inclusion")
    
    # Gamma = pipemesh.Gamma
    # GammaNorm = uw.function.expression(r"|\Gamma|", sympy.sqrt(Gamma.dot(Gamma)), "Scaling for surface normals")
    # GammaN = Gamma / GammaNorm
    # stokes.add_natural_bc(100000 * v_soln.sym.dot(GammaN) * GammaN, "inclusion")
    
    stokes.add_dirichlet_bc((0.0, 0.0), "top")
    stokes.add_dirichlet_bc((0.0, 0.0), "bottom")
    stokes.add_dirichlet_bc((U0, 0.0), "left")

    stokes.view()


# %%
if write_file:
    stokes.solve(zero_init_guess=True)

    pipemesh.write_timestep("TargetSolution", 
                        index=0,
                        outputPath=".",
                        meshVars=[v_soln, p_soln]
                )

# %%
# check the mesh if in a notebook / serial

if write_file and uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", 
        surface_streamlines=True, max_steps=100
    )



    pl = pv.Plotter(window_size=(1500, 400))

    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["V"], 
                  mag=0.01 / U0, opacity=0.25, show_scalar_bar=False)


    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False)
    
    pl.add_mesh(pvstream)


    
    pl.camera.position = (2.0, 0.5, 3)
    pl.camera.focal_point=(2.0,0.5,0.0)

    pl.show(jupyter_backend="html")

# %% [markdown]
# ## Structural Optimisation
#
# In structural optimisation, the geometry of the domain is the subject of an optimisation problem. In a finite element model, that might entail moving boundary nodes or internally meshed interfaces in order to fin a local minimimum of some objective function. The topology is fixed in the sense that the shape of the boundaries might change but not their number or their connectivity. 
#
# References (see below)
#
# ## Topological Optimisation
#
# Topological optimisation is a form of structural optimisation that attempts to relax the need to preserve the topology of the interfaces that are deformed during the optimisation phase. This can be done by penalising the solution in parts of the domain that are masked out by the movement of the interface, retaining them in the problem to allow them to be considered as unknowns if later steps in the optimisation procedure require it. 
#
# In a Stokes-flow problem, for example, we add a force term:
#
# $$f_\textrm{penalty} = -\lambda \beta(\mathbf{x}) \mathbf{v} $$
#
# where $\lambda$ is a large, positive penalty factor, $\beta$ is a scalar field that takes the value 0 (points lie within the current computational domain) or 1 (points are excluded, velocity solution is zero), and $\mathbf{v}$ is the unknown flow velocity. Topological optimisation evolves the $\beta$ field by determining a pseudo-velocity of its interface which reduces the objective function of the optimisation. 
#
# ```python
#     stokes.bodyforce =  sympy.Matrix(-100000 * obstruction_function * v_soln1.sym)
# ```

# %%
## Equivalent mesh 

openmesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), maxCoords=(width, 1.0), qdegree=3, cellSize=0.05)

v_phi =  uw.discretisation.MeshVariable("V_phi",openmesh, openmesh.dim, degree=2, varsymbol=r"V_\phi")
v_solno =  uw.discretisation.MeshVariable("Vo",openmesh, openmesh.dim, degree=2)
v_soln1 = uw.discretisation.MeshVariable("V1", openmesh, openmesh.dim, degree=2)
p_soln1 = uw.discretisation.MeshVariable("P1", openmesh, 1, degree=1, continuous=True)
u_soln1 = uw.discretisation.MeshVariable("U1", openmesh, openmesh.dim, degree=2)
q_soln1 = uw.discretisation.MeshVariable("Q1", openmesh, 1, degree=1, continuous=True)

obstruction = uw.discretisation.MeshVariable("Beta", openmesh, 1, degree=2, continuous=True, varsymbol=r"\beta")
deltaBeta = uw.discretisation.MeshVariable("dBeta", openmesh, 1, degree=2, continuous=True, varsymbol=r"\delta\beta")

obs_penalty = uw.function.expression(r"\beta_\textrm{max}", 100000, r"Obstruction penalty factor")

v_solno.read_timestep("TargetSolution", "V0", 0, outputPath=".", verbose=True)


# %%
obs_penalty.view()

# %%
## This is the forward problem

stokes_forward = uw.systems.Stokes(
    openmesh,
    velocityField=v_soln1,
    pressureField=p_soln1,
    verbose=False)

xo, yo = openmesh.X

theta = sympy.pi / 6
xr = xo * sympy.cos(theta) + yo * sympy.sin(theta)
yr = xo * sympy.sin(theta) - yo * sympy.cos(theta)

stokes_forward.petsc_options["snes_monitor"] = None
stokes_forward.petsc_options["ksp_monitor"] = None
stokes_forward.tolerance = 0.0001
stokes_forward.penalty = 1

# Options to improve solution speed
stokes_forward.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes_forward.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes_forward.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes_forward.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes_forward.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes_forward.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes_forward.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
stokes_forward.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

stokes_forward.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Velocity boundary conditions
stokes_forward.add_dirichlet_bc((0.0, 0.0), "Top")
stokes_forward.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes_forward.add_dirichlet_bc((U0, 0.0), "Left")

## Viscosity is a step function (differentiable approximation) obtained from the obstruction field
## This produces low strain rates in the obstruction zone (they are also penalized to give low
## velocity values

obstruction_function = (1 + sympy.tanh(10*obstruction.sym[0])) / 2
stokes_forward.constitutive_model.Parameters.shear_viscosity_0 = 1 + 99 * obstruction_function

## We add the penalty to the solution where the obstruction exists

stokes_forward.bodyforce.sym = obs_penalty * obstruction_function * v_soln1.sym

# %% [markdown]
# ## Check the forward problem
#
# We can check that the forward problem has the correct structure for both the penalty 
# (blockage / obstruction) term, and the viscosity by asking the solver to 
# describe how it is set up with `stokes_forward.view()`

# %%
stokes_forward.view()

# %% [markdown]
# ## Define an objective function
#
# Assuming the forward problem has been defined and the topological penalty term added (above), we next need to write the objective function in terms of the unknowns of the forward problem, find its derivative, and then construct the adjoint problem. From the forward and adjoint solutions, we can construct an interface-update pseudo-velocity.
#
# In this problem, we minimize the fit between a velocity-field previously calculated on a complicated mesh to the solution of a problem where the mesh complexity is the target of the optimisation:
#
# $$ J_1 = \frac{1}{2} (\mathbf{v} - \mathbf{v_0})\cdot (\mathbf{v} - \mathbf{v_0})$$
#
# an alternative cost function is strongly weighted towards regions with low velocity
#
# $$ J_2 = \frac{1}{2} (\mathbf{v} - \mathbf{v_0})\cdot (\mathbf{v} - \mathbf{v_0}) / 
#                      (\mathbf{v_0} \cdot \mathbf{v_0})
# $$
#
# We can use both if we think that will help (they may need a relative weighting).

# %%
## Optimisation parameter

j1 = (v_soln1.sym - v_solno.sym).dot((v_soln1.sym - v_solno.sym)) / 2
j2 = (v_soln1.sym - v_solno.sym).dot((v_soln1.sym - v_solno.sym)) / (v_solno.sym.dot(v_solno.sym) + epsilon)
j3 = sympy.log((v_soln1.sym).dot((v_soln1.sym) / (v_solno.sym.dot(v_solno.sym) + epsilon)))**2

Wj = uw.function.expression(r"w_j", sympy.sympify(0)/10 , "weighting")

J1 = uw.function.expression(r"\mathcal{J}_1", j1 , "Objective Function - part 1")
J2 = uw.function.expression(r"\mathcal{J}_2", j2 , "Objective Function - part 2")
J = uw.function.expression(r"\mathcal{J}", J1 + Wj * J2 , "Objective Function")


# %% [markdown]
# ## Compute the objective funtion and its derivatives (adjoint RHS terms)
#
# The adjoint of the forward problem is used to compute the changes to the forward problem that will reduce the error in the misfit or cost function, $J$. Solving the adjoint problem ensures that the path we take to reduce the misfit is always constrained to follow valid solutions to the forward problem. 
#
# The right hand side of the adjoint equation comes from the derivative of $J$ the cost function with respect to the unknowns of the forward problem and thus can be computed symbolically from $J$. In this case, $J$ only depends on the velocity unknown. 

# %%
# This is the force term for the adjoint velocity equation
dJ_dv = uw.function.deferred_derivative(J, v_soln1.sym)

# This is the force term for the adjoint pressure / constraint equation
dJ_dp = uw.function.deferred_derivative(J, p_soln1.sym)

# %%
uw.function.deferred_derivative(J1, v_soln1.sym)[0,0].sym

# %%
uw.function.deferred_derivative(J2, v_soln1.sym)[0,0].sym

# %%
dJ_dv

# %%
Wj.sym=0
dJ_dv[0,0].sym

# %%
Wj.sym=0
dJ_dv[0,0].sym

# %% [markdown]
# ## Construct the Adjoint Equations
#
# The adjoint problem is closely related to the forward problem and can be constructed in various ways. It is possible to auto-differentiate the code for the forward problem, it is also possible to symbolically construct the adjoint from symbolic representation of the forward problem, or one can construct by inspection (as is usually documented in publications). Here we follow the latter path, but we try to avoid double-handling of expressions that appear in both forward and adjoint equations to avoid introducing opportunities to generate bugs.
#
# The Stokes adjoint is identical to the Stokes forward equation unless the constitutive properties depend directly on the velocity unknowns of the forward problem (and not their gradients).  This does not occur in the Cartesian formulation.
#
# We do the following:
#
#   - Construct a clone of the Stokes solver
#   - Substitute the adjoint variables for the forward variables in the force term
#   - Add the velocity-data misfit term $\partial J / \partial \mathbf{v}$ on the RHS of the momentum equation
#   - Add the pressure-data misfit term $\partial J / \partial \mathbf{p}$ on the RHS of the continuity equation
#   - Adjoint Boundary conditions - most boundary condition terms are zero but some are complicated to avoid violating assumptions of independence between the forward and adjoint variables.
#
# In our sample problem, having cloned the solver itself,
#
#   - we clone the constitituve model
#   - clone the body force term, substitute the forward unknowns with the adjoint unknowns.
#   - introduce the $\partial J / \partial \mathbf{v}$ on the RHS
#   - set boundary conditions
#

# %%
## The adjoint problem
## Stokes with a penalty on velocity is self-adjoint ... 

## This is the adjoint problem

stokes_adjoint = uw.systems.Stokes(
    openmesh,
    velocityField=u_soln1,
    pressureField=q_soln1,
    verbose=False)

stokes_adjoint.petsc_options["snes_monitor"] = None
stokes_adjoint.petsc_options["ksp_monitor"] = None
stokes_adjoint.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes_adjoint.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes_adjoint.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes_adjoint.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes_adjoint.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes_adjoint.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes_adjoint.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
stokes_adjoint.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

stokes_adjoint.tolerance = 0.001

stokes_adjoint.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_adjoint.constitutive_model.Parameters.shear_viscosity_0 = (
    stokes_forward.constitutive_model.Parameters.shear_viscosity_0.sym
)

# Constant visc

stokes_adjoint.penalty = 1
stokes_adjoint.bodyforce = stokes_forward.bodyforce.sym.subs({v_soln1.sym[0]: u_soln1.sym[0], 
                                                              v_soln1.sym[1]: u_soln1.sym[1], 
                                                              p_soln1.sym[0]: q_soln1.sym[0]}) + dJ_dv

# Velocity boundary conditions ??

stokes_adjoint.add_dirichlet_bc((0.0, 0.0), "Top")
stokes_adjoint.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes_adjoint.add_dirichlet_bc((0.0, 0.0), "Left")
stokes_adjoint.add_dirichlet_bc((0.0, 0.0), "Right")


# %%
stokes_adjoint.bodyforce.sym

# %%
uw.function.fn_unwrap(stokes_adjoint.bodyforce.sym[0])

# %% [markdown]
# ## Check the adjoint solver is properly constructed
#
# We should check that the form of the solver is correct, that the adjoint variables have been appropriately subsituted into the right hand side penalty term, and that the derivative of the objective function is added to this term correctly.
#
# The entire equation system can be seen using
# ```python
#     stokes_adjoint.view()
# ```
# which reports the equation structure (with simplification and combination of terms). Sub-expressions such as the viscosity are left as symbols but their expansion is also reported. Boundary conditions are listed in a table. 

# %%
stokes_adjoint.view()

# %%
# Set up initial distribution of the obstruction parameter

with openmesh.access(obstruction):
    obstruction.data[:,0] = -1 + 2 * uw.function.evalf(sympy.sin(5*xr*sympy.pi)**2 * sympy.sin(5*yr * sympy.pi)**2, obstruction.coords)

stokes_forward.solve()
stokes_adjoint.solve()

# %%
## We use this for the velocity penalty
VPen = obs_penalty * (v_soln1.sym.dot(u_soln1.sym) * uw.function.derivative(obstruction_function, obstruction.sym[0]))

# We use this for the strain rate

dSigma_dphi = uw.function.derivative(stokes_forward.F1, obstruction.sym[0])
E_adjoint = stokes_adjoint.Unknowns.E
StrRatePen = uw.maths.tensor.rank2_inner_product(E_adjoint, dSigma_dphi)

compute_delta = uw.systems.Projection(openmesh, deltaBeta)
compute_delta.uw_function = StrRatePen + VPen
compute_delta.smoothing = 1.0e-2

# compute_delta.add_essential_bc((0), "Top")
# compute_delta.add_essential_bc((0), "Bottom")
# compute_delta.add_essential_bc((0), "Left")
# compute_delta.add_essential_bc((0), "Right")

compute_delta.solve()

# %%
phi_gradient =  openmesh.vector.gradient(obstruction.sym[0])
unit_phi_gradient = phi_gradient / (epsilon + sympy.sqrt(phi_gradient.dot(phi_gradient)))
unit_phi_gradient

# %%
compute_v_phi = uw.systems.Vector_Projection(openmesh, v_phi)
compute_v_phi.uw_function =  unit_phi_gradient
compute_v_phi.smoothing = 1.0e-2
compute_v_phi.solve()

with openmesh.access(v_phi): 
        v_phi.data[:,:] = v_phi.data[:,:] / np.hypot(v_phi.data[:,0], v_phi.data[:,1]).reshape(-1,1)
        v_phi.data[:,:] = v_phi.data[:,:] * deltaBeta.data[:]


# %%
field_advection = uw.systems.AdvDiffusion(openmesh, u_Field=obstruction, V_fn=v_phi, order=1)
field_advection.constitutive_model = uw.constitutive_models.DiffusionModel
field_advection.constitutive_model.Parameters.diffusivity = 1.0
field_advection.estimate_dt()

# %%
compute_misfit = uw.maths.Integral(openmesh, J)
misfit = compute_misfit.evaluate()
misfit

# %%
stokes_adjoint.view()

# %%
J.unwrap()

# %%
tstep = 0

# %%

# %%
# Evolution loop

for step in range(1,3):

    # We first focus on the stagnant regions, then bias the system towards the 
    # Note, derivatives of expressions are not lazy (shame !) so we need to recompute them
    
    if step%10 == 0:
        Wj.sym /= 5
        dJ_dv = uw.function.derivative(J, v_soln1.sym)
        stokes_adjoint.bodyforce = stokes_forward.bodyforce.sym.subs({v_soln1.sym[0]: u_soln1.sym[0], 
                                                              v_soln1.sym[1]: u_soln1.sym[1], 
                                                              p_soln1.sym[0]: q_soln1.sym[0]}) + dJ_dv
  
    #1 Update obstruction field using advection heuristic
    timestep = field_advection.estimate_dt()
    # print(f"timestep {timestep}")
    field_advection.solve(zero_init_guess=True, timestep=timestep)

    with openmesh.access(obstruction, deltaBeta):
        obstruction.data[:,0] *= 1.0001 # steepen slightly
        obstruction.data[:,0] = np.clip(obstruction.data[:,0], -1.0, 1.0)
    
    #2 Solve forward problem
    stokes_forward.solve(zero_init_guess=False, picard=0, verbose=False)
    
    #3 Solve adjoint problem (there should be no good guess for this, but under-relaxation ...)
    stokes_adjoint.solve(zero_init_guess=False, picard=0, verbose=False)

    #4 Compute gradients of obstruction field
    with openmesh.access(v_phi): 
        dB = deltaBeta.data[:].copy()
        
    compute_delta.solve(verbose=False)
    compute_v_phi.solve()
    
    with openmesh.access(v_phi): 
        v_phi.data[:,:] = v_phi.data[:,:] / np.hypot(v_phi.data[:,0], v_phi.data[:,1]).reshape(-1,1)
        v_phi.data[:,:] = v_phi.data[:,:] * (deltaBeta.data[:] + dB[:])/2

    misfit = compute_misfit.evaluate()
    if uw.mpi.rank == 0:
        print(f"{tstep}: Misfit: {misfit}")

    ## Save the data

    openmesh.write_timestep(
        "SOpt_6",
        meshUpdates=True,
        meshVars=[p_soln1, v_soln1, obstruction, v_phi, deltaBeta],
        outputPath="output",
        index=tstep)


    ## Keep track

    tstep += 1


# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:


    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(openmesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln1.sym)
    pvmesh.point_data["Vo"] = vis.vector_fn_to_pv_points(pvmesh, v_solno.sym)
    pvmesh.point_data["dV"] = pvmesh.point_data["Vo"] - pvmesh.point_data["V"] 

    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln1.sym.dot(v_soln1.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln1.sym)


    viscosity = stokes_forward.constitutive_model.Parameters.shear_viscosity_0
    pvmesh.point_data["Visc"] = vis.scalar_fn_to_pv_points(pvmesh, viscosity)
    
    pvmesh.point_data["Beta"] = vis.scalar_fn_to_pv_points(pvmesh, obstruction_function)
    pvmesh.point_data["dBeta"] = vis.scalar_fn_to_pv_points(pvmesh, obstruction.sym[0] - (-1 + 2 *sympy.sin(2*xr*sympy.pi)**2 * sympy.sin(2*yr * sympy.pi)**2))
    pvmesh.point_data["dL"] = vis.scalar_fn_to_pv_points(pvmesh, deltaBeta.sym )

    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln1)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln1.sym)
    velocity_points.point_data["Vo"] = vis.vector_fn_to_pv_points(velocity_points, v_solno.sym)
    velocity_points.point_data["U"] = vis.vector_fn_to_pv_points(velocity_points, u_soln1.sym)
    velocity_points.point_data["Vphi"] = vis.vector_fn_to_pv_points(velocity_points, v_phi.sym)

    # point sources at cell centres
    skip=5
    points = np.zeros((openmesh._centroids[::skip].shape[0], 3))
    points[:, 0] = openmesh._centroids[::skip, 0]
    points[:, 1] = openmesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="Vo", integration_direction="forward", 
        surface_streamlines=True, max_steps=100)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["Vphi"], 
                  mag=1.0e-5 , opacity=1, 
                  color="Red",
                  show_scalar_bar=False)

    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["V"], 
                  mag=5.0e-2, opacity=1, 
                  color="Green",
                  show_scalar_bar=False)
    
    # pl.add_arrows(velocity_points.points, 
    #               velocity_points.point_data["Vo"], 
    #               mag=1.0e-2, opacity=1, 
    #               color="Blue",
    #               show_scalar_bar=False)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Visc",
        use_transparency=False,
        opacity=1,
        # clim=[-1, 1],
        show_scalar_bar=True)

    # pl.add_mesh(
    #     pvmesh,
    #     copy_mesh=True,
    #     cmap="coolwarm",
    #     edge_color="Grey",
    #     show_edges=True,
    #     scalars="dL",
    #     use_transparency=False,
    #     #clim=[-1e6, 1e6],
    #     show_scalar_bar=True,
    #     opacity=0.5,
    # )
    
    # pl.add_mesh(pvstream, show_scalar_bar=False)
  
    pl.camera.position = (2.0, 0.5, 10)
    pl.camera.focal_point=(2.0,0.5,0.0)

    pl.show(jupyter_backend="html")

# %%
timestep

# %%
