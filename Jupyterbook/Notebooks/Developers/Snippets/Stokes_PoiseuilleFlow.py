# %% [markdown]
# # Stokes Poiseuille flow
#
# [UW2 example]( https://github.com/underworldcode/underworld2/blob/v2.15.1b/docs/test/StokesEq_PoiseuilleFlow.ipynb)

# %%
import underworld3 as uw

import numpy as np

import sympy

import os
import sys

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
    
    from matplotlib import rc
    
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':10})
    
    # Set the font used for MathJax
    rc('mathtext',**{'default':'regular'})
    rc('figure',**{'figsize':(8,6)})




# %%
### Create folder to save data
outputPath = './output/Stokes-PoiseuilleFlow/'

if uw.mpi.rank == 0:
    # checking if the directory
    # exist or not.
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

# %%
pa = 4.0
pb = 3.0



xa =   0.
xb =   2.

dp_dx = (pb-pa)/(xb-xa)

h = 1.0


viscosity = 1.


# vdegree  = int(sys.argv[1])
# pdegree  = int(sys.argv[2])

# res = int(sys.argv[3])

# simplex = sys.argv[4].lower()


# filename = str(sys.argv[6])


vdegree  = 2
pdegree  = 1
pcont = False

res = uw.options.getInt("model_resolution", default=64)
refinement = uw.options.getInt("model_refinement", default=0)
simplex = uw.options.getBool("simplex", default=True)
filename = "PoiseuilleTest"




# %%
if simplex == 'true':
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(xa, 0.), maxCoords=(xb, h), cellSize=1/res, qdegree=max(pdegree, vdegree))
else:
    mesh  = uw.meshing.StructuredQuadBox(minCoords=(xa, 0.), maxCoords=(xb, h), elementRes=(int(2*res), res), qdegree=max(pdegree, vdegree))

v= uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=vdegree)


v_a = uw.discretisation.MeshVariable("U_a", mesh, mesh.dim, degree=vdegree)

if pcont == 'true':
    p = uw.discretisation.MeshVariable("P_n", mesh, 1, degree=pdegree, continuous=True)
    p_a = uw.discretisation.MeshVariable("P_a", mesh, 1, degree=pdegree, continuous=True)
else:
    p = uw.discretisation.MeshVariable("P_n", mesh, 1, degree=pdegree, continuous=False)
    p_a = uw.discretisation.MeshVariable("P_a", mesh, 1, degree=pdegree, continuous=False)
    

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

# %%
for solver in [stokes]:
    solver.petsc_options.setValue("snes_monitor", None) 
    solver.petsc_options.setValue("ksp_monitor", None) 
    
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0=viscosity
    

    ### pressure gradient from left to right
    
    # solver.add_natural_bc( sympy.Matrix([1000*(p.sym[0]-pa), 0.0]) , 'Left' )
    # solver.add_natural_bc( sympy.Matrix([1000*(pb-p.sym[0]), 0.0]) , 'Right')
    
    solver.add_natural_bc( sympy.Matrix([-pa, 0.0]) , 'Left' )
    solver.add_natural_bc( sympy.Matrix([ pb, 0.0]) , 'Right')
    
    
    solver.add_dirichlet_bc( [0., 0.], "Left",  [1] )  # no slip on the base
    solver.add_dirichlet_bc( [0., 0.], "Right", [1] )  # no slip on the top
    
    
    solver.add_dirichlet_bc( [0.,0.], "Bottom", [0, 1] )  # no slip on the base
    solver.add_dirichlet_bc( [0.,0.], "Top",    [0, 1] )  # no slip on the top
    
    
    if uw.mpi.size == 1:
        solver.petsc_options['pc_type'] = 'lu'
    #     # solver.petsc_options['ksp_type'] = 'preonly'

    # ### set the tolerance of the solver
    # stokes.petsc_options["snes_rtol"] = 1.0e-6
    # stokes.petsc_options["ksp_rtol"] = 1.0e-6
    
    ### see the SNES output
    solver.petsc_options["snes_converged_reason"] = None
    # solver.petsc_options["snes_monitor_short"] = None
    
    # solver.tolerance = 1e-8
    
    


# %%
solver.solve()

# %%
# check the mesh if in a notebook / serial

if 1 and uw.mpi.size == 1:

    v_soln = v
    p_soln = p

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres
    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.025, opacity=0.75)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_points(
        point_cloud,
        color="Black",
        render_points_as_spheres=False,
        point_size=5,
        opacity=0.66,
    )

    
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1, opacity=0.75)
    pl.add_mesh(pvstream)

    pl.show()


# %%
def L1_norm_integration_p(solver, p_a):
    numeric_solution   = solver.p.sym[0]
    analytic_solution  = p_a.sym[0]

    I = uw.maths.Integral(solver.mesh, sympy.Abs(numeric_solution-analytic_solution))

    return I

def L2_norm_integration_p(solver, p_a):
    numeric_solution   = solver.p.sym[0]
    analytic_solution  = p_a.sym[0]

    I = uw.maths.Integral(solver.mesh, (numeric_solution-analytic_solution)**2)

    L2_p = np.sqrt( I.evaluate() )

    return L2_p

def L1_norm_integration_v(solver, v_a):
    numeric_solution_vx   = solver.u.sym[0]
    analytic_solution_vx  = v_a.sym[0]

    vx_l1_norm = sympy.Abs(numeric_solution_vx - analytic_solution_vx)

    numeric_solution_vy   = solver.u.sym[1]
    analytic_solution_vy  = v_a.sym[1]

    vy_l1_norm = sympy.Abs(numeric_solution_vy - analytic_solution_vy)
    

    I = uw.maths.Integral(solver.mesh, (vx_l1_norm+vy_l1_norm) )

    return I

def L2_norm_integration_v(solver, v_a):
    numeric_solution_vx   = solver.u.sym[0]
    analytic_solution_vx  = v_a.sym[0]

    vx_l2_norm = (numeric_solution_vx - analytic_solution_vx)**2

    vx_L2 = np.sqrt( uw.maths.Integral(solver.mesh, vx_l2_norm ).evaluate() )

    numeric_solution_vy   = solver.u.sym[1]
    analytic_solution_vy  = v_a.sym[1]

    vy_l2_norm = (numeric_solution_vy - analytic_solution_vy)**2

    vy_L2 = np.sqrt( uw.maths.Integral(solver.mesh, vy_l2_norm ).evaluate() )
    

    vmag_L2 = np.sqrt( uw.maths.Integral(solver.mesh, (vx_l2_norm+vy_l2_norm) ).evaluate() )


    return vx_L2, vy_L2, vmag_L2

# %%
with mesh.access(v, p, v_a, p_a):
    v_a.data[:,0] =  1.0 / (2.0 * viscosity)* dp_dx * (v_a.coords[:,1]**2 - h*v_a.coords[:,1])
    v_a.data[:,1] =  0.
    p_a.data[:,0] = pa + (dp_dx*p_a.coords[:,0])

    # cbar = plt.scatter(v.coords[:,0], v.coords[:,1], c=np.abs(v_a.data[:,1] - v.data[:,1]))
    # plt.colorbar(cbar)




# %%
### Create columns of file if it doesn't exist
try:
    with open(f'{outputPath}{filename}', 'x') as f:
        f.write(f'res,P degree,V degree,cell size,L2_norm_p,L2_norm_vx,L2_norm_vy,L2_norm_vmag')
except:
    pass

if stokes.snes.getConvergedReason() > 0:
    L2_norm_vel = L2_norm_integration_v(stokes, v_a)
    L2_norm_p = L2_norm_integration_p(stokes, p_a)
    
else:
    L2_norm_vel = np.nan, np.nan, np.nan
    L2_norm_p = np.nan
    

results = np.column_stack([ res, pdegree, vdegree, stokes.mesh.get_min_radius(), L2_norm_p, L2_norm_vel[0], L2_norm_vel[1], L2_norm_vel[2] ])

### Append the data
with open(f'{outputPath}{filename}', 'a') as f:
    for i, item in enumerate(results.flatten()):
        if i == 0:
            f.write(f'\n{item}')
        else:
            f.write(f',{item}')

# %%

# import matplotlib.pyplot as plt

# n = 5

# plt.figure(figsize=(10, 4))
# plt.title('Pressure [MPa]')
# visc_scatter = plt.scatter(mesh_QB.data[:,0], mesh_QB.data[:,1], c=(uw.function.evalf(p_QB.sym[0], mesh_QB.data)), cmap='RdYlBu_r')
# plt.quiver(mesh_QB._centroids[:,0][::n], mesh_QB._centroids[:,1][::n], uw.function.evalf(vel_QB.sym[0], mesh_QB._centroids)[::n], uw.function.evalf(vel_QB.sym[1], mesh_QB._centroids)[::n])
# plt.colorbar(visc_scatter)

# %%
# with stokes_QB.mesh.access(stokes_QB.u, stokes_QB.p):
#     analytic_P_QB = pa + (dp_dx*stokes_QB.p.coords[:,0])
#     L1_norm_p_QB = np.abs(stokes_QB.p.data[:,0] - analytic_P_QB)

#     analytic_u_QB = 1.0 / (2.0 * viscosity)* dp_dx * (stokes_QB.u.coords[:,1]**2 - h*stokes_QB.u.coords[:,1])
#     L1_norm_u_QB = np.abs(stokes_QB.u.data[:,0] - analytic_u_QB)

# with stokes_SB.mesh.access(stokes_SB.u, stokes_SB.p):
#     analytic_P_SB = pa + (dp_dx*stokes_SB.p.coords[:,0])
#     L1_norm_p_SB = np.abs(stokes_SB.p.data[:,0] - analytic_P_SB)
    
#     analytic_u_SB = 1.0 / (2.0 * viscosity)* dp_dx * (stokes_SB.u.coords[:,1]**2 - h*stokes_SB.u.coords[:,1])
#     L1_norm_u_SB = np.abs(stokes_SB.u.data[:,0] - analytic_u_SB)

# %%
# f, ax = plt.subplots(2, 2,  sharey=True, sharex=True, figsize=(8, 4), width_ratios=[4, 4,])

# with stokes_QB.mesh.access(stokes_QB.u, stokes_QB.p):
#     cbar = ax[0,0].scatter(stokes_QB.p.coords[:,0], stokes_QB.p.coords[:,1], c=L1_norm_p_QB, marker='s', s=1)
#     plt.colorbar(cbar, ax=ax[0,0])

#     cbar = ax[0,1].scatter(stokes_QB.u.coords[:,0], stokes_QB.u.coords[:,1], c=L1_norm_u_QB, marker='s', s=2)
#     plt.colorbar(cbar, ax=ax[0,1])

#     # ax[0,2].plot(stokes_QB.u.coords[:,0], stokes_QB.u.coords[:,1], c=L1_norm_u_QB, marker='s')

#     cbar = ax[1,0].scatter(stokes_SB.p.coords[:,0], stokes_SB.p.coords[:,1], c=L1_norm_p_SB, marker='s', s=3.5)
#     plt.colorbar(cbar, ax=ax[1,0])

#     cbar = ax[1,1].scatter(stokes_SB.u.coords[:,0], stokes_SB.u.coords[:,1], c=L1_norm_u_SB, marker='s', s=1)
#     plt.colorbar(cbar, ax=ax[1,1])

#     ax[0,0].set_title('A) |e$_p$| quad box')

#     ax[0,1].set_title('B) |e$_u$| quad box')

#     ax[1,0].set_title('C) |e$_p$| simplex box')

#     ax[1,1].set_title('D) |e$_u$| simplex box')

#     ax[0,0].set_xlim(0,2)
#     ax[0,0].set_ylim(0,1)

#     plt.savefig('L1-norm-Poiseuille_flow.pdf', bbox_inches='tight'  )
#     plt.savefig('L1-norm-Poiseuille_flow.jpg', bbox_inches='tight'  )


# %%
# profile = np.zeros(shape=(50, 2))
# profile[:,0] = 1.
# profile[:,1] = np.linspace(0.01, 0.99, 50)

# %%
# vel_profile = uw.function.evaluate(vel.sym, profile)

# %%
# def exact_vx(y):
#     ana_u = 1.0 / (2.0 * viscosity)* dp_dx * (y**2 - h*y)
#     return ana_u

# %%
# l1_norm = np.abs((exact_vx(profile[:,1]) - vel_profile[:,0][:,0]))

# %%

# f, (ax1, ax2, ax3) = plt.subplots(1, 3, width_ratios=[1, 1, 3], sharey=True, figsize=(10,4))

# ax1.set_ylim(0,1)

# ax1.set_title('A) Velocity')
# ax1.plot(vel_profile[:,0], profile[:,1], label='numeric v$_x$')
# ax1.scatter(exact_vx(profile[:,1]), profile[:,1], label='analytic v$_x$', marker='x')

# # ax1.scatter(l1_norm, profile[:,1], label='L1-norm')

# ax1.legend()

# ax2.set_title('B) v$_x$ L1-norm')
# ax2.plot(l1_norm, profile[:,1], label='L1-norm')
# ax1.legend()


# ax3.set_title('C) Pressure')
# visc_scatter = ax3.scatter(mesh.data[:,0], mesh.data[:,1], c=(uw.function.evalf(p.sym[0], mesh.data)), cmap='RdYlBu_r')
# ax3.quiver(mesh._centroids[:,0][::n], mesh._centroids[:,1][::n], uw.function.evalf(vel.sym[0], mesh._centroids)[::n], uw.function.evalf(vel.sym[1], mesh._centroids)[::n])
# plt.colorbar(visc_scatter, ax=ax3)

# ax3.set_xlim(0,2)

# plt.savefig('PoiseuilleFlow_benchmark.pdf')
# plt.savefig('PoiseuilleFlow_benchmark.jpg')



# %%
