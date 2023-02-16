# +
## Mesh refinement ... 

import os
os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import petsc4py
from petsc4py import PETSc

from underworld3 import timing
from underworld3 import adaptivity

import underworld3 as uw
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = True


# +
# Earth-like ratio of inner to outer
r_o = 1.0
r_i = 0.547
res = 500 / 6730 

mesh0 = uw.meshing.SphericalShell(radiusOuter=r_o, 
                           radiusInner=r_i, 
                           cellSize=res,
                           filename="tmp_low_r.msh")

phi = uw.discretisation.MeshVariable("\phi", mesh0, 1)
U = uw.discretisation.MeshVariable("U", mesh0, mesh0.dim, degree=2)
H = uw.discretisation.MeshVariable("H", mesh0, 1)

# Add a swarm to this mesh

swarm = uw.swarm.Swarm(mesh=mesh0)
theta_s = uw.swarm.SwarmVariable(r"\theta", swarm, proxy_degree=1, num_components=1)
swarm.populate(fill_param=2)
# -

for v in mesh0.vars:
    print(v)

# +
with mesh0.access(H, phi):
    H.data[:,0] = uw.function.evaluate(1.0 + 2000.0 * mesh0.CoordinateSystem.N[1]**2, mesh0.data, mesh0.N)
    phi.data[:,0] = uw.function.evaluate(sympy.cos(mesh0.CoordinateSystem.R[2]), mesh0.data, mesh0.N)
    
with swarm.access(theta_s):
    theta_s.data[:,0] = uw.function.evaluate(sympy.cos(mesh0.CoordinateSystem.R[1]), swarm.particle_coordinates.data, mesh0.N)


# +
# # Needs some validation ... 

# def dm_stack_bcs(dm, boundaries, stacked_bc_label_name):
    
#     if boundaries is None:
#         return
        
#     dm.removeLabel(stacked_bc_label_name)
#     dm.createLabel(stacked_bc_label_name)
#     stacked_bc_label = dm.getLabel(stacked_bc_label_name)
    
#     for b in boundaries:
#         bc_label_name = b.name
#         lab = dm.getLabel(bc_label_name)
        
#         if not lab:
#             continue 
            
#         lab_is = lab.getStratumIS(b.value)
    
#         # Load this up on the stack
#         stacked_bc_label.setStratumIS(b.value, lab_is)

        
# def dm_unstack_bcs(dm, boundaries, stacked_bc_label_name):
#     '''Unpack boundary labels to the list of names'''
    
#     if boundaries is None:
#         return
    
#     stacked_bc_label = dm.getLabel(stacked_bc_label_name)
#     vals = stacked_bc_label.getNonEmptyStratumValuesIS().getIndices()
        
#     for v in vals:
#         try:
#             b = boundaries(v)  # ValueError if mismatch
#         except ValueError:
#             continue
                
#         dm.removeLabel(b.name)
#         dm.createLabel(b.name)
#         b_dmlabel = dm.getLabel(b.name)
        
#         lab_is = stacked_bc_label.getStratumIS(v)
#         b_dmlabel.setStratumIS(v, lab_is)

#     return
        
        
# def mesh_adapt_meshVar(mesh, meshVarH):
        
#     # Create / use a field on the old mesh to hold the metric
#     # Perhaps that should be a user-definition
    
#     boundaries = mesh.boundaries
    
#     field_id = None
#     for i in range(mesh.dm.getNumFields()):
#         f,_ = mesh.dm.getField(i)
#         if f.getName() == "AdaptationMetricField":
#             field_id = i
            
#     if field_id is None:
#         field_id = mesh0.dm.getNumFields()
        
#     hvec = meshVarH._lvec
#     metric_vec = mesh.dm.metricCreateIsotropic(hvec, field_id)
#     f, _ = mesh.dm.getField(field_id)
#     f.setName("AdaptationMetricField")
    
#     dm_stack_bcs(mesh.dm, boundaries, "CombinedBoundaries")
#     dm_a = mesh.dm.adaptMetric(metric_vec, bdLabel="CombinedBoundaries")
#     dm_stack_bcs(dm_a, boundaries, "CombinedBoundaries")

#     meshA = uw.meshing.Mesh(dm_a,
#                             simplex = mesh.dm.isSimplex,
#                             coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
#                             qdegree = mesh.qdegree,
#                             refinement = None,
#                             refinement_callback = mesh.refinement_callback,
#                             boundaries = mesh.boundaries,                           
#                            )
    

#     return meshA
    
        

# +
# This is how we adapt the mesh

meshA = adaptivity.mesh_adapt_meshVar(mesh0, H)

# Add the variables we need to carry over 
phiA = uw.discretisation.MeshVariable("\phi^A", meshA, 1)
UA = uw.discretisation.MeshVariable("U^A", meshA, meshA.dim, degree=2)

# +
# We switch the swarm to the new mesh

swarm.mesh = meshA
# -



# +
# Interpolate mesh variables

with meshA.access(phiA):
    phiA.data[:] = phi.rbf_interpolate(meshA.data)
# -

# Any solvers ? they will also need rebuilding


# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh0.vtk("tmp_meshball0.vtk")
    pvmesh0 = pv.read("tmp_meshball0.vtk")

    meshA.vtk("tmp_meshball.vtk")
    pvmeshA = pv.read("tmp_meshball.vtk")
    
    pvmeshA.points *= 0.999
    



# +
with mesh0.access():
    pvmesh0.point_data["phiA"] = phi.data.copy()

with meshA.access():
    pvmeshA.point_data["phiA"] = phiA.data.copy()

with meshA.access():
    pvmeshA.point_data["thetaS"] = theta_s._meshVar.data.copy()


# +
pl = pv.Plotter(window_size=[1000, 1000])
pl.add_axes()

pl.add_mesh(
    pvmeshA, 
    cmap="coolwarm",
    edge_color="Black",
    style="surface",
    scalars = "thetaS",
    show_edges=True,
)

# pl.add_mesh(
#     pvmesh0, 
#     edge_color="Blue",
#     style="surface",
#     color="Blue", 
#     scalars = "phiA",
#     render_lines_as_tubes=True,
# )


# pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
# OR
pl.show(cpos="xy")
# -




