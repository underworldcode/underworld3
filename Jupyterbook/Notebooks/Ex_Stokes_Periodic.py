# %% [markdown]
# # Periodic Mesh Example

# %% [markdown]
# ## Generate Periodic mesh using GMSH

# %%
import gmsh
gmsh.initialize()

# %%
gmsh.model.add("Periodic x")

# %%
minCoords = (0., 0.)
maxCoords = (1.0, 1.0)
cellSize = 0.1

# %%
boundaries = {
   "Bottom": 1,
   "Top": 2,
   "Right": 3,
   "Left": 4,
}

# %%
xmin, ymin = minCoords
xmax, ymax = maxCoords

# %%
p1 = gmsh.model.geo.add_point(xmin,ymin,0., meshSize=cellSize)
p2 = gmsh.model.geo.add_point(xmax,ymin,0., meshSize=cellSize)
p3 = gmsh.model.geo.add_point(xmin,ymax,0., meshSize=cellSize)
p4 = gmsh.model.geo.add_point(xmax,ymax,0., meshSize=cellSize)

l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])

cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
surface = gmsh.model.geo.add_plane_surface([cl])

# %%
gmsh.model.geo.synchronize()

# %%
translation = [1, 0, 0, 1, 
               0, 1, 0, 0,
               0, 0, 1, 0, 
               0, 0, 0, 1]

# %%
gmsh.model.mesh.setPeriodic(1, [boundaries["Right"]], [boundaries["Left"]], translation)

# %%
# Add Physical groups
for name, tag in boundaries.items():
    gmsh.model.add_physical_group(1, [tag] , tag)
    gmsh.model.set_physical_name(1, tag, name)

gmsh.model.addPhysicalGroup(2, [surface], surface)
gmsh.model.setPhysicalName(2, surface, "Elements")

# %%
gmsh.model.mesh.generate(2) 
gmsh.write("periodicx.msh")
gmsh.finalize()

# %% [markdown]
# ## Import Mesh into PETSc

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
options["ksp_rtol"] =  1.0e-3 # For demonstration purposes only 1.0e-3 is OK
options["ksp_monitor_short"] = None
# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_max_it"] = 1
options["pc_type"] = "fieldsplit"
options["pc_fieldsplit_type"] = "schur"
options["pc_fieldsplit_schur_factorization_type"] ="full"
options["pc_fieldsplit_schur_precondition"] = "a11"
options["fieldsplit_velocity_pc_type"] = "lu"
options["fieldsplit_pressure_ksp_rtol"] = 1.0e-3
options["fieldsplit_pressure_pc_type"] = "lu"

# %%
plex = PETSc.DMPlex().createFromFile('periodicx.msh')

# %%
for name, tag in boundaries.items():
    plex.createLabel(name)
    label = plex.getLabel(name)
    indexSet = plex.getStratumIS("Face Sets", tag)
    if indexSet:
        label.insertIS(indexSet, 1)
    else:
        plex.removeLabel(name)

plex.removeLabel("Face Sets") 

# %%
plex.view()

# %%
from underworld3.discretisation import Mesh

# %%
mesh = Mesh(plex, degree=2)

# %%
mesh.dm.view()

# %%
# Create Stokes object
stokes = Stokes(mesh,u_degree=2,p_degree=1)
# Constant visc
stokes.viscosity = 1.
# No slip boundary conditions
stokes.add_dirichlet_bc( (0.5,0.), ["Top"], (0,1) )
stokes.add_dirichlet_bc( (-0.5,0.), ["Bottom"], (0,1))

# %%
# Write density into a variable for saving
densvar = uw.discretisation.MeshVariable("density",mesh,1)
with mesh.access(densvar):
    densvar.data[:,0] = 1.0

# %%
# body force
import sympy


unit_rvec = mesh.rvec / sympy.sqrt(mesh.rvec.dot(mesh.rvec))
stokes.bodyforce = -unit_rvec*1.0
#stokes.bodyforce = mesh.N.i + mesh.N.j

# %%
# Solve time
stokes.solve()

# %%
stokes.u.degree

# %%
import os
os.makedirs("output",exist_ok=True)
savefile = "output/stokes_periodic_2d.h5" 
mesh.save(savefile)
stokes.u.save(savefile)
stokes.p.save(savefile)
densvar.save(savefile)
mesh.generate_xdmf(savefile)

# %% [markdown]
# ## Pyvista visualisation
#
# `Pyvista` is a python vtk toolkit with a working model that closely matches scripted `paraview` in jupyter. 

# %%
import numpy as np
import pyvista as pv
import vtk

pv.global_theme.background = 'white'
pv.global_theme.window_size = [1000, 500]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = 'pythreejs'
pv.global_theme.smooth_shading = True


# %%

# %%
import meshio, io

mesh.vtk("spheremesh.vtk")
    
umag = stokes.u.fn.dot(stokes.u.fn) # Stokes object was re-built

pv_vtkmesh = pv.UnstructuredGrid("spheremesh.vtk")
pv_vtkmesh.point_data['density'] = uw.function.evaluate(density,mesh.data)
pv_vtkmesh.point_data['umag']    = uw.function.evaluate(umag, mesh.data)
pv_vtkmesh.point_data['urange'] = 0.5 + 0.5 * np.minimum(1.0,pv_vtkmesh.point_data['umag'] / pv_vtkmesh.point_data['umag'].mean())


# %%
clipped = pv_vtkmesh.clip(normal=(1, 0, 0), invert=False)
contours = pv_vtkmesh.contour([1.0,5.0, 10.0], scalars="density")


# %%
pl = pv.Plotter()
pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, 
            scalars="density", opacity="urange", use_transparency=False)
pl.add_mesh(contours, opacity=0.5)

pl.show()

# %%
umag = uw.function.evaluate(umag, mesh.data)

# %%
umag.max()
