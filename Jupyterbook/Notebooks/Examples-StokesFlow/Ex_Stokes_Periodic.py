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
minCoords = (0.0, 0.0)
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
p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])

cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
surface = gmsh.model.geo.add_plane_surface([cl])

# %%
gmsh.model.geo.synchronize()

# %%
translation = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

# %%
gmsh.model.mesh.setPeriodic(1, [boundaries["Right"]], [boundaries["Left"]], translation)

# %%
# Add Physical groups
for name, tag in boundaries.items():
    gmsh.model.add_physical_group(1, [tag], tag)
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


# %%
plex = PETSc.DMPlex().createFromFile("periodicx.msh")

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
mesh = Mesh(plex, degree=1)

# %%
mesh.dm.view()

# %%
# Create Stokes object

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity=1)

# No slip boundary conditions
stokes.add_dirichlet_bc((0.5, 0.0), ["Top"], (0, 1))
stokes.add_dirichlet_bc((-0.5, 0.0), ["Bottom"], (0, 1))

# %%
# Write density into a variable for saving
densvar = uw.discretisation.MeshVariable("density", mesh, 1)
with mesh.access(densvar):
    densvar.data[:, 0] = 1.0

# %%
# body force
import sympy

x, y = mesh.X

unit_rvec = mesh.rvec / sympy.sqrt(mesh.rvec.dot(mesh.rvec))
stokes.bodyforce = 0 * mesh.X  # -mesh.X / sympy.sqrt(x**2 + y**2)

# %%
# Solve time
stokes.solve()

# %%
import os

os.makedirs("output", exist_ok=True)
savefile = "output/stokes_periodic_2d.h5"
mesh.save(savefile)
stokes.u.save(savefile)
stokes.p.save(savefile)
densvar.save(savefile)
mesh.generate_xdmf(savefile)

# %%
# check if that works

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    pv.start_xvfb()

    # mesh.vtk("ignore_periodic_mesh.vtk")
    pvmesh = pv.read("periodicx.msh")

    # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

    with mesh.access():
        vsol = v.data.copy()

    arrow_loc = np.zeros((v.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v.coords[...]

    arrow_length = np.zeros((v.coords.shape[0], 3))
    arrow_length[:, 0:2] = vsol[...]

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Black", "wireframe")

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
    #               use_transparency=False, opacity=0.5)

    pl.add_arrows(arrow_loc, arrow_length, mag=5.0e-1, opacity=0.5)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")

# %%
