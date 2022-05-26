# +
from petsc4py import PETSc
import underworld3 as uw
import numpy as np

from collections import namedtuple
from enum import Enum
# -



# ## Some pygmsh tips, tricks and trips
#
# The documentation for `pygmsh` is very unhelpful but the code itself is nicely written and is part of the `meshio` family of codes. The best way to see how the interface works is to look at how the high-level structures in the geometry module are put together from the low level gmsh python interface. See [geometry.py](https://github.com/nschloe/pygmsh/blob/main/src/pygmsh/geo/geometry.py) for example.

# +

csize_local = 0.05
cell_size_lower = 0.05
cell_size_upper = 0.05
radius_inner = 0.4
radius_outer = 1.0

import pygmsh
import meshio

# Generate local mesh.
with pygmsh.geo.Geometry() as geom:
    geom.characteristic_length_max = csize_local

    inner  = geom.add_circle((0.0,0.0,0.0),radius_inner, make_surface=False, mesh_size=cell_size_lower)
    outer  = geom.add_circle((0.0,0.0,0.0),radius_outer, make_surface=False, mesh_size=cell_size_upper)
    domain = geom.add_circle((0.0,0.0,0.0),radius_outer*1.25, mesh_size=10*cell_size_upper, holes=[inner])
    
    for l in outer.curve_loop.curves:
        geom.in_surface(l, domain.plane_surface)
        
    geom.synchronize()
 
    # Numerical labels for these starts at one and increments in order
    # It is therefore helpful to create some ordered dictionary of labels while we
    # create this ... limitations on gmsh / dmplex mean we need to anticipate
    # special cases like corners where the boundaries will overlap
    
    # physical_label = namedtuple('Label', ('name', 'index') )
    physical_label_group = namedtuple('Group', ('name', 'labels') )
    
    # A better way here would be to create a label/group stack and 
    # a mechanism to harvest those into something the dmplex labelling
    # can digest

    label_groups = []
        
    # Example of a group: the upper boundary is 3 curves (by default) and we can 
    # label them separately, group them later.
    geom.add_physical(outer.curve_loop.curves[0], label="Upper_0")
    geom.add_physical(outer.curve_loop.curves[1], label="Upper_1")
    geom.add_physical(outer.curve_loop.curves[2], label="Upper_2")
    
    label_groups.append(physical_label_group("Upper", ["Upper_0","Upper_1","Upper_2"]))
    
    geom.add_physical(domain.curve_loop.curves, label="Celestial_Sphere")
 
    # This is not really needed in the label list - it's everything else
    geom.add_physical(domain.plane_surface, label="Everything")
        
    geom.generate_mesh(verbose=True)
    geom.save_geometry("ignore_test.msh")
    geom.save_geometry("ignore_test.vtk")
    
    meshio_mesh = meshio.read("ignore_test.msh")
    meshio_mesh.remove_lower_dimensional_cells()
    
    

# +
# check the mesh if in a notebook / serial

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = pv.read("ignore_test.msh")
    pvmesh.plot(show_edges=True)

# +
dm = PETSc.DMPlex().createFromFile("ignore_test.msh")
meshio_mesh2 = meshio.read("ignore_test.msh")
print(meshio_mesh2.field_data)

label_dict = meshio_mesh2.field_data
# -

for l in label_dict:
    print(l)
for g in label_groups:
    print(g.name, ":", g.labels)

# +
# Code to generate labels and label groups assuming the gmsh physical labels

label_dict = meshio_mesh.field_data

## Face Sets are boundaries defined by element surfaces (1d or 2d entities)
## Vertex Sets are discrete points 
## pygmsh interlaces their labelling so, we have to try each one.

for l in label_dict:
    dm.createLabel(str(l).encode('utf8'))
    label = dm.getLabel(str(l).encode('utf8'))
    
    indexSet = dm.getStratumIS("Face Sets", label_dict[l][0])
    if not indexSet: # try the other one 
        indexSet = dm.getStratumIS("Vertex Sets", label_dict[l][0])
     
    if indexSet:
        label.insertIS(indexSet, 1)
        
    indexSet.destroy()
    

## Groups

for g in label_groups:
    dm.createLabel(str(g.name).encode('utf8'))
    label = dm.getLabel(str(g.name).encode('utf8'))
    
    for l in label_dict:
        print("Looking for {} in {}".format(l,g.labels))
        if l in g.labels:
            indexSet = dm.getStratumIS("Face Sets", label_dict[l][0])
            if not indexSet: # try the other one 
                indexSet = dm.getStratumIS("Vertex Sets", label_dict[l][0])

            if indexSet:
                label.insertIS(indexSet, 1)

            indexSet.destroy()

# -

dm.view()

# +
# This should be the same thing but it is necessary to specify the groups of labels (by name)
# if you want to combine boundaries

uwmesh = uw.mesh.MeshFromGmshFile(dim=2, filename="ignore_test.msh", label_groups=label_groups, simplex=True)


# +
# This approach is used by the built in uw meshes which keep track of the labels etc,
# and just keep track of the label names (all you need to use these for bcs)

# uwmesh2 = uw.mesh.SphericalShell(dim=2, radius_inner=0.4, radius_outer=1.0, cell_size=0.05 )
# print("===========")
# print(uwmesh2.labels)

