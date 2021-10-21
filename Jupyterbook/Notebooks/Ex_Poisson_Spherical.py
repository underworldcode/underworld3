# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np
options = PETSc.Options()
# options["pc_type"]  = "svd"
options["ksp_rtol"] = 1.0e-7
# options["ksp_monitor_short"] = None
# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
options["snes_rtol"] = 1.0e-7

# %%
# Set some things
k = 1. 
h = 0.
t_i = 2.
t_o = 1.
r_i = 0.5
r_o = 1.0

# %%
# first do 2D
cell_size=0.02
mesh = uw.mesh.SphericalShell(dim=2,radius_inner=r_i, radius_outer=r_o,cell_size=cell_size)
# Create Poisson object
poisson = Poisson(mesh)
poisson.k = k
poisson.h = h

# %%
import sympy
abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))
bc = sympy.Piecewise( ( t_i,  abs_r < 0.5*(r_i+r_o) ),
                      ( t_o,                 True ) )
poisson.add_dirichlet_bc( bc, mesh.boundary.ALL_BOUNDARIES )

# %%
bc

# %%
# Solve time
poisson.solve()

# %%
# Check. Construct simple solution for above config.
import math
A = (t_i-t_o)/(math.log(r_i)-math.log(r_o))
B = t_o - A*math.log(r_o)
r = np.linalg.norm(poisson.u.coords,axis=1)
sol = A*np.log(r) + B
import numpy as np
with mesh.access():
    if not np.allclose(sol,poisson.u.data[:,0],rtol=5e-04):
        raise RuntimeError("Unexpected values encountered.")

# %%
savefile = "output/poisson_spherical_2d.h5" 
mesh.save(savefile)
poisson.u.save(savefile)
mesh.generate_xdmf(savefile)

# %%
import k3d
import plot
vertices_2d = plot.mesh_coords(mesh)
vertices = np.zeros((vertices_2d.shape[0],3),dtype=np.float32)
vertices[:,0:2] = vertices_2d[:]
indices = plot.mesh_faces(mesh)
kplot = k3d.plot()
with mesh.access():
    kplot += k3d.mesh(vertices, indices,attribute=poisson.u.data, color_map=k3d.basic_color_maps.BlackBodyRadiation,wireframe=True)
kplot.grid_visible=False
kplot.display()
kplot.camera = [-0.2, 0.2, 2.0,0.,0.,0.,-0.5,1.0,-0.1]  # these are some adhoc settings

# %%
# now do 3D
cell_size=0.025
mesh = uw.mesh.SphericalShell(dim=3,radius_inner=r_i, radius_outer=r_o,cell_size=cell_size)
# Create Poisson object
poisson = Poisson(mesh)
poisson.k = k
poisson.h = h

# %%
import sympy
abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))
bc = sympy.Piecewise( ( t_i,  abs_r < 0.5*(r_i+r_o) ),
                      ( t_o,                 True ) )
poisson.add_dirichlet_bc( bc, mesh.boundary.ALL_BOUNDARIES )

# %%
bc

# %%
# Solve time
poisson.solve()

# %%
# Check. Construct simple solution for above config.
import math
A = (t_i-t_o)/(1./r_i-1./r_o)
B = t_o - A/r_o
r = np.linalg.norm(poisson.u.coords,axis=1)
sol = A/r + B
import numpy as np
with mesh.access():
    if not np.allclose(sol,poisson.u.data[:,0],rtol=2e-02):
        # abs_diff = np.abs(sol-poisson.u.data[:,0])
        # argmax = abs_diff.argmax()
        # print(argmax,(sol[argmax],poisson.u.data[argmax,0]))
        raise RuntimeError("Unexpected values encountered.")

# %%
vertices = np.array(plot.mesh_coords(mesh),dtype=np.float32)
indices = plot.mesh_faces(mesh)
kplot = k3d.plot()
with mesh.access():
    kplot += k3d.mesh(vertices, indices,attribute=poisson.u.data, color_map=k3d.basic_color_maps.BlackBodyRadiation,wireframe=True)
kplot.display()

# %%
savefile = "output/poisson_spherical_3d.h5" 
mesh.save(savefile)
poisson.u.save(savefile)
mesh.generate_xdmf(savefile)
