# %% [markdown]
"""
# 🎓 AnnulusFS

**PHYSICS:** free_surface  
**DIFFICULTY:** advanced  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# #!/usr/bin/env python
# coding: utf-8

## From Neng Lu


import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local

comm = uw.mpi.comm
rank = uw.mpi.rank
size = uw.mpi.size
if size == 1:
    import matplotlib.pyplot as plt

from underworld3.coordinates import CoordinateSystemType

# %%

# %%
u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

H = 3000.  * u.kilometer ### thickness of mantle r_o-r_i
velocity     = 1e-9 * u.meter / u.second
#g    =   10.0 * u.meter / u.second**2  
#bodyforce    = 3300  * u.kilogram / u.metre**3 * g 
mu           = 1e21  * u.pascal * u.second

KL = H
Kt = KL / velocity
KM = mu * KL * Kt

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM



# %%
def plot_mesh(title,mesh,showFig=True,color='Black'):
    import numpy as np
    import pyvista as pv
    import vtk
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    pl.add_mesh(pvmesh,color, 'wireframe')

    pl.add_title(title,font_size=11)
    if showFig:
        pl.show(cpos="xy")
    else:
        pl.camera_position = 'xy' 
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data() 
    pvmesh.clear_point_data()


def plot_stream(title,showFig=False):
    import numpy as np
    import pyvista as pv
    import vtk
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True
    
    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    #pl.add_mesh(pvmesh,'k', 'wireframe')
    
    dim = 2
    coords = pvmesh.points[:, 0:dim]
    vector_values = np.zeros_like(pvmesh.points)
    vector_values[:, 0:dim] = uw.function.evalf(v.sym, coords)
    pvmesh.point_data["V"] = vector_values
    pvmesh.point_data["P"] = uw.function.evalf(p.sym, coords)
    
    skip = 1
    points = np.zeros((mesh._centroids[::skip, 0].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points[::3])
    
    pvstream = pvmesh.streamlines_from_source(point_cloud,
                                                vectors="V",
                                                integrator_type=45,
                                                integration_direction="both",
                                                max_steps=1000,
                                                max_time=0.1,
                                                initial_step_length=0.001,
                                                max_step_length=0.01)
    
    pl.add_mesh(pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=False,
            scalars="P",
            use_transparency=False,
            opacity=1.0)
    pl.add_mesh(pvstream, line_width=1.0)
    pl.remove_scalar_bar("V")
    pl.remove_scalar_bar("P")
    
    
    #pl.add_title(title,font_size=11)
    
    if showFig:
        pl.show(cpos="xy")
    else:
        pl.camera_position = 'xy' 
    
    #pl.remove_scalar_bar("V")
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data() 
    pvmesh.clear_point_data()


# %%
# %%

# %%
def get_ra_analytical(x0,load_time):
    z = 0
    x = x0
    tau = (D*k+np.sinh(D*k)*np.cosh(D*k))/(np.sinh(D*k)**2)*tau0
    A = -F0/k/tau0
    B = -F0/k/tau
    C = F0/tau
    E = F0/tau/np.tanh(D*k)
    phi = np.sin(k*x)*np.exp(-tmax/tau)*(A*np.sinh(k*z)+B*np.cosh(k*z)+C*z*np.sinh(k*z)+E*z*np.cosh(k*z))

    F_t = F0*np.exp(-load_time/tau)
    ra_analytical_t = F_t*np.cos(k*x) 
    return  ra_analytical_t


# %%

# %%
render = True
use_fssa = True

r_o = ndim(6000*u.kilometer)
r_i = ndim(3000*u.kilometer)
timeratio = 1/16
res =64   # 8, 16, 32, 64, 128
cellsize =  2*np.pi*r_i/res                 #r_i/res


# parameters for initial topography
D = r_o*2*np.pi
ratio = 8

wavelength = D/ratio
k = 2.0 * np.pi / wavelength
mu0 = ndim(1e21  * u.pascal * u.second)
g = ndim(10.0 * u.meter / u.second**2 )
rho0 = ndim(4500  * u.kilogram / u.metre**3)
drho = rho0
F0 = ndim(300*u.kilometer)

tau0 = 2*k*mu0/drho/g
print(dim(tau0,u.kiloyear))
# print((ratio*4*np.pi*1e21/4500/3e6/10)/(365.25*24*60*60*1000))  #kiloyear
tmax = tau0*4


dt_set    = tau0*timeratio
max_time  = tmax #+ dt_set
save_every = 1 #int(tau0/dt_set/4)
if save_every < 1:
    save_every = int(1)

# q1dq1
pdegree, vdegree, pcontinuous= 1, 1, True
# q1dq0
#pdegree, vdegree, pcontinuous= 0, 1, False

if use_fssa:
    outputPath = "op_2DRelaxation_FreeSurfFSSA_Annulus" + "_res{:n}_dt{:.2f}ka_Tmax{:.1f}ka_wavel{}/".format(res,dim(dt_set,u.kiloyear).m,dim(max_time,u.kiloyear).m,ratio)
else:
    outputPath = "op_2DRelaxation_FreeSurf_Annulus" + "_res{:n}_dt{:.2f}ka_Tmax{:.1f}ka_wavel{}/".format(res,dim(dt_set,u.kiloyear).m,dim(max_time,u.kiloyear).m,ratio)
if uw.mpi.rank == 0:
    # delete previous model run
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+ i)
            
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

#print(ndim(max_time),ndim(dt_set),ndim(bodyforce))
print(outputPath)


# %%


mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=cellsize, qdegree=max(pdegree, vdegree), degree=1)
mesh0 = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=cellsize, qdegree=max(pdegree, vdegree), degree=1)
init_mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=cellsize, qdegree=max(pdegree, vdegree), degree=1)

botwall = petsc_dm_find_labeled_points_local(mesh.dm,"Lower")
topwall = petsc_dm_find_labeled_points_local(mesh.dm,"Upper")

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=vdegree,continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=pdegree,continuous=pcontinuous)
timeField     = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)


# %%

# %%
if render:

    thmin, thmax = -np.pi, np.pi
    th0 = np.linspace(thmin,thmax,res+1)

    x0 = th0*r_o
    y0 = F0*np.cos(k*x0) 

    y_analytical_dt = get_ra_analytical(x0,dt_set)
    y_analytical = get_ra_analytical(x0,tmax)

    fig, ax1 = plt.subplots(nrows=1, figsize=(10,5))
    ax1.set(xlabel=r'$\theta/2\pi$', ylabel=r'$(r-r_o)/F0$') 
    ax1.plot(th0/(2*np.pi),y0/F0,'r',label='Inital')
    ax1.plot(th0/(2*np.pi),y_analytical_dt/F0,'b',label=r'Analytical $Time ={}\tau_0$'.format(timeratio))
    ax1.plot(th0/(2*np.pi),y_analytical/F0,'g',label=r'Analytical $Time = 4\tau_0$')
    ax1.set_xlim([-0.5,0.5])
    ax1.legend(loc = 'upper left',prop = {'size':8})


# %%


if render:
    plot_mesh('mesh0',mesh,showFig=True)


# %%

# %%
ra, th = mesh.CoordinateSystem.xR

def perturbation_init(mesh):
    ra, th = mesh.CoordinateSystem.xR
    ra_new = ra+sympy.cos(th*ra*k)*F0
    return ra_new,th 

Rmesh = uw.discretisation.MeshVariable("Rmesh", init_mesh, 1, degree=1)
Bmesh = uw.discretisation.MeshVariable("Bmesh", init_mesh, 1, degree=1)

mesh_solver = uw.systems.Poisson(init_mesh, u_Field=Rmesh)
mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
mesh_solver.f = 0.0
mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Upper",0)
mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Lower",0)

_ra, _l1 = init_mesh.CoordinateSystem.xR
_ra_new, _l1_new = perturbation_init(init_mesh)
with init_mesh.access(Bmesh):
    #Bmesh.data[:,0] = uw.function.evaluate(_ra,init_mesh.X.coords)
    Bmesh.data[botwall,0] = uw.function.evaluate(_ra,init_mesh.X.coords[botwall])
    Bmesh.data[topwall,0] = uw.function.evaluate(_ra_new,init_mesh.X.coords[topwall])
mesh_solver.solve()

def update_mesh():
    with init_mesh.access(Rmesh):
        new_mesh_coords = init_mesh.X.coords
        new_mesh_th = uw.function.evaluate(init_mesh.CoordinateSystem.xR[1],init_mesh.X.coords)
        new_mesh_coords[:,0] = Rmesh.data[:,0]*np.cos(new_mesh_th)
        new_mesh_coords[:,1] = Rmesh.data[:,0]*np.sin(new_mesh_th)
    return new_mesh_coords
new_mesh_coords = update_mesh()
mesh._deform_mesh(new_mesh_coords)

if render:
    plot_mesh('mesh1',mesh,showFig=True,color='red')


# %%

# %%
if render:

    import numpy as np
    import pyvista as pv
    import vtk

    title = "Mesh0vsMesh1"
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True
    
    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    pl.add_mesh(pvmesh,'red', 'wireframe',label='mesh1')
    
    mesh0.vtk("tmp_box_mesh0.vtk")
    pvmesh0 = pv.read("tmp_box_mesh0.vtk")
    pl.add_mesh(pvmesh0,'Black', 'wireframe',label='mesh0')

    # init_mesh.vtk("tmp_box_mesh_init.vtk")
    # pvmesh_init = pv.read("tmp_box_mesh_init.vtk")
    # pl.add_mesh(pvmesh_init,'b', 'wireframe',label='mesh_init')
    
    pl.add_title(title,font_size=11)
    pl.add_legend(bcolor='w',loc='upper right',face=None)
    pl.show(cpos="xy")
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data() 
    pvmesh.clear_point_data()


# %%

# %%
if render:
    top_x0 = mesh0.data[topwall,0]
    top_y0 = mesh0.data[topwall,1]
    top_l0 = uw.function.evaluate(mesh0.CoordinateSystem.xR[1],mesh0.data[topwall])
    top_r0 = uw.function.evaluate(mesh0.CoordinateSystem.xR[0],mesh0.data[topwall])
    
    top_x1 = mesh.X.coords[topwall,0]
    top_y1 = mesh.X.coords[topwall,1]
    top_l1 = uw.function.evaluate(mesh.CoordinateSystem.xR[1],mesh.X.coords[topwall])
    top_r1 = uw.function.evaluate(mesh.CoordinateSystem.xR[0],mesh.X.coords[topwall])
    
    fname = r"Mesh0Mesh1_xy"
    fig, ax1 = plt.subplots(nrows=1, figsize=(5,5))
    ax1.set(xlabel=r'x coordinate$', ylabel='y coordinates') 
    x0 = top_x0[np.argsort(top_l0)] 
    y0 = top_y0[np.argsort(top_l0)] 
    ax1.plot(x0,y0,'k',label='mesh0')
    
    x1 = top_x1[np.argsort(top_l1)] 
    y1 = top_y1[np.argsort(top_l1)] 
    ax1.plot(x1,y1,'red',label='mesh1')
    ax1.legend(loc = 'upper left',prop = {'size':8})
    plt.savefig(fname+'.png',dpi=150,bbox_inches='tight')


# %%

# %%
top_r0 = top_r0[np.argsort(top_l0)] 
top_r1 = top_r1[np.argsort(top_l1)]

top_l0 = top_l0[np.argsort(top_l0)] 
top_l1 = top_l1[np.argsort(top_l1)]

fname = r"Mesh0Mesh1_th_r"
fig, ax1 = plt.subplots(nrows=1, figsize=(10,5))
ax1.set(xlabel=r'$\theta/2\pi$', ylabel=r'$r-r_o$') 
ax1.plot(top_l0/(2*np.pi),top_r0-r_o,'k',label='mesh0')
ax1.plot(top_l1/(2*np.pi),top_r1-r_o,'red',label='mesh1')
ax1.set_xlim([-0.5,0.5])
ax1.legend(loc = 'upper left',prop = {'size':8})
plt.savefig(fname+'.png',dpi=150,bbox_inches='tight')


# %%

# %%
# swarm  = uw.swarm.Swarm(mesh)
# material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=1, proxy_degree=1)  
# fill_parameter= 1 # swarm fill parameter
# swarm.populate_petsc(fill_param=fill_parameter,layout=uw.swarm.SwarmPICLayout.GAUSS)

# MIndex = 0
# with swarm.access(material):
#     material.data[:] = MIndex

# density_fn = material.createMask([densityM])
# visc_fn = material.createMask([viscM])

densityM = rho0
viscM = mu0
ND_gravity = ndim(g)

density_fn = densityM
visc_fn = viscM

ra_fn = sympy.sqrt(mesh.rvec.dot(mesh.rvec))
unit_rvec = mesh.X/(ra_fn)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.bodyforce = -1 * ND_gravity * density_fn* unit_rvec
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.shear_viscosity_0
stokes.add_dirichlet_bc((0.0,0.0), "Lower", (0,1))

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = 1.0e-9
stokes.petsc_options["ksp_rtol"] = 1.0e-9
stokes.petsc_options["ksp_atol"] = 1.0e-9
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None


# %%

# %%
# ## Test

# stokes.solve(zero_init_guess=False,_force_setup=True)
# from scipy.interpolate import interp1d
# from scipy.interpolate import CloughTocher2DInterpolator
# top = topwall
# _dt = dt_set

# coords = mesh.X.coords[top]
# x = coords[:,0]
# y = coords[:,-1]
# vx = uw.function.evalf(v.sym[0], coords)
# vy = uw.function.evalf(v.sym[1], coords)

# # Advect top surface
# x2 = x + vx * _dt
# y2 = y + vy * _dt

# mask2 = np.where(x2!=0) 
# th2 = np.zeros_like(x2)
# th2[mask2] = np.arctan2(y2[mask2],x2[mask2])
# ra2 = np.sqrt(x2**2 + y2**2)

# mask = np.where(x!=0) 
# th = np.zeros_like(x)
# th[mask] = np.arctan2(y[mask],x[mask])
# ra = np.sqrt(x**2 + y**2)

# # Spline top surface
# f = interp1d(th2, ra2, kind='cubic', fill_value='extrapolate')
# ra_new = f(th)   

# ra_sol = ra_new[np.argsort(th)]  
# th_sol = th[np.argsort(th)]  

# ra_sol1 = ra2[np.argsort(th2)]  
# th_sol1 = th2[np.argsort(th2)]

# fname = r"Mesh_th_r_dt"
# fig, ax1 = plt.subplots(nrows=1, figsize=(10,5))
# ax1.set(xlabel=r'$\theta/2\pi$', ylabel='r') 
# #ax1.plot(top_l0/(2*np.pi),top_r0,'k',label='mesh0')
# ax1.plot(th0/(2*np.pi),y0,'--k',label='Inital')
# ax1.plot(th0/(2*np.pi),y_analytical_dt,'b',label=r'Analytical $Time ={}\tau_0$'.format(timeratio))
# ax1.plot(th_sol/(2*np.pi),ra_sol ,'red',label='Numerical')
# #ax1.plot(th_sol1/(2*np.pi),ra_sol1 ,'--y',label='Numerical1')
# ax1.set_xlim([-0.5,0.5])
# ax1.legend(loc = 'upper left',prop = {'size':8})
# plt.savefig(fname+'.png',dpi=150,bbox_inches='tight')


# %%

# %%

# %%
if use_fssa:
    theta = 0.5*density_fn*ND_gravity*dt_set
    #Gamma = mesh.CoordinateSystem.unit_e_0
    #Gamma = mesh.Gamma
    Gamma = mesh.Gamma / sympy.sqrt(mesh.Gamma.dot(mesh.Gamma))
    FSSA_traction = theta*Gamma.dot(v.sym) * Gamma
    stokes.add_natural_bc(FSSA_traction, "Upper")


# %%

# %%
def _adjust_time_units(val):
    """ Adjust the units used depending on the value """
    if isinstance(val, u.Quantity):
        mag = val.to(u.years).magnitude
    else:
        val = dim(val, u.years)
        mag = val.magnitude
    exponent = int("{0:.3E}".format(mag).split("E")[-1])

    if exponent >= 9:
        units = u.gigayear
    elif exponent >= 6:
        units = u.megayear
    elif exponent >= 3:
        units = u.kiloyears
    elif exponent >= 0:
        units = u.years
    elif exponent > -3:
        units = u.days
    elif exponent > -5:
        units = u.hours
    elif exponent > -7:
        units = u.minutes
    else:
        units = u.seconds
    return val.to(units)


# %%

# %%
from scipy.interpolate import interp1d
from scipy.interpolate import CloughTocher2DInterpolator

class FreeSurfaceProcessor_Annulus(object): 
    def __init__(self,mesh,v,dt):
        """
        Parameters
        ----------
        _init_mesh : the original mesh
        mesh : the updating model mesh
        vel : the velocity field of the model
        dt : dt for advecting the surface
        """
        self.init_mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=cellsize, qdegree=max(pdegree, vdegree), degree=1)
        self.Rmesh = uw.discretisation.MeshVariable("Rmesh", self.init_mesh, 1, degree=1)
        self.Bmesh = uw.discretisation.MeshVariable("Bmesh", self.init_mesh, 1, degree=1)
        self.mesh_solver = uw.systems.Poisson(self.init_mesh , u_Field=self.Rmesh)
        self.mesh_solver.constitutive_model = uw.constitutive_models.DiffusionModel
        self.mesh_solver.constitutive_model.Parameters.diffusivity = 1. 
        self.mesh_solver.f = 0.0
        self.mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Upper",0)
        self.mesh_solver.add_dirichlet_bc(Bmesh.sym[0], "Lower",0)
        
        self.v = v
        self._dt   = dt

        self.bot0 = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Lower")
        self.top0 = petsc_dm_find_labeled_points_local(self.init_mesh.dm,"Upper")

        self.mesh = mesh
        self.bot = petsc_dm_find_labeled_points_local(self.mesh.dm,"Lower")
        self.top = petsc_dm_find_labeled_points_local(self.mesh.dm,"Upper")
        
    def _solve_sle(self):
        self.mesh_solver.solve()
     
    def _advect_surface(self):
        if self.top.size > 0:
            coords = self.mesh.X.coords[self.top]
            x = coords[:,0]
            y = coords[:,-1]
            vx = uw.function.evalf(self.v.sym[0], coords)
            vy = uw.function.evalf(self.v.sym[1], coords)
            
            # Advect top surface
            x2 = x + vx * self._dt
            y2 = y + vy * self._dt

            mask2 = np.where(x2!=0) 
            th2 = np.zeros_like(x2)
            th2[mask2] = np.arctan2(y2[mask2],x2[mask2])
            ra2 = np.sqrt(x2**2 + y2**2)

            mask = np.where(x!=0) 
            th = np.zeros_like(x)
            th[mask] = np.arctan2(y[mask],x[mask])
            ra = np.sqrt(x**2 + y**2)
    
            # Spline top surface
            f = interp1d(th2, ra2, kind='cubic', fill_value='extrapolate')

            with self.init_mesh.access(self.Bmesh):
                self.Bmesh.data[self.bot0, 0] = uw.function.evaluate(self.init_mesh.CoordinateSystem.xR[0],self.init_mesh.X.coords[self.bot0])
                self.Bmesh.data[self.top0, 0] = f(th)      
        uw.mpi.barrier()
        #self.Bmesh.syncronise()

    def _update_mesh(self):
        with self.init_mesh.access():
            new_mesh_coords = self.init_mesh.X.coords
            new_mesh_l1 = uw.function.evaluate(self.init_mesh.CoordinateSystem.xR[1],self.init_mesh.X.coords)
            new_mesh_coords[:,0] = self.Rmesh.data[:,0]*np.cos(new_mesh_l1)
            new_mesh_coords[:,1] = self.Rmesh.data[:,0]*np.sin(new_mesh_l1)
        return new_mesh_coords
    
    def solve(self):
        self._advect_surface()
        self._solve_sle()
        new_mesh_coords = self._update_mesh()
        return new_mesh_coords


# %%

# %%
step      = 0
max_steps = 2
time      = 0
dt        = 0

# if rank == 0:
#     fw = open(outputPath + "ParticlePosition.txt","w")
#     fw.write("Time \t W \t dWdT \n")
#     fw.close()
# uw.mpi.barrier()
w = []
dwdt = []
times = []

while time < (max_time+ dt_set):
#while step < max_steps:
    
    if uw.mpi.rank == 0:
        string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
        step, _adjust_time_units(time),
        _adjust_time_units(dt),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.stdout.write(string)
        sys.stdout.flush()
    
    #stokes = build_stokes_solver(mesh,v,p)
    #stokes.solve(zero_init_guess=False)
    stokes.solve(zero_init_guess=False,_force_setup=True)


    if step%save_every ==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data:')
        with mesh.access(timeField):
            timeField.data[:,0] = dim(time, u.megayear).m

        mesh.petsc_save_checkpoint(meshVars=[v, p, timeField], index=step, outputPath=outputPath)
        #swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputPath) 
        #interfaceSwarm.petsc_save_checkpoint(swarmName='interfaceSwarm', index=step, outputPath=outputPath) 
        #plot_mesh('mesh_step{}'.format(step),mesh,showFig=False)
        title = "Fig_stream_step{}".format(step)
        plot_stream(title,showFig=False)
        
    times.append(time)
    dt_solver = stokes.estimate_dt()
    dt = min(dt_solver,dt_set)
    
    #swarm.advection(V_fn=stokes.u.sym, delta_t=dt)
    #interfaceSwarm.advection(V_fn=stokes.u.sym, delta_t=dt)

    freesuface = FreeSurfaceProcessor_Annulus(mesh,v,dt)
    new_mesh_coords=freesuface.solve()
    mesh._deform_mesh(new_mesh_coords)
    # if uw.mpi.rank == 0:
    #     print(f'\nrepopulate start:')
    # pop_control.repopulate(mesh,material)
    # if uw.mpi.rank == 0:
    #     print(f'\nrepopulate end:')
    #repopulate(swarm,mesh,updateField=material)
    #pop_control.redistribute(material)
    #pop_control.repopulate(material)

    step += 1
    time += dt


# %%

# %%
top_x2 = mesh.X.coords[topwall,0]
top_y2 = mesh.X.coords[topwall,1]
top_l2 = uw.function.evaluate(mesh.CoordinateSystem.xR[1],mesh.X.coords[topwall])
top_r2 = uw.function.evaluate(mesh.CoordinateSystem.xR[0],mesh.X.coords[topwall])

ra_sol = top_r2[np.argsort(top_l2)]  
th_sol = top_l2[np.argsort(top_l2)]  


ra_analytical_t = get_ra_analytical(th_sol*r_o,time)

fname = r"Mesh_th_r_dt"
fig, ax1 = plt.subplots(nrows=1, figsize=(10,5))
ax1.set(xlabel=r'$\theta/2\pi$', ylabel=r'(r-r_o)/F0')
#ax1.plot(top_l0/(2*np.pi),top_r0,'k',label='mesh0')
#ax1.plot(th0/(2*np.pi),y0,'-k',label='Inital')
ax1.plot(th_sol/(2*np.pi),ra_analytical_t/F0,'k',label=r'Analytical $dt ={}\tau_0$'.format(timeratio))
ax1.plot(th_sol/(2*np.pi),(ra_sol -r_o)/F0,'--r',label='Numerical')
#ax1.plot(th_sol1/(2*np.pi),ra_sol1 ,'--y',label='Numerical1')
ax1.set_xlim([-0.5,0.5])
ax1.legend(loc = 'upper left',prop = {'size':8})
plt.savefig(outputPath+fname+'.png',dpi=150,bbox_inches='tight')


# %%


# #!pip install msgpack


# %%





# %%
