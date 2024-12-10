#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld3 as uw
import numpy as np
import sympy

import pyvista as pv
import underworld3.visualisation as vis


# In[2]:


from posixpath import pardir
import petsc4py.PETSc as PETSc

import os
import warnings
from typing import Optional, Tuple

import underworld3 as uw
from underworld3.utilities._api_tools import Stateful
from underworld3.utilities._api_tools import uw_object

import underworld3.timing as timing
from underworld3.swarm import SwarmVariable

class IndexSwarmVariable_B(SwarmVariable):
    """
    The IndexSwarmVariable is a class for managing material point
    behaviour. The material index variable is rendered into a
    collection of masks each representing the extent of one material
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        name,
        swarm,
        indices=1,
        npoints=5,
        proxy_degree=1,
        proxy_continuous=True,
    ):
        self.indices = indices
        self.nnn = npoints

        # These are the things we require of the generic swarm variable type
        super().__init__(
            name,
            swarm,
            size=1,
            vtype=None,
            dtype=int,
            _proxy=False,
        )
        """
        vtype = (None,)
        dtype = (float,)
        proxy_degree = (1,)
        proxy_continuous = (True,)
        _register = (True,)
        _proxy = (True,)
        _nn_proxy = (False,)
        varsymbol = (None,)
        rebuild_on_cycle = (True,)
        """
        # The indices variable defines how many "level set" maps we create as components in the proxy variable

        import sympy

        self._MaskArray = sympy.Matrix.zeros(1, self.indices)
        self._meshLevelSetVars = [None] * self.indices

        for i in range(indices):
            self._meshLevelSetVars[i] = uw.discretisation.MeshVariable(
                name + R"^{[" + str(i) + R"]}",
                self.swarm.mesh,
                num_components=1,
                degree=proxy_degree,
                continuous=proxy_continuous,
            )
            self._MaskArray[0, i] = self._meshLevelSetVars[i].sym[0, 0]

        return

    # This is the sympy vector interface - it's meaningless if these are not spatial arrays
    @property
    def sym(self):
        return self._MaskArray

    @property
    def sym_1d(self):
        return self._MaskArray

    # We can  also add a __getitem__ call to access each mask

    def __getitem__(self, index):
        return self.sym[index]

    def createMask(self, funcsList):
        """
        This creates a masked sympy function of swarm variables required for Underworld's solvers
        """

        if not isinstance(funcsList, (tuple, list)):
            raise RuntimeError("Error input for createMask() - wrong type of input")

        if len(funcsList) != self.indices:
            raise RuntimeError("Error input for createMask() - wrong length of input")

        symo = sympy.simplify(0)
        for i in range(self.indices):
            symo += funcsList[i] * self._MaskArray[i]

        return symo

    def visMask(self):
        return self.createMask(list(range(self.indices)))

    def view(self):
        """
        Show information on IndexSwarmVariable
        """
        if uw.mpi.rank == 0:
            print(f"IndexSwarmVariable {self}")
            print(f"Numer of indices {self.indices}")

    def _update(self):
        kd = uw.kdtree.KDTree(self._meshLevelSetVars[0].coords)
        kd.build_index()
        
        with self.swarm.access():
            n_indices, n_distance = kd.find_closest_n_points(self.nnn,self.swarm.particle_coordinates.data)
            kd_swarm = uw.kdtree.KDTree(self.swarm.particle_coordinates.data)
            kd_swarm.build_index()
            n, d, b = kd_swarm.find_closest_point(self._meshLevelSetVars[0].coords)
    
        for ii in range(self.indices):
            meshVar = self._meshLevelSetVars[ii]
        
            with self.swarm.mesh.access(meshVar), self.swarm.access():
                node_values = np.zeros((meshVar.data.shape[0],))
                w = np.zeros((meshVar.data.shape[0],))
                
                for i in range(self.data.shape[0]):
                    tem = np.isclose(n_distance[i,:],n_distance[i,0])
                    dist = n_distance[i,tem]
                    indices = n_indices[i,tem]
                    for j,ind in enumerate(indices):
                        node_values[ind] += (np.isclose(self.data[i], ii) /(1.0e-16 + dist[j]))[0]
                        w[ind] +=  1.0 / (1.0e-16 + dist[j])
            
                node_values[np.where(w > 0.0)[0]] /= w[np.where(w > 0.0)[0]]  
                meshVar.data[:,0] = node_values[...]

                # if there is no material found, 
                # impose a near-neighbour hunt for a valid material and set that one 
            
                if np.where(w == 0.0)[0].shape[0] > 0:
                    if self.data[n[np.where(w == 0.0)]]==ii:
                        meshVar.data[np.where(w == 0.0)] = 1.0
        return
        
class IndexSwarmVariable_C(SwarmVariable):
    """
    The IndexSwarmVariable is a class for managing material point
    behaviour. The material index variable is rendered into a
    collection of masks each representing the extent of one material
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        name,
        swarm,
        indices=1,
        npoints=4,
        npoints_bc=2,
        ind_bc = None,
        proxy_degree=1,
        proxy_continuous=True,
    ):
        self.indices = indices
        self.nnn = npoints
        self.nnn_bc = npoints_bc 
        self.ind_bc = ind_bc

        # These are the things we require of the generic swarm variable type
        super().__init__(
            name,
            swarm,
            size=1,
            vtype=None,
            dtype=int,
            _proxy=False,
        )
        """
        vtype = (None,)
        dtype = (float,)
        proxy_degree = (1,)
        proxy_continuous = (True,)
        _register = (True,)
        _proxy = (True,)
        _nn_proxy = (False,)
        varsymbol = (None,)
        rebuild_on_cycle = (True,)
        """
        # The indices variable defines how many "level set" maps we create as components in the proxy variable

        import sympy

        self._MaskArray = sympy.Matrix.zeros(1, self.indices)
        self._meshLevelSetVars = [None] * self.indices

        for i in range(indices):
            self._meshLevelSetVars[i] = uw.discretisation.MeshVariable(
                name + R"^{[" + str(i) + R"]}",
                self.swarm.mesh,
                num_components=1,
                degree=proxy_degree,
                continuous=proxy_continuous,
            )
            self._MaskArray[0, i] = self._meshLevelSetVars[i].sym[0, 0]

        return

    # This is the sympy vector interface - it's meaningless if these are not spatial arrays
    @property
    def sym(self):
        return self._MaskArray

    @property
    def sym_1d(self):
        return self._MaskArray

    # We can  also add a __getitem__ call to access each mask

    def __getitem__(self, index):
        return self.sym[index]

    def createMask(self, funcsList):
        """
        This creates a masked sympy function of swarm variables required for Underworld's solvers
        """

        if not isinstance(funcsList, (tuple, list)):
            raise RuntimeError("Error input for createMask() - wrong type of input")

        if len(funcsList) != self.indices:
            raise RuntimeError("Error input for createMask() - wrong length of input")

        symo = sympy.simplify(0)
        for i in range(self.indices):
            symo += funcsList[i] * self._MaskArray[i]

        return symo

    def visMask(self):
        return self.createMask(list(range(self.indices)))

    def view(self):
        """
        Show information on IndexSwarmVariable
        """
        if uw.mpi.rank == 0:
            print(f"IndexSwarmVariable {self}")
            print(f"Numer of indices {self.indices}")

    def _update(self):
        with self.swarm.access():
            kd = uw.kdtree.KDTree(self.swarm.particle_coordinates.data)
            n_indices, n_distance = kd.find_closest_n_points(self.nnn,self._meshLevelSetVars[0].coords)
            
        for ii in range(self.indices):
            meshVar = self._meshLevelSetVars[ii]
            with self.swarm.mesh.access(meshVar), self.swarm.access():
                node_values = np.zeros((meshVar.data.shape[0],))
                w = np.zeros((meshVar.data.shape[0],))
                for i in range(meshVar.data.shape[0]):
                    # for mesh nodes not on the boundary, cal from nearest N particles
                    if i not in ind_bc:
                       a =  1.0 / (n_distance[i,:]+1.0e-16)
                       w[i] = np.sum(a)
                       b = np.isclose(self.data[n_indices[i,:]], ii)
                       node_values[i] = np.sum(np.dot(a,b))
                    # for mesh nodes on the boundary, cal from nearest N_bc particles
                    else:
                       a =  1.0 / (n_distance[i,:self.nnn_bc]+1.0e-16)
                       w[i] = np.sum(a)
                       b = np.isclose(self.data[n_indices[i,:self.nnn_bc]], ii)
                       node_values[i] = np.sum(np.dot(a,b))
            
                node_values[np.where(w > 0.0)[0]] /= w[np.where(w > 0.0)[0]]  
                meshVar.data[:,0] = node_values[...]
        return


# In[3]:


xmin, xmax = -1,1
ymin, ymax = -1,1
xres,yres = 2,2
dx = (xmax-xmin)/xres
dy = (ymax-ymin)/yres

ppdegree = 1
ppcont = True

# fill_params = [1,2,3,4,5]
# for fill_param in fill_params:
fill_param = 2 
print(fill_param)

mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))
npointsB = 4
npointsC,npointsC_bc = 4,2

#mesh = uw.meshing.UnstructuredSimplexBox(cellSize=dx,  minCoords=(xmin, ymin), maxCoords=(xmax, ymax),regular=False,refinement=0)
#npointsB = 3
#npointsC = 5
#npointsC_bc=3

from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local
ind_bc = petsc_dm_find_labeled_points_local(mesh.dm,'All_Boundaries')
Pmesh = uw.discretisation.MeshVariable("P", mesh, 1, degree=ppdegree,continuous=ppcont)

swarm = uw.swarm.Swarm(mesh)
materialA = uw.swarm.IndexSwarmVariable("MA", swarm, indices=2, proxy_degree=ppdegree,proxy_continuous=ppcont) 
materialB = IndexSwarmVariable_B("MB", swarm, indices=2, proxy_degree=ppdegree,proxy_continuous=ppcont,npoints=npointsB)
materialC = IndexSwarmVariable_C("MC", swarm, indices=2, proxy_degree=ppdegree,proxy_continuous=ppcont,ind_bc=ind_bc,npoints=npointsC,npoints_bc=npointsC_bc)

swarm.populate(fill_param=fill_param)

amplitude, offset, wavelength= 0.5, 0., 1
k = 2.0 * np.pi / wavelength
interfaceSwarm = uw.swarm.Swarm(mesh)
npoints = 101
x = np.linspace(mesh.data[:,0].min(), mesh.data[:,0].max(), npoints)
y = offset + amplitude * np.cos(k * x)
interface_coords = np.ascontiguousarray(np.array([x,y]).T)
interfaceSwarm.add_particles_with_coordinates(interface_coords)

M0Index = 0
M1Index = 1
with swarm.access(materialA):
    perturbation = offset + amplitude * np.cos(k * swarm.particle_coordinates.data[:, 0])+0.01
    materialA.data[:, 0] = np.where(swarm.particle_coordinates.data[:, 1] <= perturbation, M0Index, M1Index)
with swarm.access(materialB):
    materialB.data[:, 0] = np.where(swarm.particle_coordinates.data[:, 1] <= perturbation, M0Index, M1Index)
with swarm.access(materialC):
    materialC.data[:, 0] = np.where(swarm.particle_coordinates.data[:, 1] <= perturbation, M0Index, M1Index)

P0, P1 = 1,10
P_fnA = materialA.createMask([P0,P1])
P_fnB = materialB.createMask([P0,P1])
P_fnC = materialC.createMask([P0,P1])

### for test 
### compare the value on the Symmetrical Point on the left and riht wall
# with mesh.access(Bmesh):
#     Bmesh.data[:,0] = uw.function.evaluate(B_fn,Bmesh.coords)
#     assert np.allclose(Bmesh.data[0],Bmesh.data[1], atol=0.01)
#     assert np.allclose(Bmesh.data[2],Bmesh.data[3], atol=0.01)
#     assert np.allclose(Bmesh.data[6],Bmesh.data[7], atol=0.01)


# In[4]:


def plot_mesh_var(title,mesh,var,var_fn,swarm,material,line_coords):
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pv.global_theme.background = "white"
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    pl = pv.Plotter(window_size=(600, 600))
    pl.add_mesh(pvmesh,'Black', 'wireframe')

    with mesh.access():
        pvmesh.point_data["B"] = uw.function.evaluate(var_fn,var.coords)
    
    labels = []
    for i in range(pvmesh.points.shape[0]):
        labels.append(str(np.round(pvmesh.point_data["B"][i],2)))
    pl.add_point_labels(pvmesh,labels,italic=True,font_size=20,point_color='black',
                        point_size=3,render_points_as_spheres=True,always_visible=True,shadow=False)
    pl.add_points(pvmesh, cmap="RdBu_r", scalars='B',render_points_as_spheres=True,
                    use_transparency=False, opacity=0.95, point_size= 15)

    points = np.column_stack((line_coords[:,0], line_coords[:,1], np.zeros_like(line_coords[:,0])))
    pl.add_lines(points, color='black', width=3, connected=True)

    pvswarm = vis.swarm_to_pv_cloud(swarm)
    with swarm.access():
        pvswarm.point_data["MIndex"] = material.data.copy()
    pl.add_points(pvswarm,scalars="MIndex",cmap="coolwarm",render_points_as_spheres=True,opacity=0.95,point_size=15,)    
    pl.remove_scalar_bar("MIndex")
    
    pl.add_title(title,font_size=11)
    pl.show(cpos="xy")
    pl.screenshot(title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data()
    pvmesh.clear_point_data()
    pvswarm.clear_data()
    pvswarm.clear_point_data()

def plot_meshnode_swarm_index(title,mesh,swarm,line_coords,add_label=None,add_label_color=None,material=None):
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pv.global_theme.background = "white"
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    pl = pv.Plotter(window_size=(600, 600))
    pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    labels = []
    for i in range(pvmesh.points.shape[0]):
        labels.append(str(np.round(i,2)))
    pl.add_point_labels(pvmesh,labels,italic=True,font_size=20,point_color='grey',
                        point_size=15,render_points_as_spheres=True,always_visible=True,shadow=False,shape=None)
    
    pvswarm = vis.swarm_to_pv_cloud(swarm)
    labels = []
    for i in range(pvswarm.points.shape[0]):
        labels.append(str(np.round(i,2)))
    pl.add_point_labels(pvswarm,labels,italic=True,font_size=15,point_color='grey',text_color = 'black',
                        point_size=10,render_points_as_spheres=True,always_visible=True,shadow=False,shape=None)

    if material is not None:
        with swarm.access():
            pvswarm.point_data["MIndex"] = material.data.copy()
        pl.add_points(pvswarm,scalars="MIndex",cmap="coolwarm",render_points_as_spheres=True,opacity=0.95,point_size=15,)    
        pl.remove_scalar_bar("MIndex")

    if add_label is not None:
        for ii,tem in enumerate(add_label):
            labels = []
            with swarm.access():
                points = pvswarm.points[tem]
                for i in range(len(tem)):
                    labels.append(str(tem[i]))
            pl.add_point_labels(points,labels,italic=True,font_size=15,point_color=add_label_color[ii],text_color = 'black',
                                point_size=15,render_points_as_spheres=True,always_visible=True,shadow=False,shape=None)

    points = np.column_stack((line_coords[:,0], line_coords[:,1], np.zeros_like(line_coords[:,0])))
    pl.add_lines(points, color='black', width=3, connected=True)

    pl.add_title(title,font_size=11)
    pl.show(cpos="xy")
    pl.screenshot(title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data()
    pvmesh.clear_point_data()
    pvswarm.clear_data()
    pvswarm.clear_point_data()


# In[5]:


plot_mesh_var('P_MA',mesh,Pmesh,P_fnA,swarm,materialA,interface_coords)


# In[6]:


plot_mesh_var('P_MB',mesh,Pmesh,P_fnB,swarm,materialB,interface_coords)


# In[7]:


plot_mesh_var('P_MC',mesh,Pmesh,P_fnC,swarm,materialC,interface_coords)


# In[8]:


plot_meshnode_swarm_index('MeshNode&Swarm_Index',mesh,swarm,interface_coords,material=materialA)


# In[9]:


# Method A
# origin code from uw.swarm.IndexSwarmVariable._update
self = materialA
kd = uw.kdtree.KDTree(self._meshLevelSetVars[0].coords)
kd.build_index()

ii = 0
tem6 = []
tem7 = []

meshVar = self._meshLevelSetVars[ii]
with self.swarm.mesh.access(meshVar), self.swarm.access():
    n, d, b = kd.find_closest_point(self.swarm.data)
    for i in range(self.data.shape[0]):
        if b[i]:
            if n[i] == 6:
                tem6.append(i)
            if n[i] == 7:
                tem7.append(i)

plot_meshnode_swarm_index('MA_Index',mesh,swarm,interface_coords,add_label=[tem6,tem7],add_label_color=['blue','red'])


# In[10]:


# Method B
self = materialB
kd = uw.kdtree.KDTree(self._meshLevelSetVars[0].coords)
kd.build_index()
with self.swarm.access():
    n_indices, n_distance = kd.find_closest_n_points(self.nnn,self.swarm.particle_coordinates.data)

ii = 0
tem6 = []
tem7 = []
meshVar = self._meshLevelSetVars[ii]
with self.swarm.mesh.access(meshVar), self.swarm.access():
    for i in range(self.data.shape[0]):
        tem = np.isclose(n_distance[i,:],n_distance[i,0])
        dist = n_distance[i,tem]
        indices = n_indices[i,tem]
        for j,ind in enumerate(indices):
            if ind == 6:
                tem6.append(i)
            if ind == 7:
                tem7.append(i)

plot_meshnode_swarm_index('MB_Index',mesh,swarm,interface_coords,add_label=[tem6,tem7],add_label_color=['blue','red'])


# In[11]:


# Method C
self = materialC
with self.swarm.access():
    kd = uw.kdtree.KDTree(self.swarm.particle_coordinates.data)
    n_indices, n_distance = kd.find_closest_n_points(self.nnn,self._meshLevelSetVars[0].coords)
    
ii = 0
tem6 = []
tem7 = []
meshVar = self._meshLevelSetVars[ii]
with self.swarm.mesh.access(meshVar), self.swarm.access():
    for i in range(meshVar.data.shape[0]):
        if i == 6:
            tem6 = n_indices[i,:self.nnn_bc]
        if i == 7:
            tem7 = n_indices[i,:self.nnn_bc]

plot_meshnode_swarm_index('MC_Index',mesh,swarm,interface_coords,add_label=[tem6,tem7],add_label_color=['blue','red'])


# In[ ]:




